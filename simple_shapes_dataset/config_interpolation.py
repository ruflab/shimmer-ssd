from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, model_validator


@dataclass
class ParsingContext:
    in_interpolation: bool = False
    is_escaped: bool = False
    interpolation_key: str = ""


def dict_get_from_key_seq(
    dotlist: Sequence[str],
    data: Any,
    full_key: str | None = None,
) -> str:
    if full_key is None:
        full_key = ".".join(dotlist)
    if len(dotlist) == 0:
        raise KeyError(f"{full_key} cannot be interpolated because does not exist.")

    key = dotlist[0]
    if isinstance(data, Sequence):
        if not dotlist[0].isdigit():
            raise KeyError(f"{dotlist[0]} should be an int when data is a sequence")

    if len(dotlist) == 1 and isinstance(data, Mapping):
        return str(data[key])
    elif len(dotlist) == 1 and isinstance(data, Sequence):
        return str(data[int(key)])

    elif isinstance(data, Mapping):
        return dict_get_from_key_seq(dotlist[1:], data[key], full_key)
    elif isinstance(data, Sequence):
        return dict_get_from_key_seq(dotlist[1:], data[int(key)], full_key)
    # Should never happen
    raise NotImplementedError


def interpolate(
    query: str,
    data: Mapping[str, Any] | Sequence[Any],
    context: ParsingContext | None = None,
) -> str:
    if not len(query):
        return ""

    if context is None:
        context = ParsingContext()

    letter, rest = query[0], query[1:]
    match letter:
        case "\\" if not context.is_escaped and not context.in_interpolation:
            return interpolate(
                rest,
                data,
                ParsingContext(
                    context.in_interpolation,
                    is_escaped=True,
                    interpolation_key=context.interpolation_key,
                ),
            )
        case "{" if not context.is_escaped:
            return interpolate(
                rest,
                data,
                ParsingContext(
                    in_interpolation=True, is_escaped=False, interpolation_key=""
                ),
            )
        case "}" if context.in_interpolation and not context.is_escaped:
            interpolated = dict_get_from_key_seq(
                context.interpolation_key.split("."), data
            )
            return interpolated + interpolate(
                rest,
                data,
                ParsingContext(
                    in_interpolation=False, is_escaped=False, interpolation_key=""
                ),
            )
        case x if context.in_interpolation:
            return interpolate(
                rest,
                data,
                ParsingContext(
                    in_interpolation=True,
                    is_escaped=False,
                    interpolation_key=context.interpolation_key + x,
                ),
            )
        case "{" | "}" | "\\" if context.is_escaped:
            return letter + interpolate(
                rest,
                data,
                ParsingContext(
                    in_interpolation=False, is_escaped=False, interpolation_key=""
                ),
            )
        case x if context.is_escaped:
            return (
                "\\"
                + letter
                + interpolate(
                    rest,
                    data,
                    ParsingContext(
                        in_interpolation=False, is_escaped=False, interpolation_key=""
                    ),
                )
            )
        case x:
            return letter + interpolate(
                rest,
                data,
                ParsingContext(
                    in_interpolation=False, is_escaped=False, interpolation_key=""
                ),
            )


def interpolate_seq(
    data: Sequence[Any], base_data: Mapping[str, Any] | Sequence[Any]
) -> Sequence[Any]:
    new_data: list[Any] = []
    for val in data:
        if isinstance(val, str) and "{" in val and "}" in val:
            new_data.append(interpolate(val, base_data))
        elif isinstance(val, dict):
            new_data.append(interpolate_map(val, base_data))
        elif isinstance(val, list):
            new_data.append(interpolate_seq(val, base_data))
        else:
            new_data.append(val)
    return new_data


def interpolate_map(
    data: Mapping[str, Any], base_data: Mapping[str, Any] | Sequence[Any]
) -> dict[str, Any]:
    new_data: dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, str) and "{" in val and "}" in val:
            new_data[key] = interpolate(val, base_data)
        elif isinstance(val, dict):
            new_data[key] = interpolate_map(val, base_data)
        elif isinstance(val, list):
            new_data[key] = interpolate_seq(val, base_data)
        else:
            new_data[key] = val
    return new_data


class InterpolationModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def interpolate_variables(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            return interpolate_map(data, data)
        elif isinstance(data, Sequence):
            return interpolate_seq(data, data)
        else:
            print(type(data))
            return data
