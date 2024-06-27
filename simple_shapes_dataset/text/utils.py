import math
from itertools import permutations
from typing import Any

from attributes_to_language.composer import Composer

from simple_shapes_dataset.dataset.domain import Choice


def inspect_writers(composer: Composer) -> dict[str, int]:
    choices: dict[str, int] = {}
    for writer_name, writers in composer.writers.items():
        if len(writers) > 1:
            choices[f"writer_{writer_name}"] = len(writers)
        for k, writer in enumerate(writers):
            for variant_name, variant in writer.variants.items():
                if len(variant) > 1:
                    choices[f"writer_{writer_name}_{k}_{variant_name}"] = len(variant)
    return choices


def inspect_all_choices(composer: Composer) -> dict[str, int]:
    choices: dict[str, int] = {}
    choices["structure"] = sum(math.factorial(len(group)) for group in composer.groups)

    for variant_name, variant in composer.variants.items():
        if len(variant) > 1:
            choices[f"variant_{variant_name}"] = len(variant)

    choices.update(inspect_writers(composer))
    return choices


def structure_category_from_choice(
    composer: Composer, choice: Choice
) -> dict[str, int]:
    categories: dict[str, int] = {}
    # structure
    class_val = 0
    for k, groups in enumerate(composer.groups):
        if choice.structure != k:
            class_val += math.factorial(len(groups))
        else:
            for i, permutation in enumerate(permutations(range(len(groups)))):
                if choice.groups == list(permutation):
                    categories["structure"] = class_val
                    class_val += len(groups) - i
                    break
                class_val += 1
    # variants
    for name in composer.variants:
        categories[f"variant_{name}"] = choice.variants.get(name, 0)
    # writers
    for name in inspect_writers(composer):
        split_name = name.split("_")
        writer_name = split_name[1]
        categories[name] = 0
        if len(split_name) == 2:
            categories[name] = choice.writers[writer_name]["_writer"]
        elif writer_name in choice.writers:
            variant_name = split_name[3]
            variant_choice = int(split_name[2])
            if (
                variant_name in choice.writers[writer_name]
                and choice.writers[writer_name]["_writer"] == variant_choice
            ):
                categories[name] = choice.writers[writer_name][variant_name]
    return categories


def choices_from_structure_categories(
    composer: Composer, grammar_predictions: dict[str, list[int]]
) -> list[dict[str, Any]]:
    all_choices: list[dict[str, Any]] = []
    for i in range(len(grammar_predictions["structure"])):
        choices: dict[str, Any] = {
            "variants": {
                name.replace("variant_", ""): variant[i]
                for name, variant in grammar_predictions.items()
                if "variant_" in name
            },
            "writers": {},
        }
        # writers
        for name, variant in grammar_predictions.items():
            if "writer_" in name:
                split_name = name.split("_")
                writer_name = split_name[1]
                if writer_name not in choices["writers"]:
                    choices["writers"][writer_name] = {}
                if len(split_name) == 2:
                    choices["writers"][writer_name]["_writer"] = variant[i]
                else:
                    variant_choice = int(split_name[2])
                    if (
                        grammar_predictions[f"writer_{writer_name}"][i]
                        == variant_choice
                    ):
                        variant_name = split_name[3]
                        choices["writers"][writer_name][variant_name] = variant[i]
        # structure
        category = grammar_predictions["structure"][i]
        for k, groups in enumerate(composer.groups):
            if category < math.factorial(len(groups)):
                choices["structure"] = k
                choices["groups"] = list(
                    list(permutations(range(len(groups))))[category]
                )
                all_choices.append(choices)
                break
            category -= math.factorial(len(groups))
    return all_choices
