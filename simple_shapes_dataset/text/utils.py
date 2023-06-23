import math
import re
from typing import Any

from attributes_to_language.composer import Composer


def inspect_writers(composer):
    choices = dict()
    for writer_name, writers in composer.writers.items():
        if len(writers) > 1:
            choices[f"writer_{writer_name}"] = len(writers)
        for k, writer in enumerate(writers):
            for variant_name, variant in writer.variants.items():
                if len(variant) > 1:
                    choices[f"writer_{writer_name}_{k}_{variant_name}"] = len(
                        variant
                    )
    return choices


def inspect_all_choices(composer: Composer) -> dict[str, Any]:
    num_structures = 0
    choices = dict()
    for structure in composer.script_structures:
        num_structures += math.factorial(
            len(re.findall(r"<[^>]+>", structure))
        )
    choices["structure"] = num_structures
    for variant_name, variant in composer.variants.items():
        if len(variant) > 1:
            choices[f"variant_{variant_name}"] = len(variant)
    choices.update(inspect_writers(composer))
    return choices
