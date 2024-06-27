from attributes_to_language.composer import Composer, VariantsT
from attributes_to_language.types import ComputedAttributeT, VariantT

from simple_shapes_dataset.text.writers import writers


def a_has_n(sentence: str):
    def aux(attributes: ComputedAttributeT | None) -> str:
        vowels = ["a", "e", "i", "o", "u"]
        if attributes is None:
            return sentence.format(**{"n?": ""})
        if attributes[attributes["_next"]][0] in vowels:
            return sentence.format(**{"n?": "n"})
        return sentence.format(**{"n?": ""})

    return aux


# or from a kwargs given to the composer method.
script_structures_simple = [
    "{start} <{size}>{link}<{color}>{link}<{shape}>{link}<{location}>{"
    "link}<{rotation}>.",
]

# Elements in the list of each variant is randomly chosen.
variants_simple: VariantsT = {
    "start": [
        "",
        "It is",
        "A kind of",
        "This is",
        "There is",
        "The image is",
        "The image represents",
        "The image contains",
    ],
    "link": [". It is ", ", and is ", " ", ", it's ", ". It's "],
}

random_composer = Composer(script_structures_simple, writers, variants_simple)

script_structures = [
    "{start} {size} {colorBefore} {shape}{link} <{location}>{link} <{" "rotation}>.",
    "{start} {size} {colorBefore} {object} <{location}>{link} <{rotation}>{"
    "link2} {a} {shape}.",
    "{start} {size} {object} <{location}>{link} <{rotation}>{link} {"
    "colorBoth}{link2} {a} {shape}.",
    "{start} {size} {object} in {colorAfter} <{location}>{link} <{"
    "rotation}>{link2} {a} {shape}.",
    "{start} {shape}{link} {size}{link3} {colorBoth}{link} <{location}>{"
    "link} <{rotation}>.",
    "{start} {color}{colored?} {size} {shape}{link} <{location}>{link} <{"
    "rotation}>.",
    "{start} {color}{colored?} {shape}{link} <{size}>{link} <{location}>{"
    "link} <{rotation}>.",
]

pre_start_variant = [
    "A",
    "It is a",
    "This is a",
    "There is a",
    "The image is a",
    "The image represents a",
    "The image contains a",
]
start_variant: list[VariantT] = [a_has_n(x + "{n?}") for x in pre_start_variant] + [
    "A kind of"
]

# Elements in the list of each variant is randomly chosen.
variants: VariantsT = {
    "start": start_variant,
    "a": [a_has_n("a{n?}")],
    "object": ["shape", "object"],
    "colored?": ["", " colored"],
    "colorBefore": ["{color}", "{color} colored"],
    "colorAfter": ["{color}", "{color} color"],
    "colorBoth": [
        "{color}",
        "in {color}",
        "{color} colored",
        "in {color} color",
    ],
    ",?": [", ", " "],
    "link": [
        ". It is",
        ", it is",
        "{,?}and is",
        "{,?}and it is",
        ",",
    ],
    "link2": [
        ". It looks like",
        ", it looks like",
        "{,?}and looks like",
        "{,?}and it looks like",
    ],
    "link3": [
        ". It is",
        ", it is",
        "{,?}and is",
        "{,?}and it is",
        ",",
        " and",
    ],
}

composer = Composer(script_structures, writers, variants)
