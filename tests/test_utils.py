import numpy as np

from simple_shapes_dataset.cli.utils import get_deterministic_name, get_domain_alignment


def test_get_deterministic_name():
    domain_alignment = {frozenset(["v"]): 0.2, frozenset(["t"]): 0.3}
    seed = 3
    max_size = 10
    name = get_deterministic_name(domain_alignment, seed, max_size)

    assert name == "t:0.3_v:0.2_seed:3_ms:10"


def test_get_deterministic_name_mutiple_domain():
    domain_alignment = {frozenset(["v"]): 0.2, frozenset(["t", "v"]): 0.3}
    seed = 3
    max_size = 10
    name = get_deterministic_name(domain_alignment, seed, max_size)

    assert name == "t,v:0.3_v:0.2_seed:3_ms:10"


def test_get_deterministic_name_mutiple_domain_different_order():
    domain_alignment = {frozenset(["v"]): 0.2, frozenset(["v", "t"]): 0.3}
    seed = 3
    max_size = 10
    name = get_deterministic_name(domain_alignment, seed, max_size)
    assert name == "t,v:0.3_v:0.2_seed:3_ms:10"


def test_get_domain_alignment_split():
    dataset_size = 100
    domain_groups = get_domain_alignment(
        seed=0,
        allowed_indices=np.arange(dataset_size),
        alignment_groups_props={
            frozenset(["v"]): 0.8,
            frozenset(["t"]): 0.9,
            frozenset(["a"]): 1.0,
            frozenset(["v", "t"]): 0.4,
            frozenset(["t", "a"]): 0.2,
        },
    )

    for domain_group in domain_groups:
        assert domain_group in [
            frozenset(["v"]),
            frozenset(["t"]),
            frozenset(["a"]),
            frozenset(["v", "t"]),
            frozenset(["t", "a"]),
        ]
    assert domain_groups[frozenset(["v"])].shape[0] == 80
    assert (
        np.intersect1d(
            domain_groups[frozenset(["v", "t"])],
            domain_groups[frozenset(["v"])],
            assume_unique=True,
        ).shape[0]
        == 40
    )
