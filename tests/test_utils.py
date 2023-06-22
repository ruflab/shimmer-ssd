from simple_shapes_dataset.cli.utils import get_deterministic_name


def test_get_deterministic_name():
    domain_alignment = {frozenset(["v"]): 0.2, frozenset(["t"]): 0.3}
    seed = 3
    name = get_deterministic_name(domain_alignment, seed)

    assert name == "t:0.3_v:0.2_seed:3"


def test_get_deterministic_name_mutiple_domain():
    domain_alignment = {frozenset(["v"]): 0.2, frozenset(["t", "v"]): 0.3}
    seed = 3
    name = get_deterministic_name(domain_alignment, seed)

    assert name == "t,v:0.3_v:0.2_seed:3"


def test_get_deterministic_name_mutiple_domain_different_order():
    domain_alignment = {frozenset(["v"]): 0.2, frozenset(["v", "t"]): 0.3}
    seed = 3
    name = get_deterministic_name(domain_alignment, seed)
    assert name == "t,v:0.3_v:0.2_seed:3"
