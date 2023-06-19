from simple_shapes_dataset.cli.utils import get_deterministic_name


def test_get_deterministic_name():
    domain_alignment = [("v", 0.2), ("t", 0.3)]
    seed = 3
    name = get_deterministic_name(domain_alignment, seed)

    assert name == "t:0.3_v:0.2_seed:3"


def test_get_deterministic_name_mutiple_domain():
    domain_alignment = [("v", 0.2), ("t,v", 0.3)]
    seed = 3
    name = get_deterministic_name(domain_alignment, seed)

    assert name == "t,v:0.3_v:0.2_seed:3"


def test_get_deterministic_name_mutiple_domain_wrong_order():
    domain_alignment = [("v", 0.2), ("v,t", 0.3)]
    seed = 3
    name = get_deterministic_name(domain_alignment, seed)
    assert name == "t,v:0.3_v:0.2_seed:3"


def test_get_deterministic_name_mutiple_domain_repeated():
    domain_alignment = [("v", 0.2), ("t,v", 0.3), ("v,t", 0.1)]
    seed = 3
    name = get_deterministic_name(domain_alignment, seed)

    assert name == "t,v:0.1_v:0.2_seed:3"
