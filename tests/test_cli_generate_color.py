import numpy as np

from simple_shapes_dataset.cli.utils import ShapeDependentColorSampler


def test_generate_color_dragon():
    n_samples = 4
    sampled_attrs = {"classes": np.array([0, 1, 0, 2])}
    color_ranges = {
        0: (-30, 30),
        1: (30, 90),
        2: (90, 150),
    }
    sampler = ShapeDependentColorSampler(color_ranges)
    rgb, hls = sampler(n_samples, sampled_attrs)
    assert -1 not in hls
    assert -1 not in rgb
