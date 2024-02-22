import pytest

from simple_shapes_dataset.config import interpolate


def test_interpolation():
    data = {"a": "baz"}
    query = "foo bar {a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar baz"


def test_interpolation_several():
    data = {"a": "baz", "b": "test"}
    query = "foo bar {a} {b}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar baz test"


def test_interpolation_several_same():
    data = {"a": "baz"}
    query = "foo bar {a}{a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar bazbaz"


def test_interpolation_int():
    data = {"a": "1"}
    query = "foo bar {a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar 1"


def test_interpolation_nested():
    data = {"a": {"b": "baz"}}
    query = "foo bar {a.b}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar baz"


def test_interpolation_escaped():
    data = {"a": "baz"}
    query = "foo bar \\{test\\} {a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar {test} baz"


def test_interpolation_escaped_in_interpolation():
    data = {"\\a": "baz"}
    query = "foo bar {\\a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar baz"


def test_interpolation_escaped_2():
    data = {"a": "baz"}
    query = "foo bar \\{test} {a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar {test} baz"


def test_interpolation_other():
    data = {"a": "baz"}
    query = "foo bar \\n {a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar \\n baz"


def test_interpolation_other_2():
    data = {"a": "baz"}
    query = "foo bar \\\\\\{test} {a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar \\{test} baz"


def test_interpolation_other_3():
    data = {"a": "baz"}
    query = "foo bar \\\\{a} {a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar \\baz baz"


def test_interpolation_other_4():
    data = {"a": "baz"}
    query = "foo bar \\\\a {a}"
    interpolated = interpolate(query, data)
    assert interpolated == "foo bar \\a baz"


def test_interpolation_missing():
    data = {"a": "baz"}
    query = "foo bar {b}"
    with pytest.raises(ValueError):
        interpolate(query, data)


def test_interpolation_missing_nested():
    data = {"a": {"b": "baz"}}
    query = "foo bar {b}"
    with pytest.raises(ValueError):
        interpolate(query, data)


def test_interpolation_missing_nested_2():
    data = {"a": {"b": "baz"}}
    query = "foo bar {a.c}"
    with pytest.raises(ValueError):
        interpolate(query, data)
