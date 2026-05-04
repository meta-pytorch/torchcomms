# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pytest

from hatchet.frame import Frame


def test_constructors():
    f1 = Frame({"name": "foo", "file": "bar.c"})
    f2 = Frame(name="foo", file="bar.c")
    f3 = Frame({"name": "foo"}, file="bar.c")
    f4 = Frame({"name": "foo", "file": "baz.h"}, file="bar.c")

    assert f1 == f2 == f3 == f4


def test_no_attrs():
    with pytest.raises(ValueError):
        Frame({})

    with pytest.raises(ValueError):
        Frame()


def test_comparison():
    assert Frame(a=1) == Frame(a=1)

    assert Frame(a=1) < Frame(a=2)
    assert Frame(a=1) != Frame(a=2)
    assert Frame(a=2) > Frame(a=1)

    assert Frame(a=1) < Frame(b=1)
    assert Frame(a=1) != Frame(b=1)
    assert Frame(b=1) > Frame(a=1)

    assert Frame(a=1, b=1) < Frame(b=1, c=1)
    assert Frame(a=1, b=1) != Frame(b=1, c=1)
    assert Frame(b=1, c=1) > Frame(a=1, b=1)


def test_copy():
    f = Frame(a=1)
    assert f == f.copy()
    assert not (f != f.copy())
    assert not (f > f.copy())
    assert not (f < f.copy())

    f = Frame(a="foo", b="bar", c="baz")
    assert f == f.copy()
    assert not (f != f.copy())
    assert not (f > f.copy())
    assert not (f < f.copy())


def test_getitem():
    f = Frame(a=1, b=2, c=3)
    assert f["a"] == 1
    assert f["b"] == 2
    assert f["c"] == 3

    assert f.get("a") == 1
    assert f.get("b") == 2
    assert f.get("c") == 3

    assert f.get("d") is None
    assert f.get("d", "foo") == "foo"

    with pytest.raises(KeyError):
        assert f["d"]


def test_values():
    f = Frame(a=1, b=2, c=3)
    assert f.values("a") == 1
    assert f.values(["b"]) == (2,)
    assert f.values(("a", "a", "a")) == (1, 1, 1)
    assert f.values(["a", "b"]) == (1, 2)
    assert f.values(["a", "b", "c"]) == (1, 2, 3)
    assert f.values(["b", "c"]) == (2, 3)
    assert f.values("c") == 3

    assert f.values(["foo", "a", "bar", "c"]) == (None, 1, None, 3)


def test_repr():
    assert (
        repr(Frame(foo="baz", bar="quux"))
        == "Frame({'bar': 'quux', 'foo': 'baz', 'type': 'None'})"
    )


def test_str():
    assert (
        str(Frame(foo="baz", bar="quux"))
        == "{'bar': 'quux', 'foo': 'baz', 'type': 'None'}"
    )
