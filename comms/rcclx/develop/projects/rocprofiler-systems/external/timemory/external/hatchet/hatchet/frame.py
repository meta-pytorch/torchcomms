# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from functools import total_ordering


@total_ordering
class Frame:
    """The frame index for a node. The node only stores its frame.

    Arguments:
       attrs (dict): dictionary of attributes and values
    """

    def __init__(self, attrs=None, **kwargs):
        """Construct a frame from a dictionary, or from immediate kwargs.

        Arguments:
            attrs (dict, optional): dictionary of attributes for this Frame

        Keyword arguments are optional, but if they are provided, they
        will be used to update the dictionary.  Keys in kwargs take
        precedence over anything in the attrs dictionary.

        So, these are all functionally equivalent::

            Frame({"name": "foo", "file": "bar.c"})
            Frame(name="foo", file="bar.c")
            Frame({"name": "foo"}, file="bar.c")
            Frame({"name": "foo", "file": "baz.h"}, file="bar.c")

        """
        # attributes dictionary
        self.attrs = attrs if attrs else {}

        # add keyword arguments, if any.
        if kwargs:
            self.attrs.update(kwargs)

        if not self.attrs:
            raise ValueError("Frame must be constructed with attributes!")

        # add type to frame if type is not in the attributes dict or kwargs
        if "type" not in self.attrs:
            self.attrs["type"] = "None"

        self._tuple_repr = None

    def __eq__(self, other):
        return self.tuple_repr == other.tuple_repr

    def __lt__(self, other):
        return self.tuple_repr < other.tuple_repr

    def __gt__(self, other):
        return self.tuple_repr > other.tuple_repr

    def __hash__(self):
        return hash(self.tuple_repr)

    def __str__(self):
        """str() with sorted attributes, so output is deterministic."""
        return "{%s}" % ", ".join("'%s': '%s'" % (k, v) for k, v in self.tuple_repr)

    def __repr__(self):
        return "Frame(%s)" % self

    @property
    def tuple_repr(self):
        """Make a tuple of attributes and values based on reader."""
        if not self._tuple_repr:
            self._tuple_repr = tuple(sorted((k, v) for k, v in self.attrs.items()))
        return self._tuple_repr

    def copy(self):
        return Frame(self.attrs.copy())

    def __getitem__(self, name):
        return self.attrs[name]

    def get(self, name, default=None):
        return self.attrs.get(name, default)

    def values(self, names):
        """Return a tuple of attribute values from this Frame."""
        if isinstance(names, (list, tuple)):
            return tuple(self.attrs.get(name) for name in names)
        else:
            return self.attrs.get(names)
