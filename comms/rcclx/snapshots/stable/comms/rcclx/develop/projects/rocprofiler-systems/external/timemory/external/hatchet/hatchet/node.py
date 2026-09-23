# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from functools import total_ordering

from .frame import Frame


def traversal_order(node):
    """Deterministic key function for sorting nodes in traversals."""
    return (node.frame, id(node))


@total_ordering
class Node:
    """A node in the graph. The node only stores its frame."""

    def __init__(self, frame_obj, parent=None, hnid=-1, depth=-1):
        self.frame = frame_obj
        self._depth = depth
        self._hatchet_nid = hnid

        self.parents = []
        if parent is not None:
            self.add_parent(parent)
        self.children = []

    def add_parent(self, node):
        """Adds a parent to this node's list of parents."""
        assert isinstance(node, Node)
        self.parents.append(node)

    def add_child(self, node):
        """Adds a child to this node's list of children."""
        assert isinstance(node, Node)
        self.children.append(node)

    def paths(self):
        """List of tuples, one for each path from this node to any root.

        Paths are tuples of node objects.
        """
        node_value = (self,)
        if not self.parents:
            return [node_value]
        else:
            paths = []
            for parent in self.parents:
                parent_paths = parent.paths()
                paths.extend([path + node_value for path in parent_paths])
            return paths

    def path(self, attrs=None):
        """Path to this node from root. Raises if there are multiple paths.

        This is useful for trees (where each node only has one path), as
        it just gets the only element from ``self.paths``.  This will
        fail with a MultiplePathError if there is more than one path to
        this node.
        """
        paths = self.paths()
        if len(paths) > 1:
            raise MultiplePathError("Node has more than one path: " % paths)
        return paths[0]

    def dag_equal(self, other, vs=None, vo=None):
        """Check if DAG rooted at self has the same structure as that rooted at
        other.
        """
        if vs is None:
            vs = set()
        if vo is None:
            vo = set()

        vs.add(self._hatchet_nid)
        vo.add(other._hatchet_nid)

        # if number of children do not match, then nodes are not equal
        if len(self.children) != len(other.children):
            return False

        # sort children of each node by its frame
        ssorted = sorted(self.children, key=lambda x: x.frame)
        osorted = sorted(other.children, key=lambda x: x.frame)

        for self_child, other_child in zip(ssorted, osorted):
            # if frames do not match, then nodes are not equal
            if self_child.frame != other_child.frame:
                return False

            visited_s = self_child._hatchet_nid in vs
            visited_o = other_child._hatchet_nid in vo

            # check for duplicate nodes
            if visited_s != visited_o:
                return False

            # skip visited nodes
            if visited_s or visited_o:
                continue

            # recursive check for node equality
            if not self_child.dag_equal(other_child, vs, vo):
                return False

        return True

    def traverse(self, order="pre", attrs=None, visited=None):
        """Traverse the tree depth-first and yield each node.

        Arguments:
            order (str):  "pre" or "post" for preorder or postorder (default: pre)
            attrs (list or str, optional): if provided, extract these fields
                from nodes while traversing and yield them
            visited (dict, optional): dictionary in which each visited
                node's in-degree will be stored
        """
        if order not in ("pre", "post"):
            raise ValueError("order must be one of 'pre' or 'post'")

        if visited is None:
            visited = {}

        key = id(self)
        if key in visited:
            # count the number of times we reached
            visited[key] += 1
            return
        visited[key] = 1

        def value(node):
            return node if attrs is None else node.frame.values(attrs)

        if order == "pre":
            yield value(self)

        for child in sorted(self.children, key=traversal_order):
            for item in child.traverse(order=order, attrs=attrs, visited=visited):
                yield item

        if order == "post":
            yield value(self)

    def __hash__(self):
        return self._hatchet_nid

    def __eq__(self, other):
        return self._hatchet_nid == other._hatchet_nid

    def __lt__(self, other):
        return self._hatchet_nid < other._hatchet_nid

    def __gt__(self, other):
        return self._hatchet_nid > other._hatchet_nid

    def __str__(self):
        """Returns a string representation of the node."""
        return str(self.frame)

    def copy(self):
        """Copy this node without preserving parents or children."""
        return Node(frame_obj=self.frame.copy())

    @classmethod
    def from_lists(cls, lists):
        r"""Construct a hierarchy of nodes from recursive lists.

For example, this will construct a simple tree:

.. code-block:: python

    Node.from_lists(
        ["a",
            ["b", "d", "e"],
            ["c", "f", "g"],
        ]
    )

.. code-block:: console

         a
        / \
       b   c
     / |   | \
    d  e   f  g

And this will construct a simple diamond DAG:

.. code-block:: python

    d = Node(Frame(name="d"))
    Node.from_lists(
        ["a",
            ["b", d],
            ["c", d]
        ]
    )

.. code-block:: console

      a
     / \
    b   c
     \ /
      d

In the above examples, the 'a' represents a Node with its
`frame == Frame(name="a")`.
"""

        def _from_lists(lists, parent):
            if isinstance(lists, (tuple, list)):
                if isinstance(lists[0], Node):
                    node = lists[0]
                elif isinstance(lists[0], str):
                    node = Node(Frame(name=lists[0]))
                children = lists[1:]
                for val in children:
                    _ = _from_lists(val, node)
            elif isinstance(lists, str):
                node = Node(Frame(name=lists))
            elif isinstance(lists, Node):
                node = lists
            else:
                raise ValueError("Argument must be str, list, or Node: %s" % lists)

            if parent:
                node.add_parent(parent)
                parent.add_child(node)

            return node

        return _from_lists(lists, None)

    def __repr__(self):
        return "Node({%s})" % ", ".join(
            "%s: %s" % (repr(k), repr(v)) for k, v in sorted(self.frame.attrs.items())
        )


class MultiplePathError(Exception):
    """Raised when a node is asked for a single path but has multiple."""
