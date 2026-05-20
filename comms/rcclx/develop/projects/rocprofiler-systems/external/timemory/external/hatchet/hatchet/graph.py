# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from collections import defaultdict

from .node import Node, traversal_order


def index_by(attr, objects):
    """Put objects into lists based on the value of an attribute.

    Returns:
        (dict): dictionary of lists of objects, keyed by attribute value
    """
    index = defaultdict(lambda: [])
    for obj in objects:
        index[getattr(obj, attr)].append(obj)
    return index


class Graph:
    """A possibly multi-rooted tree or graph from one input dataset."""

    def __init__(self, roots):
        assert roots is not None
        self.roots = roots

    def traverse(self, order="pre", attrs=None, visited=None):
        """Preorder traversal of all roots of this Graph.

        Arguments:
            attrs (list or str, optional): If provided, extract these
                fields from nodes while traversing and yield them. See
                :func:`~hatchet.node.traverse` for details.

        Only preorder traversal is currently supported.
        """
        # share visited dict so that we visit each node at most once.
        if visited is None:
            visited = {}

        # iterate over roots in order
        for root in sorted(self.roots, key=traversal_order):
            for value in root.traverse(order=order, attrs=attrs, visited=visited):
                yield value

    def is_tree(self):
        """True if this graph is a tree, false otherwise."""
        if len(self.roots) > 1:
            return False

        visited = {}
        list(self.traverse(visited=visited))
        return all(v == 1 for v in visited.values())

    def find_merges(self):
        """Find nodes that have the same parent and frame.

        Find nodes that have the same parent and duplicate frame, and
        return a mapping from nodes that should be eliminated to nodes
        they should be merged into.

        Return:
            (dict): dictionary from nodes to their merge targets

        """
        merges = {}  # old_node -> merged_node
        inverted_merges = defaultdict(
            lambda: []
        )  # merged_node -> list of corresponding old_nodes
        processed = []

        def _find_child_merges(node_list):
            index = index_by("frame", node_list)
            for frame, children in index.items():
                if len(children) > 1:
                    min_id = min(children, key=id)
                    for child in children:
                        prev_min = merges.get(child, min_id)
                        # Get the new merged_node
                        curr_min = min([min_id, prev_min], key=id)
                        # Save the new merged_node to the merges dict
                        # so that the merge can happen later.
                        merges[child] = curr_min
                        # Update inverted_merges to be able to set node_list
                        # to the right value.
                        inverted_merges[curr_min].append(child)

        _find_child_merges(self.roots)
        for node in self.traverse():
            if node in processed:
                continue
            nodes = None
            # If node is going to be merged with other nodes,
            # collect the set of those nodes' children. This is
            # done to ensure that equivalent children of merged nodes
            # also get merged.
            if node in merges:
                new_node = merges[node]
                nodes = []
                for node_to_merge in inverted_merges[new_node]:
                    nodes.extend(node_to_merge.children)
                processed.extend(inverted_merges[new_node])
            # If node is not going to be merged, simply get the list of
            # node's children.
            else:
                nodes = node.children
                processed.append(node)
            _find_child_merges(nodes)

        return merges

    def merge_nodes(self, merges):
        """Merge some nodes in a graph into others.

        ``merges`` is a dictionary keyed by old nodes, with values equal
        to the nodes that they need to be merged into.  Old nodes'
        parents and children are connected to the new node.

        Arguments:
            merges (dict): dictionary from source nodes -> targets

        """

        def transform(node_list):
            return sorted(set(merges.get(n, n) for n in node_list))

        for old, new in merges.items():
            new.parents = transform(new.parents + old.parents)
            for parent in new.parents:
                parent.children = transform(parent.children)
            new.children = transform(new.children + old.children)
            for child in new.children:
                child.parents = transform(child.parents)
        self.roots = transform(self.roots)

    def normalize(self):
        merges = self.find_merges()
        self.merge_nodes(merges)
        return merges

    def copy(self, old_to_new=None):
        """Create and return a copy of this graph.

        Arguments:
            old_to_new (dict, optional): if provided, this dictionary will
                be populated with mappings from old node -> new node
        """
        # create a mapping dict if one wasn't passed in.
        if old_to_new is None:
            old_to_new = {}

        # first pass creates new nodes
        for node in self.traverse():
            old_to_new[node] = node.copy()

        # second pass hooks up parents and children
        for old, new in old_to_new.items():
            for old_parent in old.parents:
                new.parents.append(old_to_new[old_parent])
            for old_child in old.children:
                new.children.append(old_to_new[old_child])

        graph = Graph([old_to_new[r] for r in self.roots])
        graph.enumerate_traverse()

        return graph

    def union(self, other, old_to_new=None):
        """Create the union of self and other and return it as a new Graph.

        This creates a new graph and does not modify self or other. The
        new Graph has entirely new nodes.

        Arguments:
            other (Graph): another Graph
            old_to_new (dict, optional): if provided, this dictionary will
                be populated with mappings from old node -> new node

        Return:
            (Graph): new Graph containing all nodes and edges from self and other
        """
        if old_to_new is None:
            old_to_new = {}  # mapping from old nodes to new nodes

        def _merge(self_children, other_children, parent):
            """Recursively merge children of self and other.

            Arguments:
                self_children (list or tuple): List of children nodes from self
                other_children (list or tuple): List of children nodes from other
                parent (Node): Parent node for self and other child(ren)

            Modifies old_to_new (dict): Updated dict mapping old nodes from self and other to new
                unioned nodes

            Return:
                (list): list of merged children
            """

            def make_node(*nodes):
                """Make a new node to represent the union of other nodes."""
                new_node = nodes[0].copy()
                for node in nodes:
                    old_to_new[id(node)] = new_node
                return new_node

            new_children = []

            def connect(parent, new_node):
                if parent:
                    parent.add_child(new_node)
                    new_node.add_parent(parent)
                new_children.append(new_node)

            # step through both lists and merge nodes
            self_children, other_children = iter(self_children), iter(other_children)
            self_child = next(self_children, None)
            other_child = next(other_children, None)

            while self_child and other_child:
                if self_child.frame < other_child.frame:
                    # self_child is unique
                    new_node = old_to_new.get(id(self_child))
                    if not new_node:
                        new_node = make_node(self_child)
                        _merge(
                            sorted(self_child.children, key=lambda n: n.frame),
                            (),
                            new_node,
                        )
                    connect(parent, new_node)
                    self_child = next(self_children, None)

                elif self_child.frame > other_child.frame:
                    # other_child is unique
                    new_node = old_to_new.get(id(other_child))
                    if not new_node:
                        new_node = make_node(other_child)
                        _merge(
                            (),
                            sorted(other_child.children, key=lambda n: n.frame),
                            new_node,
                        )
                    connect(parent, new_node)
                    other_child = next(other_children, None)

                else:
                    # self_child and other_child are equal
                    self_mapped = old_to_new.get(id(self_child))
                    other_mapped = old_to_new.get(id(other_child))
                    if not self_mapped and not other_mapped:
                        new_node = make_node(self_child, other_child)
                    else:
                        new_node = self_mapped or other_mapped

                    # map whichever node was not mapped yet
                    if not self_mapped:
                        old_to_new[id(self_child)] = new_node
                        self_side = self_child.children
                    else:
                        self_side = []

                    if not other_mapped:
                        old_to_new[id(other_child)] = new_node
                        other_side = other_child.children
                    else:
                        other_side = []

                    _merge(
                        sorted(self_side, key=lambda n: n.frame),
                        sorted(other_side, key=lambda n: n.frame),
                        new_node,
                    )

                    connect(parent, new_node)
                    self_child = next(self_children, None)
                    other_child = next(other_children, None)

            # finish off whichever list of children is longer
            while self_child:
                new_node = old_to_new.get(id(self_child))
                if not new_node:
                    new_node = make_node(self_child)
                    _merge(
                        sorted(self_child.children, key=lambda n: n.frame),
                        (),
                        new_node,
                    )
                connect(parent, new_node)
                self_child = next(self_children, None)

            while other_child:
                new_node = old_to_new.get(id(other_child))
                if not new_node:
                    new_node = make_node(other_child)
                    _merge(
                        (),
                        sorted(other_child.children, key=lambda n: n.frame),
                        new_node,
                    )
                connect(parent, new_node)
                other_child = next(other_children, None)

            return new_children

        # First establish which nodes correspond to each other
        new_roots = _merge(
            sorted(self.roots, key=lambda n: n.frame),
            sorted(other.roots, key=lambda n: n.frame),
            None,
        )

        graph = Graph(new_roots)
        graph.enumerate_traverse()

        return graph

    def enumerate_depth(self):
        def _iter_depth(node, visited):
            for child in node.children:
                if child not in visited:
                    visited.add(child)
                    # depth of child is depth of node + 1
                    child._depth = node._depth + 1
                    _iter_depth(child, visited)

        visited = set()
        for root in self.roots:
            root._depth = 0  # depth of root node is 0
            _iter_depth(root, visited)

    def enumerate_traverse(self):
        if not self._check_enumerate_traverse():
            for i, node in enumerate(self.traverse()):
                node._hatchet_nid = i

            self.enumerate_depth()

    def _check_enumerate_traverse(self):
        for i, node in enumerate(self.traverse()):
            if i != node._hatchet_nid:
                return False

    def __len__(self):
        """Size of the graph in terms of number of nodes."""
        return sum(1 for _ in self.traverse())

    def __eq__(self, other):
        """Check if two graphs have the same structure by comparing frame at each
        node.
        """
        vs = set()
        vo = set()

        # if both graphs are pointing to the same object, then graphs are equal
        if self is other:
            return True

        # if number of roots do not match, then graphs are not equal
        if len(self.roots) != len(other.roots):
            return False

        if len(self) != len(other):
            return False

        # sort roots by its frame
        ssorted = sorted(self.roots, key=lambda x: x.frame)
        osorted = sorted(other.roots, key=lambda x: x.frame)

        for self_root, other_root in zip(ssorted, osorted):
            # if frames do not match, then nodes are not equal
            if self_root.frame != other_root.frame:
                return False

            if not self_root.dag_equal(other_root, vs, vo):
                return False

        return True

    def __ne__(self, other):
        return not (self == other)

    @staticmethod
    def from_lists(*roots):
        """Convenience method to invoke Node.from_lists() on each root value."""
        if not all(isinstance(r, (list, tuple)) for r in roots):
            raise ValueError(
                "All arguments to Graph.from_lists() must be lists: %s" % roots
            )

        graph = Graph([Node.from_lists(r) for r in roots])
        graph.enumerate_traverse()

        return graph
