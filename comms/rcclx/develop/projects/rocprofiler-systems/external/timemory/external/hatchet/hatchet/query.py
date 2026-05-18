# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from abc import abstractmethod

try:
    from abc import ABC
except ImportError:
    from abc import ABCMeta

    ABC = ABCMeta("ABC", (object,), {"__slots__": ()})

from itertools import groupby
from numbers import Real
import re
import sys
import pandas as pd
from pandas import DataFrame
from pandas.core.indexes.multi import MultiIndex
from textx import metamodel_from_str
from textx.exceptions import TextXError

# Flake8 to ignore this import, it does not recognize that eval("np.nan") needs
# numpy package
import numpy as np  # noqa: F401

from .node import Node, traversal_order


class AbstractQuery(ABC):
    """Abstract Base Class defining a Hatchet Query"""

    @abstractmethod
    def apply(self, gf):
        """Apply the query to a GraphFrame.

        Arguments:
            gf (GraphFrame): the GraphFrame on which to apply the query.

        Returns:
            (list): A list representing the set of nodes from paths that match this query.
        """
        pass

    def __and__(self, other):
        """Create an AndQuery with this query and another.

        Arguments:
            other (GraphFrame): the other query to use in the AndQuery.

        Returns:
            (AndQuery): A query object representing the intersection of the two queries.
        """
        return AndQuery(self, other)

    def __or__(self, other):
        """Create an OrQuery with this query and another.

        Arguments:
            other (GraphFrame): the other query to use in the OrQuery.

        Returns:
            (OrQuery): A query object representing the union of the two queries.
        """
        return OrQuery(self, other)

    def __xor__(self, other):
        """Create a XorQuery with this query and another.

        Arguments:
            other (GraphFrame): the other query to use in the XorQuery.

        Returns:
            (XorQuery): A query object representing the symmetric difference of the two queries.
        """
        return XorQuery(self, other)

    def __invert__(self):
        """Create a NotQuery with this query.

        Returns:
            (NotQuery): A query object representing all nodes that don't match this query.
        """
        return NotQuery(self)


class NaryQuery(AbstractQuery):
    """Abstract Base Class defining a compound query
    that acts on and merges N separate subqueries"""

    def __init__(self, *args):
        """Create a new NaryQuery object.

        Arguments:
            *args (tuple): the subqueries (high-level, low-level, or compound) to be performed.
        """
        self.subqueries = []
        if isinstance(args[0], tuple) and len(args) == 1:
            args = args[0]
        for query in args:
            if isinstance(query, list):
                self.subqueries.append(QueryMatcher(query))
            elif isinstance(query, str):
                self.subqueries.append(CypherQuery(query))
            elif issubclass(type(query), AbstractQuery):
                self.subqueries.append(query)
            else:
                raise TypeError(
                    "Subqueries for NaryQuery must be either a \
                                high-level query or a subclass of AbstractQuery"
                )

    @abstractmethod
    def _perform_nary_op(self, query_results, gf):
        """Perform the NaryQuery subclass's designated operation on the results of the subqueries.

        Arguments:
            query_results (list): the results of the subqueries.
            gf (GraphFrame): the GraphFrame on which the query is applied.

        Returns:
            (list): A list of nodes representing the result of applying the subclass-designated operation to the results of the subqueries.
        """
        pass

    def apply(self, gf):
        """Apply the NaryQuery to a GraphFrame.

        Arguments:
            gf (GraphFrame): the GraphFrame on which to apply the query.

        Returns:
            (list): A list of nodes representing the result of applying the subclass-designated operation to the results of the subqueries.
        """
        results = []
        for query in self.subqueries:
            results.append(query.apply(gf))
        return self._perform_nary_op(results, gf)


class QueryMatcher(AbstractQuery):
    """Process and apply queries to GraphFrames."""

    def __init__(self, query=None):
        """Create a new QueryMatcher object.

        Arguments:
            query (list, optional): if provided, convert the contents of the high-level API query into an internal representation.
        """
        # Initialize containers for query and memoization cache.
        self.query_pattern = []
        self.search_cache = {}
        # If a high-level API list is provided, process it.
        if query is not None:
            assert isinstance(query, list)

            def _convert_dict_to_filter(attr_filter):
                """Converts high-level API attribute filter to a lambda"""
                compops = ("<", ">", "==", ">=", "<=", "<>", "!=")  # ,
                # Currently not supported
                #           "is", "is not", "in", "not in")

                # This is a dict to work around Python's non-local variable
                # assignment rules.
                #
                # TODO: Replace this with the use of the "nonlocal" keyword
                #       once Python 2.7 support is dropped.
                first_no_drop_indices = {"val": True}

                def filter_series(df_row):
                    def filter_single_series(df_row, key, single_value):
                        if key == "depth":
                            node = df_row.name
                            if isinstance(
                                single_value, str
                            ) and single_value.lower().startswith(compops):
                                return eval("{} {}".format(node._depth, single_value))
                            if isinstance(single_value, Real):
                                return node._depth == single_value
                            raise InvalidQueryFilter(
                                "Attribute {} has a numeric type. Valid filters for this attribute are a string starting with a comparison operator or a real number.".format(
                                    key
                                )
                            )
                        if key == "node_id":
                            node = df_row.name
                            if isinstance(
                                single_value, str
                            ) and single_value.lower().startswith(compops):
                                return eval(
                                    "{} {}".format(node._hatchet_nid, single_value)
                                )
                            if isinstance(single_value, Real):
                                return node._hatchet_nid == single_value
                            raise InvalidQueryFilter(
                                "Attribute {} has a numeric type. Valid filters for this attribute are a string starting with a comparison operator or a real number.".format(
                                    key
                                )
                            )
                        if key not in df_row.keys():
                            return False
                        if isinstance(df_row[key], str):
                            if not isinstance(single_value, str):
                                raise InvalidQueryFilter(
                                    "Value for attribute {} must be a string.".format(
                                        key
                                    )
                                )
                            return (
                                re.match(single_value + r"\Z", df_row[key]) is not None
                            )
                        if isinstance(df_row[key], Real):
                            if isinstance(
                                single_value, str
                            ) and single_value.lower().startswith(compops):
                                # compare nan metric value to numeric query
                                # (e.g. np.nan > 5)
                                if pd.isnull(df_row[key]):
                                    nan_str = "np.nan"
                                    # compare nan metric value to nan query
                                    # (e.g., np.nan == np.nan)
                                    if nan_str in single_value:
                                        return eval(
                                            "pd.isnull({}) == True".format(nan_str)
                                        )
                                    return eval("{} {}".format(nan_str, single_value))
                                elif np.isinf(df_row[key]):
                                    inf_str = "np.inf"
                                    # compare inf metric value to inf query
                                    # (e.g., np.inf == np.inf)
                                    if inf_str in single_value:
                                        return eval(
                                            "np.isinf({}) == True".format(inf_str)
                                        )
                                    return eval("{} {}".format(inf_str, single_value))
                                else:
                                    return eval(
                                        "{} {}".format(df_row[key], single_value)
                                    )

                            if isinstance(single_value, Real):
                                return df_row[key] == single_value
                            raise InvalidQueryFilter(
                                "Attribute {} has a numeric type. Valid filters for this attribute are a string starting with a comparison operator or a real number.".format(
                                    key
                                )
                            )
                        raise InvalidQueryFilter(
                            "Filter must be one of the following:\n  * A regex string for a String attribute\n  * A string starting with a comparison operator for a Numeric attribute\n  * A number for a Numeric attribute\n"
                        )

                    matches = True
                    for k, v in attr_filter.items():
                        try:
                            _ = iter(v)
                            # Manually raise TypeError if v is a string so that
                            # the string is processed as a non-iterable
                            if isinstance(v, str):
                                raise TypeError
                        # Runs if v is not iterable (e.g., list, tuple, etc.)
                        except TypeError:
                            matches = matches and filter_single_series(df_row, k, v)
                        else:
                            for single_value in v:
                                matches = matches and filter_single_series(
                                    df_row, k, single_value
                                )
                    return matches

                def filter_dframe(df_row):
                    if first_no_drop_indices["val"]:
                        print(
                            "==================================================================="
                        )
                        print(
                            "WARNING: You are performing a query without dropping index levels."
                        )
                        print(
                            "         This is a valid operation, but it will significantly"
                        )
                        print(
                            "         increase the time it takes for this operation to complete."
                        )
                        print(
                            "         If you don't want the operation to take so long, call"
                        )
                        print("         GraphFrame.drop_index_levels() before calling")
                        print("         GraphFrame.filter()")
                        print(
                            "===================================================================\n"
                        )
                        first_no_drop_indices["val"] = False

                    def filter_single_dframe(node, df_row, key, single_value):
                        if key == "depth":
                            if isinstance(
                                single_value, str
                            ) and single_value.lower().startswith(compops):
                                return eval("{} {}".format(node._depth, single_value))
                            if isinstance(single_value, Real):
                                return node._depth == single_value
                            raise InvalidQueryFilter(
                                "Attribute {} has a numeric type. Valid filters for this attribute are a string starting with a comparison operator or a real number.".format(
                                    key
                                )
                            )
                        if key == "node_id":
                            if isinstance(
                                single_value, str
                            ) and single_value.lower().startswith(compops):
                                return eval(
                                    "{} {}".format(node._hatchet_nid, single_value)
                                )
                            if isinstance(single_value, Real):
                                return node._hatchet_nid == single_value
                            raise InvalidQueryFilter(
                                "Attribute {} has a numeric type. Valid filters for this attribute are a string starting with a comparison operator or a real number.".format(
                                    key
                                )
                            )
                        if key not in df_row.columns:
                            return False
                        if df_row[key].apply(type).eq(str).all():
                            if not isinstance(single_value, str):
                                raise InvalidQueryFilter(
                                    "Value for attribute {} must be a string.".format(
                                        key
                                    )
                                )
                            return (
                                df_row[key]
                                .apply(
                                    lambda x: re.match(single_value + r"\Z", x)
                                    is not None
                                )
                                .any()
                            )
                        if df_row[key].apply(type).eq(Real).all():
                            if isinstance(
                                single_value, str
                            ) and single_value.lower().startswith(compops):
                                return (
                                    df_row[key]
                                    .apply(
                                        lambda x: eval("{} {}".format(x, single_value))
                                    )
                                    .any()
                                )
                            if isinstance(single_value, Real):
                                return (
                                    df_row[key].apply(lambda x: x == single_value).any()
                                )
                            raise InvalidQueryFilter(
                                "Attribute {} has a numeric type. Valid filters for this attribute are a string starting with a comparison operator or a real number.".format(
                                    key
                                )
                            )
                        raise InvalidQueryFilter(
                            "Filter must be one of the following:\n  * A regex string for a String attribute\n  * A string starting with a comparison operator for a Numeric attribute\n  * A number for a Numeric attribute\n"
                        )

                    matches = True
                    node = df_row.name.to_frame().index[0][0]
                    for k, v in attr_filter.items():
                        try:
                            _ = iter(v)
                            if isinstance(v, str):
                                raise TypeError
                        except TypeError:
                            matches = matches and filter_single_dframe(
                                node, df_row, k, v
                            )
                        else:
                            for single_value in v:
                                matches = matches and filter_single_dframe(
                                    node, df_row, k, single_value
                                )
                    return matches

                def filter_choice(df_row):
                    if isinstance(df_row, DataFrame):
                        return filter_dframe(df_row)
                    return filter_series(df_row)

                return filter_choice if attr_filter != {} else lambda row: True

            for elem in query:
                if isinstance(elem, dict):
                    self._add_node(".", _convert_dict_to_filter(elem))
                elif isinstance(elem, str) or isinstance(elem, int):
                    self._add_node(elem)
                elif isinstance(elem, tuple):
                    assert isinstance(elem[1], dict)
                    if isinstance(elem[0], str) or isinstance(elem[0], int):
                        self._add_node(elem[0], _convert_dict_to_filter(elem[1]))
                    else:
                        raise InvalidQueryPath(
                            "The first value of a tuple entry in a path must be either a string or integer."
                        )
                else:
                    raise InvalidQueryPath(
                        "A query path must be a list containing String, Integer, Dict, or Tuple elements"
                    )

    def match(self, wildcard_spec=".", filter_func=lambda row: True):
        """Start a query with a root node described by the arguments.

        Arguments:
            wildcard_spec (str, optional, ".", "*", or "+"): the wildcard status of the node (follows standard Regex syntax)
            filter_func (callable, optional): a callable accepting only a row from a Pandas DataFrame that is used to filter this node in the query

        Returns:
            (QueryMatcher): The instance of the class that called this function (enables fluent design).
        """
        if len(self.query_pattern) != 0:
            self.query_pattern = []
        self._add_node(wildcard_spec, filter_func)
        return self

    def rel(self, wildcard_spec=".", filter_func=lambda row: True):
        """Add another edge and node to the query.

        Arguments:
            wildcard_spec (str, optional, ".", "*", or "+"): the wildcard status of the node (follows standard Regex syntax)
            filter_func (callable, optional): a callable accepting only a row from a Pandas DataFrame that is used to filter this node in the query

        Returns:
            (QueryMatcher): The instance of the class that called this function (enables fluent design).
        """
        self._add_node(wildcard_spec, filter_func)
        return self

    def apply(self, gf):
        """Apply the query to a GraphFrame.

        Arguments:
            gf (GraphFrame): the GraphFrame on which to apply the query.

        Returns:
            (list): A list representing the set of nodes from paths that match this query.
        """
        self.search_cache = {}
        matches = []
        visited = set()
        for root in sorted(gf.graph.roots, key=traversal_order):
            self._apply_impl(gf, root, visited, matches)
        assert len(visited) == len(gf.graph)
        matched_node_set = list(set().union(*matches))
        # return matches
        return matched_node_set

    def _add_node(self, wildcard_spec=".", filter_func=lambda row: True):
        """Add a node to the query.
        Arguments:
            wildcard_spec (str, optional, ".", "*", or "+"): the wildcard status of the node (follows standard Regex syntax)
            filter_func (callable, optional): a callable accepting only a row from a Pandas DataFrame that is used to filter this node in the query
        """
        assert isinstance(wildcard_spec, int) or isinstance(wildcard_spec, str)
        assert callable(filter_func)
        if isinstance(wildcard_spec, int):
            for i in range(wildcard_spec):
                self.query_pattern.append((".", filter_func))
        else:
            assert wildcard_spec == "." or wildcard_spec == "*" or wildcard_spec == "+"
            self.query_pattern.append((wildcard_spec, filter_func))

    def _cache_node(self, gf, node):
        """Cache (Memoize) the parts of the query that the node matches.

        Arguments:
            gf (GraphFrame): the GraphFrame containing the node to be cached.
            node (Node): the Node to be cached.
        """
        assert isinstance(node, Node)
        matches = []
        # Applies each filtering function to the node to cache which
        # query nodes the current node matches.
        for i, node_query in enumerate(self.query_pattern):
            _, filter_func = node_query
            row = None
            if isinstance(gf.dataframe.index, MultiIndex):
                row = pd.concat([gf.dataframe.loc[node]], keys=[node], names=["node"])
            else:
                row = gf.dataframe.loc[node]
            if filter_func(row):
                matches.append(i)
        self.search_cache[node._hatchet_nid] = matches

    def _match_0_or_more(self, gf, node, wcard_idx):
        """Process a "*" wildcard in the query on a subgraph.

        Arguments:
            gf (GraphFrame): the GraphFrame being queried.
            node (Node): the node being queried against the "*" wildcard.
            wcard_idx (int): the index associated with the "*" wildcard query.

        Returns:
            (list): a list of lists representing the paths rooted at "node" that match the "*" wildcard and/or the next query node. Will return None if there is no match for the "*" wildcard or the next query node.
        """
        # Cache the node if it's not already cached
        if node._hatchet_nid not in self.search_cache:
            self._cache_node(gf, node)
        # If the node matches with the next non-wildcard query node,
        # end the recursion and return the node.
        if wcard_idx + 1 in self.search_cache[node._hatchet_nid]:
            return [[]]
        # If the node matches the "*" wildcard query, recursively
        # apply this function to the current node's children. Then,
        # collect their returned matches, and prepend the current node.
        elif wcard_idx in self.search_cache[node._hatchet_nid]:
            matches = []
            if len(node.children) == 0:
                if wcard_idx == len(self.query_pattern) - 1:
                    return [[node]]
                return None
            for child in sorted(node.children, key=traversal_order):
                sub_match = self._match_0_or_more(gf, child, wcard_idx)
                if sub_match is not None:
                    matches.extend(sub_match)
            if len(matches) == 0:
                return None
            tmp = set(tuple(m) for m in matches)
            matches = [list(t) for t in tmp]
            return [[node] + m for m in matches]
        # If the current node doesn't match the current "*" wildcard or
        # the next non-wildcard query node, return None.
        else:
            if wcard_idx == len(self.query_pattern) - 1:
                return [[]]
            return None

    def _match_1_or_more(self, gf, node, wcard_idx):
        """Process a "+" wildcard in the query on a subgraph.

        Arguments:
            gf (GraphFrame): the GraphFrame being queried.
            node (Node): the node being queried against the "+" wildcard.
            wcard_idx (int): the index associated with the "+" wildcard query.

        Returns:
            (list): a list of lists representing the paths rooted at "node" that match the "+" wildcard and/or the next query node. Will return None if there is no match for the "+" wildcard or the next query node.
        """
        # Cache the node if it's not already cached
        if node._hatchet_nid not in self.search_cache:
            self._cache_node(gf, node)
        # If the current node doesn't match the "+" wildcard, return None.
        if wcard_idx not in self.search_cache[node._hatchet_nid]:
            return None
        # Since a query can't end on a wildcard, return None if the
        # current node has no children.
        if len(node.children) == 0:
            return None
        # Use _match_0_or_more to collect all additional wildcard matches.
        matches = []
        for child in sorted(node.children, key=traversal_order):
            sub_match = self._match_0_or_more(gf, child, wcard_idx)
            if sub_match is not None:
                matches.extend(sub_match)
        # Since _match_0_or_more will capture the query node that follows
        # the wildcard, if no paths were retrieved from that function,
        # the pattern does not continue after the "+" wildcard. Thus,
        # since a pattern cannot end on a wildcard, the function
        # returns None.
        if len(matches) == 0:
            return None
        return [[node] + m for m in matches]

    def _match_1(self, gf, node, idx):
        """Process a "." wildcard in the query on a subgraph.

        Arguments:
            gf (GraphFrame): the GraphFrame being queried.
            node (Node): the node being queried against the "." wildcard.
            wcard_idx (int): the index associated with the "." wildcard query.

        Returns:
            (list): A list of lists representing the children of "node" that match the "." wildcard being considered. Will return None if there are no matches for the "." wildcard.
        """
        if node._hatchet_nid not in self.search_cache:
            self._cache_node(gf, node)
        matches = []
        for child in sorted(node.children, key=traversal_order):
            # Cache the node if it's not already cached
            if child._hatchet_nid not in self.search_cache:
                self._cache_node(gf, child)
            if idx in self.search_cache[child._hatchet_nid]:
                matches.append([child])
        # To be consistent with the other matching functions, return
        # None instead of an empty list.
        if len(matches) == 0:
            return None
        return matches

    def _match_pattern(self, gf, pattern_root, match_idx):
        """Try to match the query pattern starting at the provided root node.

        Arguments:
            gf (GraphFrame): the GraphFrame being queried.
            pattern_root (Node): the root node of the subgraph that is being queried.

        Returns:
            (list): A list of lists representing the paths rooted at "pattern_root" that match the query.
        """
        assert isinstance(pattern_root, Node)
        # Starting query node
        pattern_idx = match_idx + 1
        if (
            self.query_pattern[match_idx][0] == "*"
            or self.query_pattern[match_idx][0] == "+"
        ):
            pattern_idx = 0
        # Starting matching pattern
        matches = [[pattern_root]]
        while pattern_idx < len(self.query_pattern):
            # Get the wildcard type
            wcard, _ = self.query_pattern[pattern_idx]
            new_matches = []
            # Consider each existing match individually so that more
            # nodes can be added to them.
            for m in matches:
                sub_match = []
                # Get the portion of the subgraph that matches the next
                # part of the query.
                if wcard == ".":
                    s = self._match_1(gf, m[-1], pattern_idx)
                    if s is None:
                        sub_match.append(s)
                    else:
                        sub_match.extend(s)
                elif wcard == "*":
                    if len(m[-1].children) == 0:
                        sub_match.append([])
                    else:
                        for child in sorted(m[-1].children, key=traversal_order):
                            s = self._match_0_or_more(gf, child, pattern_idx)
                            if s is None:
                                sub_match.append(s)
                            else:
                                sub_match.extend(s)
                elif wcard == "+":
                    if len(m[-1].children) == 0:
                        sub_match.append(None)
                    else:
                        for child in sorted(m[-1].children, key=traversal_order):
                            s = self._match_1_or_more(gf, child, pattern_idx)
                            if s is None:
                                sub_match.append(s)
                            else:
                                sub_match.extend(s)
                else:
                    raise InvalidQueryFilter(
                        'Query wildcards must be one of ".", "*", or "+"'
                    )
                # Merge the next part of the match path with the
                # existing part.
                for s in sub_match:
                    if s is not None:
                        new_matches.append(m + s)
                new_matches = [uniq_match for uniq_match, _ in groupby(new_matches)]
            # Overwrite the old matches with the updated matches
            matches = new_matches
            # If all the existing partial matches were not able to be
            # expanded into full matches, return None.
            if len(matches) == 0:
                return None
            # Update the query node
            pattern_idx += 1
        return matches

    def _apply_impl(self, gf, node, visited, matches):
        """Traverse the subgraph with the specified root, and collect all paths that match the query.

        Arguments:
            gf (GraphFrame): the GraphFrame being queried.
            node (Node): the root node of the subgraph that is being queried.
            visited (set): a set that keeps track of what nodes have been visited in the traversal to minimize the amount of work that is repeated.
            matches (list): the list in which the final set of matches are stored.
        """
        # If the node has already been visited (or is None for some
        # reason), skip it.
        if node is None or node._hatchet_nid in visited:
            return
        # Cache the node if it's not already cached
        if node._hatchet_nid not in self.search_cache:
            self._cache_node(gf, node)
        # If the node matches the starting/root node of the query,
        # try to get all query matches in the subgraph rooted at
        # this node.
        if self.query_pattern[0][0] == "*":
            if 1 in self.search_cache[node._hatchet_nid]:
                sub_match = self._match_pattern(gf, node, 1)
                if sub_match is not None:
                    matches.extend(sub_match)
        if 0 in self.search_cache[node._hatchet_nid]:
            sub_match = self._match_pattern(gf, node, 0)
            if sub_match is not None:
                matches.extend(sub_match)
        # Note that the node is now visited.
        visited.add(node._hatchet_nid)
        # Continue the Depth First Search.
        for child in sorted(node.children, key=traversal_order):
            self._apply_impl(gf, child, visited, matches)


GRAMMAR = """
FullQuery: path_expr=MatchExpr(cond_expr=WhereExpr)?;
MatchExpr: 'MATCH' path=PathQuery;
PathQuery: '(' nodes=NodeExpr ')'('->' '(' nodes=NodeExpr ')')*;
NodeExpr: ((wcard=INT | wcard=STRING) ',' name=ID) | (wcard=INT | wcard=STRING) |  name=ID;
WhereExpr: 'WHERE' ConditionExpr;
ConditionExpr: conditions+=CompoundCond;
CompoundCond: UnaryCond | BinaryCond;
BinaryCond: AndCond | OrCond;
AndCond: 'AND' subcond=UnaryCond;
OrCond: 'OR' subcond=UnaryCond;
UnaryCond: NotCond | SingleCond;
NotCond: 'NOT' subcond=SingleCond;
SingleCond: StringCond | NumberCond | NoneCond | NotNoneCond;
NoneCond: name=ID '.' prop=STRING 'IS NONE';
NotNoneCond: name=ID '.' prop=STRING 'IS NOT NONE';
StringCond: StringEq | StringStartsWith | StringEndsWith | StringContains | StringMatch;
StringEq: name=ID '.' prop=STRING '=' val=STRING;
StringStartsWith: name=ID '.' prop=STRING 'STARTS WITH' val=STRING;
StringEndsWith: name=ID '.' prop=STRING 'ENDS WITH' val=STRING;
StringContains: name=ID '.' prop=STRING 'CONTAINS' val=STRING;
StringMatch: name=ID '.' prop=STRING '=~' val=STRING;
NumberCond: NumEq | NumLt | NumGt | NumLte | NumGte | NumNan | NumNotNan | NumInf | NumNotInf;
NumEq: name=ID '.' prop=STRING '=' val=NUMBER;
NumLt: name=ID '.' prop=STRING '<' val=NUMBER;
NumGt: name=ID '.' prop=STRING '>' val=NUMBER;
NumLte: name=ID '.' prop=STRING '<=' val=NUMBER;
NumGte: name=ID '.' prop=STRING '>=' val=NUMBER;
NumNan: name=ID '.' prop=STRING 'IS NAN';
NumNotNan: name=ID '.' prop=STRING 'IS NOT NAN';
NumInf: name=ID '.' prop=STRING 'IS INF';
NumNotInf: name=ID '.' prop=STRING 'IS NOT INF';
"""

mm = metamodel_from_str(GRAMMAR)


def cname(obj):
    return obj.__class__.__name__


def filter_check_types(type_check, df_row, filt_lambda):
    try:
        if type_check == "" or eval(type_check):
            return filt_lambda(df_row)
        else:
            raise InvalidQueryFilter("Type mismatch in filter")
    except KeyError:
        return False


class CypherQuery(QueryMatcher):
    def __init__(self, cypher_query):
        if sys.version_info[0] == 2:
            super(CypherQuery, self).__init__()
        else:
            super().__init__()
        model = None
        try:
            model = mm.model_from_str(cypher_query)
        except TextXError as e:
            # TODO Change to a "raise-from" expression when Python 2.7 support is dropped
            raise InvalidQueryPath(
                'Invalid "Cypher" Query Detected. Parser Error Message: {}'.format(
                    e.message
                )
            )
        self.wcards = []
        self.wcard_pos = {}
        self._parse_path(model.path_expr)
        self.filters = [[] for _ in self.wcards]
        self._parse_conditions(model.cond_expr)
        self.lambda_filters = [None for _ in self.wcards]
        self._build_lambdas()
        self._build_query()

    def _build_query(self):
        for i in range(0, len(self.wcards)):
            wcard = self.wcards[i][0]
            # TODO Remove this when Python 2.7 support is dropped.
            if sys.version_info[0] == 2 and not isinstance(wcard, Real):
                wcard = wcard.encode("ascii", "ignore")
            filt_str = self.lambda_filters[i]
            if filt_str is None:
                if i == 0:
                    self.match(wildcard_spec=wcard)
                else:
                    self.rel(wildcard_spec=wcard)
            else:
                if i == 0:
                    self.match(wildcard_spec=wcard, filter_func=eval(filt_str))
                else:
                    self.rel(wildcard_spec=wcard, filter_func=eval(filt_str))

    def _build_lambdas(self):
        for i in range(0, len(self.wcards)):
            n = self.wcards[i]
            if n[1] != "":
                bool_expr = ""
                type_check = ""
                for j, cond in enumerate(self.filters[i]):
                    if cond[0] is not None:
                        bool_expr += " {}".format(cond[0])
                    bool_expr += " {}".format(cond[1])
                    if cond[2] is not None:
                        if j == 0:
                            type_check += " {}".format(cond[2])
                        else:
                            type_check += " and {}".format(cond[2])
                bool_expr = "lambda df_row: {}".format(bool_expr)
                bool_expr = (
                    'lambda df_row: filter_check_types("{}", df_row, {})'.format(
                        type_check, bool_expr
                    )
                )
                self.lambda_filters[i] = bool_expr

    def _parse_path(self, path_obj):
        nodes = path_obj.path.nodes
        idx = len(self.wcards)
        for n in nodes:
            new_node = [n.wcard, n.name]
            if n.wcard is None or n.wcard == "" or n.wcard == 0:
                new_node[0] = "."
            self.wcards.append(new_node)
            if n.name != "":
                self.wcard_pos[n.name] = idx
            idx += 1

    def _parse_conditions(self, cond_expr):
        conditions = cond_expr.conditions
        for cond in conditions:
            converted_condition = None
            if self._is_unary_cond(cond):
                converted_condition = self._parse_unary_cond(cond)
            elif self._is_binary_cond(cond):
                converted_condition = self._parse_binary_cond(cond)
            else:
                raise RuntimeError("Bad Condition")
            self.filters[self.wcard_pos[converted_condition[1]]].append(
                [converted_condition[0], converted_condition[2], converted_condition[3]]
            )
        for i in range(0, len(self.filters)):
            if len(self.filters[i]) > 0:
                if self.filters[i][0][0] != "not":
                    self.filters[i][0][0] = None

    def _is_unary_cond(self, obj):
        if (
            cname(obj) == "NotCond"
            or self._is_str_cond(obj)
            or self._is_num_cond(obj)
            or cname(obj) in ["NoneCond", "NotNoneCond"]
        ):
            return True
        return False

    def _is_binary_cond(self, obj):
        if cname(obj) in ["AndCond", "OrCond"]:
            return True
        return False

    def _parse_binary_cond(self, obj):
        if cname(obj) == "AndCond":
            return self._parse_and_cond(obj)
        if cname(obj) == "OrCond":
            return self._parse_or_cond(obj)
        raise RuntimeError("Bad Binary Condition")

    def _parse_or_cond(self, obj):
        converted_subcond = self._parse_unary_cond(obj.subcond)
        converted_subcond[0] = "or"
        return converted_subcond

    def _parse_and_cond(self, obj):
        converted_subcond = self._parse_unary_cond(obj.subcond)
        converted_subcond[0] = "and"
        return converted_subcond

    def _parse_unary_cond(self, obj):
        if cname(obj) == "NotCond":
            return self._parse_not_cond(obj)
        return self._parse_single_cond(obj)

    def _parse_not_cond(self, obj):
        converted_subcond = self._parse_single_cond(obj.subcond)
        converted_subcond[2] = "not {}".format(converted_subcond[2])
        return converted_subcond

    def _parse_single_cond(self, obj):
        if self._is_str_cond(obj):
            return self._parse_str(obj)
        if self._is_num_cond(obj):
            return self._parse_num(obj)
        if cname(obj) == "NoneCond":
            return self._parse_none(obj)
        if cname(obj) == "NotNoneCond":
            return self._parse_not_none(obj)
        raise RuntimeError("Bad Single Condition")

    def _parse_none(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "df_row.name._depth is None",
                None,
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "df_row.name._hatchet_nid is None",
                None,
            ]
        return [
            None,
            obj.name,
            'df_row["{}"] is None'.format(obj.prop),
            None,
        ]

    def _parse_not_none(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "df_row.name._depth is not None",
                None,
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "df_row.name._hatchet_nid is not None",
                None,
            ]
        return [
            None,
            obj.name,
            'df_row["{}"] is not None'.format(obj.prop),
            None,
        ]

    def _is_str_cond(self, obj):
        if cname(obj) in [
            "StringEq",
            "StringStartsWith",
            "StringEndsWith",
            "StringContains",
            "StringMatch",
        ]:
            return True
        return False

    def _is_num_cond(self, obj):
        if cname(obj) in [
            "NumEq",
            "NumLt",
            "NumGt",
            "NumLte",
            "NumGte",
            "NumNan",
            "NumNotNan",
            "NumInf",
            "NumNotInf",
        ]:
            return True
        return False

    def _parse_str(self, obj):
        if cname(obj) == "StringEq":
            return self._parse_str_eq(obj)
        if cname(obj) == "StringStartsWith":
            return self._parse_str_starts_with(obj)
        if cname(obj) == "StringEndsWith":
            return self._parse_str_ends_with(obj)
        if cname(obj) == "StringContains":
            return self._parse_str_contains(obj)
        if cname(obj) == "StringMatch":
            return self._parse_str_match(obj)
        raise RuntimeError("Bad String Op Class")

    def _parse_str_eq(self, obj):
        return [
            None,
            obj.name,
            'df_row["{}"] == "{}"'.format(obj.prop, obj.val),
            "isinstance(df_row['{}'], str)".format(obj.prop),
        ]

    def _parse_str_starts_with(self, obj):
        return [
            None,
            obj.name,
            'df_row["{}"].startswith("{}")'.format(obj.prop, obj.val),
            "isinstance(df_row['{}'], str)".format(obj.prop),
        ]

    def _parse_str_ends_with(self, obj):
        return [
            None,
            obj.name,
            'df_row["{}"].endswith("{}")'.format(obj.prop, obj.val),
            "isinstance(df_row['{}'], str)".format(obj.prop),
        ]

    def _parse_str_contains(self, obj):
        return [
            None,
            obj.name,
            '"{}" in df_row["{}"]'.format(obj.val, obj.prop),
            "isinstance(df_row['{}'], str)".format(obj.prop),
        ]

    def _parse_str_match(self, obj):
        return [
            None,
            obj.name,
            're.match("{}", df_row["{}"]) is not None'.format(obj.val, obj.prop),
            "isinstance(df_row['{}'], str)".format(obj.prop),
        ]

    def _parse_num(self, obj):
        if cname(obj) == "NumEq":
            return self._parse_num_eq(obj)
        if cname(obj) == "NumLt":
            return self._parse_num_lt(obj)
        if cname(obj) == "NumGt":
            return self._parse_num_gt(obj)
        if cname(obj) == "NumLte":
            return self._parse_num_lte(obj)
        if cname(obj) == "NumGte":
            return self._parse_num_gte(obj)
        if cname(obj) == "NumNan":
            return self._parse_num_nan(obj)
        if cname(obj) == "NumNotNan":
            return self._parse_num_not_nan(obj)
        if cname(obj) == "NumInf":
            return self._parse_num_inf(obj)
        if cname(obj) == "NumNotInf":
            return self._parse_num_not_inf(obj)
        raise RuntimeError("Bad Number Op Class")

    def _parse_num_eq(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "df_row.name._depth == {}".format(obj.val),
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "df_row.name._hatchet_nid == {}".format(obj.val),
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'df_row["{}"] == {}'.format(obj.prop, obj.val),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]

    def _parse_num_lt(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "df_row.name._depth < {}".format(obj.val),
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "df_row.name._hatchet_nid < {}".format(obj.val),
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'df_row["{}"] < {}'.format(obj.prop, obj.val),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]

    def _parse_num_gt(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "df_row.name._depth > {}".format(obj.val),
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "df_row.name._hatchet_nid > {}".format(obj.val),
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'df_row["{}"] > {}'.format(obj.prop, obj.val),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]

    def _parse_num_lte(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "df_row.name._depth <= {}".format(obj.val),
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "df_row.name._hatchet_nid <= {}".format(obj.val),
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'df_row["{}"] <= {}'.format(obj.prop, obj.val),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]

    def _parse_num_gte(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "df_row.name._depth >= {}".format(obj.val),
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "df_row.name._hatchet_nid >= {}".format(obj.val),
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'df_row["{}"] >= {}'.format(obj.prop, obj.val),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]

    def _parse_num_nan(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "pd.isna(df_row.name._depth)",
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "pd.isna(df_row.name._hatchet_nid)",
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'pd.isna(df_row["{}"])'.format(obj.prop),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]

    def _parse_num_not_nan(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "not pd.isna(df_row.name._depth)",
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "not pd.isna(df_row.name._hatchet_nid)",
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'not pd.isna(df_row["{}"])'.format(obj.prop),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]

    def _parse_num_inf(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "np.isinf(df_row.name._depth)",
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "np.isinf(df_row.name._hatchet_nid)",
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'np.isinf(df_row["{}"])'.format(obj.prop),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]

    def _parse_num_not_inf(self, obj):
        if obj.prop == "depth":
            return [
                None,
                obj.name,
                "not np.isinf(df_row.name._depth)",
                "isinstance(df_row.name._depth, Real)",
            ]
        if obj.prop == "node_id":
            return [
                None,
                obj.name,
                "not np.isinf(df_row.name._hatchet_nid)",
                "isinstance(df_row.name._hatchet_nid, Real)",
            ]
        return [
            None,
            obj.name,
            'not np.isinf(df_row["{}"])'.format(obj.prop),
            "isinstance(df_row['{}'], Real)".format(obj.prop),
        ]


class AndQuery(NaryQuery):
    """Compound Query that returns the intersection of the results
    of the subqueries"""

    def __init__(self, *args):
        """Create a new AndQuery object.

        Arguments:
            *args (tuple): the subqueries (high-level, low-level, or compound) to be performed.
        """
        if sys.version_info[0] == 2:
            super(AndQuery, self).__init__(args)
        else:
            super().__init__(args)
        if len(self.subqueries) < 2:
            raise BadNumberNaryQueryArgs("AndQuery requires 2 or more subqueries")

    def _perform_nary_op(self, query_results, gf):
        """Perform an intersection operation on the results of the subqueries.

        Arguments:
            query_results (list): the results of the subqueries.
            gf (GraphFrame): the GraphFrame on which the query is applied.

        Returns:
            (list): A list of nodes representing the intersection of the results of the subqueries.
        """
        intersection_set = set(query_results[0]).intersection(*query_results[1:])
        return list(intersection_set)


"""Alias of AndQuery to signify the relationship to set Intersection"""
IntersectionQuery = AndQuery


class OrQuery(NaryQuery):
    """Compound Query that returns the union of the results
    of the subqueries"""

    def __init__(self, *args):
        """Create a new OrQuery object.

        Arguments:
            *args (tuple): the subqueries (high-level, low-level, or compound) to be performed.
        """
        if sys.version_info[0] == 2:
            super(OrQuery, self).__init__(args)
        else:
            super().__init__(args)
        if len(self.subqueries) < 2:
            raise BadNumberNaryQueryArgs("OrQuery requires 2 or more subqueries")

    def _perform_nary_op(self, query_results, gf):
        """Perform an union operation on the results of the subqueries.

        Arguments:
            query_results (list): the results of the subqueries.
            gf (GraphFrame): the GraphFrame on which the query is applied.

        Returns:
            (list): A list of nodes representing the union of the results of the subqueries.
        """
        union_set = set().union(*query_results)
        return list(union_set)


"""Alias of OrQuery to signify the relationship to set Union"""
UnionQuery = OrQuery


class XorQuery(NaryQuery):
    """Compound Query that returns the symmetric difference
    (i.e., set-based XOR) of the results of the subqueries"""

    def __init__(self, *args):
        """Create a new XorQuery object.

        Arguments:
            *args (tuple): the subqueries (high-level, low-level, or compound) to be performed.
        """
        if sys.version_info[0] == 2:
            super(XorQuery, self).__init__(args)
        else:
            super().__init__(args)
        if len(self.subqueries) < 2:
            raise BadNumberNaryQueryArgs("XorQuery requires 2 or more subqueries")

    def _perform_nary_op(self, query_results, gf):
        """Perform a symmetric difference operation on the results of the subqueries.

        Arguments:
            query_results (list): the results of the subqueries.
            gf (GraphFrame): the GraphFrame on which the query is applied.

        Returns:
            (list): A list of nodes representing the symmetric difference of the results of the subqueries.
        """
        xor_set = set()
        for res in query_results:
            xor_set = xor_set.symmetric_difference(set(res))
        return list(xor_set)


"""Alias of XorQuery to signify the relationship to set Symmetric Difference"""
SymDifferenceQuery = XorQuery


class NotQuery(NaryQuery):
    """Compound Query that returns all nodes in the GraphFrame that
    are not returned from the subquery."""

    def __init__(self, *args):
        """Create a new XorQuery object.

        Arguments:
            *args (tuple): the subquery (high-level, low-level, or compound) to be performed.
        """
        if sys.version_info[0] == 2:
            super(NotQuery, self).__init__(args)
        else:
            super().__init__(args)
        if len(self.subqueries) != 1:
            raise BadNumberNaryQueryArgs("NotQuery requires exactly 1 subquery")

    def _perform_nary_op(self, query_results, gf):
        """Collect all nodes in the graph not present in the query result.

        Arguments:
            query_results (list): the result of the subquery.
            gf (GraphFrame): the GraphFrame on which the query is applied.

        Returns:
            (list): A list of all nodes not found in the subquery.
        """
        nodes = set(gf.graph.traverse())
        query_nodes = set(query_results[0])
        return list(nodes.difference(query_nodes))


class InvalidQueryPath(Exception):
    """Raised when a query does not have the correct syntax"""


class InvalidQueryFilter(Exception):
    """Raised when a query filter does not have a valid syntax"""


class BadNumberNaryQueryArgs(Exception):
    """Raised when a query filter does not have a valid syntax"""
