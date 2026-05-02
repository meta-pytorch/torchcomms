# Copyright 2020-2023 The Regents of the University of California, through
# Lawrence Berkeley National Laboratory, and other Hatchet Project Developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import json
import pandas as pd
import os
import glob
from hatchet.graphframe import GraphFrame
from ..node import Node
from ..graph import Graph
from ..frame import Frame
from ..util.timer import Timer


class TimemoryReader:
    """Read in timemory JSON output"""

    def __init__(self, input, select=None, **_kwargs):
        """Arguments:
        input (str or file-stream or dict or None):
            Valid argument types are:

            1. Filename for a timemory JSON tree file
            2. Open file stream to one of these files
            3. Dictionary from timemory JSON tree

        select (list of str):
            A list of strings which match the component enumeration names, e.g. ["cpu_clock"].

        per_thread (boolean):
            Ensures that when applying filters to the graphframe, frames with
            identical name/file/line/etc. info but from different threads are not
            combined

        per_rank (boolean):
            Ensures that when applying filters to the graphframe, frames with
            identical name/file/line/etc. info but from different ranks are not
            combined
        """
        self.graph_dict = {"timemory": {}}
        self.input = input
        self.default_metric = None
        self.timer = Timer()
        self.metric_cols = []
        self.properties = {}
        self.include_tid = True
        self.include_nid = True
        self.multiple_ranks = False
        self.multiple_threads = False
        self.callpath_to_node_dict = {}  # (callpath, rank, thread): <node_dict>
        self.callpath_to_node = {}  # (callpath): <node>
        self.metadata = {"hatchet_inclusive_suffix": ".inc"}

        # the per_thread and per_rank settings make sure that
        # squashing doesn't collapse the threads/ranks
        self.per_thread = _kwargs["per_thread"] if "per_thread" in _kwargs else False
        self.per_rank = _kwargs["per_rank"] if "per_rank" in _kwargs else False

        if select is None:
            self.select = select
        elif isinstance(select, list):
            if select:

                def _get_select(val):
                    if isinstance(val, str):
                        return val.lower()
                    elif callable(val):
                        return val().lower()
                    raise TypeError(
                        "Items in select must be string or callable: {}".format(
                            type(val).__name__
                        )
                    )

                self.select = [_get_select(v) for v in select]
        else:
            raise TypeError("select must be None or list of string")

    def create_graph(self):
        """Create graph and dataframe"""
        list_roots = []

        def remove_keys(_dict, _keys):
            """Remove keys from dictionary"""
            if isinstance(_keys, str):
                if _keys in _dict:
                    del _dict[_keys]
            else:
                for _key in _keys:
                    _dict = remove_keys(_dict, _key)
            return _dict

        def add_metrics(_dict):
            """Add any keys to metric_cols which don't already exist"""
            for key, itr in _dict.items():
                if key not in self.metric_cols:
                    self.metric_cols.append(key)

        def process_regex(_data):
            """Process the regex data for func/file/line info"""
            _tmp = {}
            if _data is not None and len(_data.groups()) > 0:
                for _key in ("head", "func", "file", "line", "tail"):
                    try:
                        _val = _data.group(_key)
                        if _val:
                            _tmp[_key] = _val
                    except Exception:
                        pass
            return _tmp if _tmp else None

        def perform_regex(_prefix):
            """Performs a search for standard configurations of function + file + line"""
            import re

            _tmp = None
            for _pattern in [
                # [func][file]
                r"(^\[)(?P<func>.*)(\]\[)(?P<file>.*)(\]$)",
                # label  [func][file:line]
                r"(?P<head>.+)([ \t]+)\[(?P<func>\S+)\]\[(?P<file>\S+):(?P<line>[0-9]+)\]$",
                # label  [func/file:line]
                r"(?P<head>.+)([ \t]+)\[(?P<func>\S+)([/])(?P<file>\S+):(?P<line>[0-9]+)\]$",
                # func@file:line/tail
                # func/file:line/tail
                r"(?P<func>\S+)([@/])(?P<file>\S+):(?P<line>[0-9]+)[/]*(?P<tail>.*)",
                # func@file/tail
                # func/file/tail
                r"(?P<func>\S+)([@/])(?P<file>\S+)([/])(?P<tail>.*)",
                # func:line/tail
                r"(?P<func>\S+):(?P<line>[0-9]+)([/]*)(?P<tail>.*)",
            ]:
                _tmp = process_regex(re.search(_pattern, _prefix))
                if _tmp:
                    break
            return _tmp if _tmp else None

        def get_name_line_file(_prefix):
            """Get the standard set of dictionary entries.
            Also, parses the prefix for func-file-line info
            which is typically in the form:
                <FUNC>@<FILE>:<LINE>/...
                <FUNC>/<FILE>:<LINE>/...
                <SOURCE>    [<FUNC>/<FILE>:<LINE>]
            """
            _keys = {
                "type": "region",
                "name": _prefix,
            }
            _extra = {"file": "<unknown>", "line": "0"}
            _pdict = perform_regex(_prefix)
            if _pdict is not None:
                if "head" in _pdict:
                    _keys["name"] = _pdict["head"].rstrip()
                    _extra["line"] = _pdict["line"]
                    _extra["file"] = _pdict["file"]
                else:
                    _keys["name"] = _pdict["func"]
                    _extra["file"] = (
                        _pdict["file"] if "file" in _pdict else "<unknown file>"
                    )
                    if "line" in _pdict:
                        _extra["line"] = _pdict["line"]
                    if "tail" in _pdict:
                        _keys["name"] = "{}/{}".format(_keys["name"], _pdict["tail"])
            return (_keys, _extra)

        def format_labels(_labels):
            """Formats multi dimensional metrics which refer to multiple metrics
            stored in a 1D list.

            Example: PAPI_TOT_CYC, PAPI_TOT_INS, and PAPI_L2_TCM are stored as
            ["Total_cycles", "Instr_completed", "L2_cache_misses"].

            After formatting:
            ['Total-cycles', 'Instr-completed', 'L2-cache-misses']
            """
            _ret = []
            if isinstance(_labels, str):
                # put in a list if the label is a string.
                _ret = [_labels.lower()]
            elif isinstance(_labels, dict):
                for _key, _item in _labels.items():
                    _ret.append(
                        _key.strip().replace(" ", "-").replace("_", "-").lower()
                    )
            elif isinstance(_labels, list) or isinstance(_labels, tuple):
                for _item in _labels:
                    _ret.append(
                        _item.strip().replace(" ", "-").replace("_", "-").lower()
                    )
            return _ret

        def match_labels_and_values(_metric_stats, _metric_label, _metric_type):
            """Match metric labels with values and add '(inc)' if the metric type
            is inclusive.

            _metric_stat example 1: {'sum': 0.010, 'min': 0.001, ...}
            _metric_stat example 2: {'sum': [0.010, 0.020, 0.030], ...}

            _metric_label example 1: wall_clock
            _metric_label example 2: ['Total-cycles', 'Instr-completed', 'L2-cache-misses']

            _metric_type: '.inc' or ''
            """
            _ret = {}
            for _key, _item in _metric_stats.items():
                if isinstance(_item, dict):
                    for i, (k, v) in enumerate(_item.items()):
                        _ret["{}.{}{}".format(_key, _metric_label[i], _metric_type)] = v
                # match with metric labels if _metric_stat item is a list.
                elif isinstance(_item, list):
                    for i in range(len(_item)):
                        _ret[
                            "{}.{}{}".format(_key, _metric_label[i], _metric_type)
                        ] = _item[i]
                # check if _metric_stat item is not a dict or list
                else:
                    _ret["{}.{}{}".format(_key, _metric_label, _metric_type)] = _item
            return _ret

        def collapse_ids(_obj, _expect_scalar=False):
            """node/rank/thread id may be int, array of ints, or None.
            When the entry is a list of integers (which happens when metric values
            are aggregates of multiple ranks/threads), this function generates a consistent
            form which is NOT numerical to avoid `groupby(...).sum()` operations from producing
            something nonsensical (i.e. adding together thread-ids) but ensures the entry is
            still hashable (i.e. a list of integers is not hashable and will cause `groupby(...).sum()` to
            throw an error)

            Arguments:
                _obj (int or list of ints):
                    The node/rank/thread id(s) for the metric.
                    If a list is provided, this implies that the metric values are aggregations from multiple nodes/ranks/threads

                _expect_scalar (bool):
                    Setting this value to true means that `_obj` is expected to be an integer and the
                    return value should be converted to an integer. If this value is true and an array of ints
                    is passed, an error will be thrown

            Return Value:
                if _expect_scalar is False: string
                if _expect_scalar is True: int
            """
            if isinstance(_obj, list):
                if len(_obj) == 1:
                    return int(_obj[0])
                else:
                    if _expect_scalar:
                        raise ValueError(
                            f"collapse_ids expected per-rank or per-thread values but list of ids ({_obj}) implies that data is aggregated across multiple ranks or threads"
                        )
                    return ",".join([f"{x}" for x in _obj]).strip(",")
            elif _obj is not None:
                return f"{_obj}" if _expect_scalar else int(_obj)
            return None

        def parse_node(_metric_name, _node_data, _hparent, _rank, _parent_callpath):
            """Create callpath_to_node_dict for one node and then call the function
            recursively on all children.
            """

            # If the hash is zero, that indicates that the node
            # is a dummy for the root or is used for synchronizing data
            # between multiple threads
            # TODO: do we have some intermediate nodes that have hash = 0?
            if _node_data["node"]["hash"] == 0:
                if "children" in _node_data:
                    for _child in _node_data["children"]:
                        parse_node(
                            _metric_name,
                            _child,
                            _hparent,
                            _rank,
                            _parent_callpath,
                        )
                return

            _prop = self.properties[_metric_name]
            _frame_attrs, _extra = get_name_line_file(_node_data["node"]["prefix"])

            callpath = _parent_callpath + (_frame_attrs["name"],)

            # check if the node already exits.
            _hnode = self.callpath_to_node.get(callpath)
            if _hnode is None:
                # connect with the parent during node creation.
                _hnode = Node(Frame(_frame_attrs), _hparent)
                self.callpath_to_node[callpath] = _hnode
                if _hparent is None:
                    # if parent is none, this is a root node.
                    list_roots.append(_hnode)
                else:
                    # if parent is not none, add as a child.
                    _hparent.add_child(_hnode)

            # by placing the thread-id or rank-id in _frame_attrs, the hash
            # for the Frame(_keys) effectively circumvent Hatchet's
            # default behavior of combining similar thread/rank entries
            _tid_dict = _frame_attrs if self.per_thread else _extra
            _rank_dict = _frame_attrs if self.per_rank else _extra

            # handle the rank
            _rank_dict["rank"] = collapse_ids(_rank, self.per_rank)
            if _rank_dict["rank"] is None:
                del _rank_dict["rank"]
                self.include_nid = False

            # extract some relevant data
            _tid_dict["thread"] = collapse_ids(
                _node_data["node"]["tid"], self.per_thread
            )
            _extra["pid"] = collapse_ids(_node_data["node"]["pid"], False)
            _extra["count"] = _node_data["node"]["inclusive"]["entry"]["laps"]

            # check if there are multiple threads
            # TODO: move this outside if don't have per thread data in timemory
            if not self.multiple_threads:
                if _tid_dict["thread"] != 0:
                    self.multiple_threads = True

            # this is the name for the metrics
            _labels = None if "type" not in _prop else _prop["type"]
            # if the labels are not a single string, they are multi-dimensional
            _metrics_in_vector = True if not isinstance(_labels, str) else False
            # remove some spurious data from inclusive/exclusive stats
            _remove = ["cereal_class_version", "count"]
            _inc_stats = remove_keys(_node_data["node"]["inclusive"]["stats"], _remove)
            _exc_stats = remove_keys(_node_data["node"]["exclusive"]["stats"], _remove)

            # if multi-dimensions, create alternative "sum.<...>", etc. labels + data
            # add ".inc" to the end of every column that represents an inclusive stat
            if _metrics_in_vector:
                # Example of a multi-dimensional output: if we have 3 papi events
                # PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L2_TCM:
                # _metric_labels: ["Total_cycles", "Instr_completed", "L2_cache_misses"]
                # _exc_stats -> "sum": [8301.0, 4910.0, 275.0],
                _metric_labels = format_labels(_labels)
                _exc_stats = match_labels_and_values(_exc_stats, _metric_labels, "")
                _inc_stats = match_labels_and_values(_inc_stats, _metric_labels, ".inc")
            else:
                # add metric name and type.
                # Example: sum -> sum.wall_clock.inc
                _inc_stats = match_labels_and_values(_inc_stats, _metric_name, ".inc")
                # Example: sum -> sum.wallclock
                _exc_stats = match_labels_and_values(_exc_stats, _metric_name, "")

            # add the inclusive and exclusive columns to the list of relevant column names
            add_metrics(_exc_stats)
            add_metrics(_inc_stats)

            # we use callpath_to_node_dict instead of directly
            # using node_dicts to be able to merge metrics.
            # We use its values later as node_dicts.
            # (callpath, rank, thread): <node_dict>
            callpath_rank_thread = tuple((callpath, _rank, _tid_dict["thread"]))
            node_dict = self.callpath_to_node_dict.get(callpath_rank_thread)
            # check if we saw this (callpath, rank, thread) before
            if node_dict is None:
                # if no, create a new dict.
                self.callpath_to_node_dict[callpath_rank_thread] = dict(
                    {"node": _hnode, **_frame_attrs},
                    **_extra,
                    **_exc_stats,
                    **_inc_stats,
                )
            else:
                # if yes, don't create a new dict, just add the new metrics to
                # the existing node_dict using update().
                # we are doing this to combine different metrics on a single dataframe.
                self.callpath_to_node_dict[callpath_rank_thread].update(
                    dict(**_exc_stats, **_inc_stats)
                )

            # recursion
            if "children" in _node_data:
                for _child in _node_data["children"]:
                    parse_node(_metric_name, _child, _hnode, _rank, callpath)

        def read_graph(_metric_name, ranks_data, _rank):
            """The layout of the graph at this stage
            is subject to slightly different structures
            based on whether distributed memory parallelism (DMP)
            (e.g. MPI, UPC++) was supported and active

            Returns the last rank (_idx).
            """

            rank = None
            total_ranks = len(ranks_data)

            for i in range(total_ranks):
                # rank_data stores all the graph/cct data of a rank
                # starting from the first node in the cct.
                rank_data = ranks_data[i]
                rank = None if _rank is None else i + _rank
                if isinstance(rank_data, list):
                    for data in rank_data:
                        if len(data["children"]) != 0:
                            # empty tuple represents the parent callpath for the root node.
                            # third parameter is the parent node. It's none for the root node.
                            parse_node(
                                _metric_name,
                                data,
                                None,
                                rank,
                                tuple(),
                            )
                else:
                    if len(rank_data["children"]) != 0:
                        # empty tuple represents the parent callpath for the root node.
                        # third parameter is the parent node. It's none for the root node.
                        parse_node(_metric_name, rank_data, None, rank, tuple())

            if total_ranks > 0:
                return True
            return False

        def read_properties(properties, _metric_name, _metric_data):
            """Read in the properties for a component. This
            contains information on the type of the component,
            a description, a unit_value relative to the
            standard, a unit label, whether the data is
            only relevant per-thread, the number of MPI and/or
            UPC++ ranks (some results can theoretically use
            both UPC++ and MPI), the number of threads in
            the application, and the total number of processes
            """
            if _metric_name not in properties:
                properties[_metric_name] = {}
            try:
                properties[_metric_name]["properties"] = remove_keys(
                    _metric_data["properties"], "cereal_class_version"
                )
            except KeyError:
                pass
            for p in (
                "type",
                "description",
                "unit_value",
                "unit_repr",
                "thread_scope_only",
                "mpi_size",
                "upcxx_size",
                "thread_count",
                "process_count",
            ):
                if (
                    p not in properties[_metric_name]
                    or properties[_metric_name][p] is None
                ):
                    if p in _metric_data:
                        properties[_metric_name][p] = _metric_data[p]
                    else:
                        properties[_metric_name][p] = None

        # graph_dict[timemory] stores all metric data.
        # each metric data is another item in this dict.
        for metric_name, metric_data in self.graph_dict["timemory"].items():
            # strip out the namespace if provided
            metric_name = (
                metric_name.replace("tim::", "").replace("component::", "").lower()
            )
            # check for selection
            if self.select is not None and metric_name not in self.select:
                continue
            # read in properties
            read_properties(self.properties, metric_name, metric_data)
            # if no DMP supported
            if "graph" in metric_data:
                read_graph(metric_name, metric_data["graph"], None)
            else:
                # read in MPI results
                if "mpi" in metric_data:
                    self.multiple_ranks = read_graph(metric_name, metric_data["mpi"], 0)
                # if MPI and UPC++, report ranks
                # offset by MPI_Comm_size
                _rank = self.properties[metric_name]["mpi_size"]
                _rank = 0 if _rank is None else int(_rank)
                if "upc" in metric_data:
                    self.multiple_ranks = read_graph(
                        metric_name, metric_data["upc"], _rank
                    )
                elif "upcxx" in metric_data:
                    self.multiple_ranks = read_graph(
                        metric_name, metric_data["upcxx"], _rank
                    )

        # create the graph of the roots
        graph = Graph(list_roots)
        graph.enumerate_traverse()

        # separate out the inclusive vs. exclusive columns
        exc_metrics = []
        inc_metrics = []
        for column in self.metric_cols:
            if column.endswith(".inc"):
                inc_metrics.append(column)
            else:
                exc_metrics.append(column)

        # set the default metric
        if self.default_metric is None:
            if len(exc_metrics) > 0:
                if "sum.wall_clock" in exc_metrics:
                    self.default_metric = "sum.wall_clock"
                elif "sum.cpu_clock" in exc_metrics:
                    self.default_metric = "sum.cpu_clock"
                else:
                    self.default_metric = exc_metrics[0]
            elif len(inc_metrics) > 0:
                self.default_metric = inc_metrics[0]
            else:
                self.default_metric = "sum"

        node_dicts = list(self.callpath_to_node_dict.values())
        dataframe = pd.DataFrame(data=node_dicts)

        indices = []
        # Set indices according to rank/thread numbers.
        if self.multiple_ranks and self.multiple_threads:
            indices = ["node", "rank", "thread"]
        elif self.multiple_ranks:
            dataframe.drop(columns=["thread"], inplace=True)
            indices = ["node", "rank"]
        elif self.multiple_threads:
            dataframe.drop(columns=["rank"], inplace=True)
            indices = ["node", "thread"]
        else:
            indices = ["node"]

        dataframe.set_index(indices, inplace=True)
        dataframe.sort_index(inplace=True)

        # Fill the missing ranks
        # After unstacking and iterating over rows, there
        # will be "NaN" values for some ranks. Find the first
        # rank that has notna value and use it for other rows/ranks
        # of the multiindex.
        # TODO: iterrows() is not the best way to iterate over rows.
        if self.multiple_ranks or self.multiple_threads:
            dataframe = dataframe.unstack()
            for idx, row in dataframe.iterrows():
                # There is always a valid name for an index.
                # Take that valid name and assign to other ranks/rows.
                name = row["name"][row["name"].first_valid_index()]
                dataframe.loc[idx, "name"] = name

                # Sometimes there is no file information.
                if row["file"].first_valid_index() is not None:
                    file = row["file"][row["file"].first_valid_index()]
                    dataframe.loc[idx, "file"] = file

                # Sometimes there is no file information.
                if row["type"].first_valid_index() is not None:
                    file = row["type"][row["type"].first_valid_index()]
                    dataframe.loc[idx, "type"] = file

                # Fill the rest with 0
                dataframe.fillna(0, inplace=True)

            # Stack the dataframe
            dataframe = dataframe.stack()

        return GraphFrame(
            graph,
            dataframe,
            exc_metrics,
            inc_metrics,
            default_metric=self.default_metric,
            metadata=self.metadata,
        )

    def read_metadata(self, _inp):
        _metadata = {}
        # check if the input is a dictionary.
        if isinstance(_inp, dict):
            _metadata = _inp if "timemory" not in _inp else _inp["timemory"]
        # check if the input is a directory and get '.tree.json' files if true.
        elif os.path.isdir(_inp):
            tree_files = glob.glob(_inp + "/*metadata*.json")
            for file in tree_files:
                # read all files that end with .tree.json.
                with open(file, "r") as f:
                    # add all metrics to the same dict even though timemory
                    # creates a separate file for each metric.
                    _metadata = {**_metadata, **json.load(f)["timemory"]}
        # check if the input is a filename that ends in json
        elif isinstance(_inp, str) and _inp.endswith("json"):
            with open(_inp, "r") as f:
                _metadata = json.load(f)["timemory"]
        elif not isinstance(_inp, str):
            _metadata = json.loads(_inp.read())["timemory"]
        else:
            raise TypeError("input must be dict, directory, json file, or string")

        if "metadata" in _metadata:
            for _key, _item in _metadata["metadata"].items():
                if _key == "info":
                    for _entry, _value in _item.items():
                        self.metadata[_entry] = _value
                elif _key == "environment":
                    for itr in _item:
                        self.metadata[_item["key"]] = _item["value"]
                elif _key == "settings":
                    self.metadata["settings"] = _item
        elif "metadata_file" in _metadata:
            with open(_metadata["metadata_file"], "r") as f:
                self.read_metadata(json.load(f))

    def read(self):
        """Read timemory json."""

        def _read_metadata(_inp, _data):
            _mfile = os.path.join(os.path.basename(_inp), "metadata.json")
            if os.path.exists(_mfile) and os.path.isfile(_mfile):
                with open(_mfile, "r") as f:
                    self.read_metadata(json.load(f))
            elif _data is not None:
                self.read_metadata(_data)

        # check if the input is a dictionary.
        if isinstance(self.input, dict):
            self.graph_dict = self.input
            self.read_metadata(self.input)
        # check if the input is a directory and get '.tree.json' files if true.
        elif os.path.isdir(self.input):
            tree_files = glob.glob(self.input + "/*.tree.json")
            for file in tree_files:
                # read all files that end with .tree.json.
                with open(file, "r") as f:
                    # add all metrics to the same dict even though timemory
                    # creates a separate file for each metric.
                    _data = json.load(f)
                    self.graph_dict["timemory"].update(_data["timemory"])
                    _read_metadata(file, _data)
        # check if the input is a filename that ends in json
        elif isinstance(self.input, str) and self.input.endswith("json"):
            with open(self.input, "r") as f:
                _data = json.load(f)
                self.graph_dict = _data
                _read_metadata(self.input, _data)
        elif not isinstance(self.input, str):
            _data = json.loads(self.input.read())
            self.graph_dict = _data
            self.read_metadata(_data)
        else:
            raise TypeError("input must be dict, directory, json file, or string")

        return self.create_graph()
