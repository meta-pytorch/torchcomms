# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT


import pandas as pd
import os

import caliperreader as cr

import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.frame import Frame
from hatchet.util.timer import Timer


class CaliperNativeReader:
    """Read in a native `.cali` file using Caliper's python reader."""

    def __init__(self, filename_or_caliperreader):
        """Read in a native cali using Caliper's python reader.

        Args:
            filename_or_caliperreader (str or CaliperReader): name of a `cali` file OR
                a CaliperReader object
        """
        self.filename_or_caliperreader = filename_or_caliperreader
        self.filename_ext = ""

        self.df_nodes = {}
        self.metric_cols = []
        self.record_data_cols = []
        self.node_dicts = []
        self.callpath_to_node = {}
        self.idx_to_node = {}
        self.callpath_to_idx = {}
        self.global_nid = 0

        self.default_metric = None

        self.timer = Timer()

        if isinstance(self.filename_or_caliperreader, str):
            _, self.filename_ext = os.path.splitext(filename_or_caliperreader)

    def read_metrics(self, ctx="path"):
        all_metrics = []
        records = self.filename_or_caliperreader.records

        # read metadata from the caliper reader
        for record in records:
            node_dict = {}
            if ctx in record:
                # get the node label and callpath for the record
                if isinstance(record[ctx], list):
                    # specify how to parse cupti records
                    if "cupti.activity.kind" in record:
                        if record["cupti.activity.kind"] == "kernel":
                            node_label = record["cupti.kernel.name"]
                            node_callpath = tuple(record[ctx] + [node_label])
                        elif record["cupti.activity.kind"] == "memcpy":
                            node_label = record["cupti.activity.kind"]
                            node_callpath = tuple(record[ctx] + [node_label])
                    else:
                        node_label = record[ctx][-1]
                        node_callpath = tuple(record[ctx])
                else:
                    node_label = record[ctx][-1]
                    node_callpath = tuple([record[ctx]])

                # get node nid based on callpath
                node_dict["nid"] = self.callpath_to_idx.get(node_callpath)

                for item in record.keys():
                    if item != ctx:
                        if item not in self.record_data_cols:
                            self.record_data_cols.append(item)

                        if (
                            self.filename_or_caliperreader.attribute(
                                item
                            ).attribute_type()
                            == "double"
                        ):
                            node_dict[item] = float(record[item])
                        elif (
                            self.filename_or_caliperreader.attribute(
                                item
                            ).attribute_type()
                            == "int"
                        ):
                            node_dict[item] = int(record[item])
                        elif item == "function":
                            if isinstance(record[item], list):
                                node_dict[item] = record[item][-1]
                            else:
                                node_dict[item] = record[item]
                        else:
                            node_dict[item] = record[item]

                all_metrics.append(node_dict)

        # make list of metric columns
        for col in self.record_data_cols:
            if self.filename_or_caliperreader.attribute(col).is_value():
                self.metric_cols.append(col)

        df_metrics = pd.DataFrame.from_dict(data=all_metrics)
        return df_metrics

    def create_graph(self, ctx="path"):
        def _create_parent(child_node, parent_callpath):
            """We may encounter a parent node in the callpath before we see it
            as a child node. In this case, we need to create a hatchet node for
            the parent.

            This function recursively creates parent nodes in a callpath
            until it reaches the already existing parent in that callpath.
            """
            parent_node = self.callpath_to_node.get(parent_callpath)

            if parent_node:
                # return if arrives at the parent
                parent_node.add_child(child_node)
                child_node.add_parent(parent_node)
                return
            else:
                # else create the parent and add parent/child
                grandparent_callpath = parent_callpath[:-1]
                parent_name = parent_callpath[-1]

                parent_node = Node(
                    Frame({"type": "function", "name": parent_name}), None
                )

                self.callpath_to_node[parent_callpath] = parent_node
                self.callpath_to_idx[parent_callpath] = self.global_nid

                node_dict = dict(
                    {"name": parent_name, "node": parent_node, "nid": self.global_nid},
                )
                self.idx_to_node[self.global_nid] = node_dict
                self.global_nid += 1

                parent_node.add_child(child_node)
                child_node.add_parent(parent_node)
                _create_parent(parent_node, grandparent_callpath)

        list_roots = []
        parent_hnode = None
        records = self.filename_or_caliperreader.records

        for record in records:
            node_label = ""
            if ctx in record:
                # if it's a list, then it's a callpath
                if isinstance(record[ctx], list):
                    # specify how to parse cupti records
                    if "cupti.activity.kind" in record:
                        if record["cupti.activity.kind"] == "kernel":
                            node_label = record["cupti.kernel.name"]
                            node_callpath = tuple(record[ctx] + [node_label])
                            parent_callpath = node_callpath[:-1]
                            node_type = "kernel"
                        elif record["cupti.activity.kind"] == "memcpy":
                            node_label = record["cupti.activity.kind"]
                            node_callpath = tuple(record[ctx] + [node_label])
                            parent_callpath = node_callpath[:-1]
                            node_type = "memcpy"
                        else:
                            Exception("Haven't seen this activity kind yet")
                    else:
                        node_label = record[ctx][-1]
                        node_callpath = tuple(record[ctx])
                        parent_callpath = node_callpath[:-1]
                        node_type = "function"

                    hnode = self.callpath_to_node.get(node_callpath)

                    if not hnode:
                        frame = Frame({"type": node_type, "name": node_label})
                        hnode = Node(frame, None)
                        self.callpath_to_node[node_callpath] = hnode

                        # get parent from node callpath
                        parent_hnode = self.callpath_to_node.get(parent_callpath)

                        # create parent if it doesn't exist
                        # else if parent already exists, add child-parent
                        if not parent_hnode:
                            _create_parent(hnode, parent_callpath)
                        else:
                            parent_hnode.add_child(hnode)
                            hnode.add_parent(parent_hnode)

                        self.callpath_to_idx[node_callpath] = self.global_nid
                        node_dict = dict(
                            {"name": node_label, "node": hnode, "nid": self.global_nid},
                        )
                        self.idx_to_node[self.global_nid] = node_dict
                        self.global_nid += 1

                # if it's a string, then it's a root
                else:
                    root_label = record[ctx]
                    root_callpath = tuple([root_label])

                    if root_callpath not in self.callpath_to_node:
                        # create the root since it doesn't exist
                        frame = Frame({"type": "function", "name": root_label})
                        graph_root = Node(frame, None)

                        # store callpaths to identify the root
                        self.callpath_to_node[root_callpath] = graph_root
                        self.callpath_to_idx[root_callpath] = self.global_nid
                        list_roots.append(graph_root)

                        node_dict = dict(
                            {
                                "name": root_label,
                                "node": graph_root,
                                "nid": self.global_nid,
                            }
                        )

                        self.idx_to_node[self.global_nid] = node_dict
                        self.global_nid += 1

        return list_roots

    def read(self):
        """Read the caliper records to extract the calling context tree."""
        if isinstance(self.filename_or_caliperreader, str):
            if self.filename_ext != ".cali":
                raise ValueError("from_caliperreader() needs a .cali file")
            else:
                cali_file = self.filename_or_caliperreader
                self.filename_or_caliperreader = cr.CaliperReader()
                self.filename_or_caliperreader.read(cali_file)

        with self.timer.phase("graph construction"):
            list_roots = self.create_graph()
        self.df_nodes = pd.DataFrame(data=list(self.idx_to_node.values()))

        # create a graph object once all the nodes have been added
        graph = Graph(list_roots)
        graph.enumerate_traverse()

        with self.timer.phase("read metrics"):
            df_fixed_data = self.read_metrics()

        metrics = pd.DataFrame.from_dict(data=df_fixed_data)

        # add missing intermediate nodes to the df_fixed_data dataframe
        if "mpi.rank" in df_fixed_data.columns:
            num_ranks = metrics["mpi.rank"].max() + 1
            rank_list = range(0, num_ranks)

        # create a standard dict to be used for filling all missing rows
        default_metric_dict = {}
        for idx, col in enumerate(self.record_data_cols):
            if self.filename_or_caliperreader.attribute(col).is_value():
                default_metric_dict[list(self.record_data_cols)[idx]] = 0
            else:
                default_metric_dict[list(self.record_data_cols)[idx]] = None

        # create a list of dicts, one dict for each missing row
        missing_nodes = []
        for iteridx, row in self.df_nodes.iterrows():
            # check if df_nodes row exists in df_fixed_data
            metric_rows = df_fixed_data.loc[metrics["nid"] == row["nid"]]
            if "mpi.rank" not in self.metric_cols:
                if metric_rows.empty:
                    # add a single row
                    node_dict = dict(default_metric_dict)
                    missing_nodes.append(node_dict)
            else:
                if metric_rows.empty:
                    # add a row per MPI rank
                    for rank in rank_list:
                        node_dict = dict(default_metric_dict)
                        node_dict["nid"] = row["nid"]
                        node_dict["mpi.rank"] = rank
                        missing_nodes.append(node_dict)
                elif len(metric_rows) < num_ranks:
                    # add a row for each missing MPI rank
                    present_ranks = metric_rows["mpi.rank"].values
                    missing_ranks = [x for x in rank_list if x not in present_ranks]
                    for rank in missing_ranks:
                        node_dict = dict(default_metric_dict)
                        node_dict["nid"] = row["nid"]
                        node_dict["mpi.rank"] = rank
                        missing_nodes.append(node_dict)

        df_missing = pd.DataFrame.from_dict(data=missing_nodes)
        df_metrics = pd.concat([df_fixed_data, df_missing], sort=False)

        # dict mapping old to new column names to make columns consistent with
        # other readers
        old_to_new = {
            "mpi.rank": "rank",
            "module#cali.sampler.pc": "module",
            "sum#time.duration": "time",
            "sum#avg#sum#time.duration": "time",
            "inclusive#sum#time.duration": "time (inc)",
            "sum#avg#inclusive#sum#time.duration": "time (inc)",
        }

        # change column names
        new_cols = []
        for col in df_metrics.columns:
            if col in old_to_new:
                new_cols.append(old_to_new[col])
            else:
                new_cols.append(col)
        df_metrics.columns = new_cols

        # create list of exclusive and inclusive metric columns
        exc_metrics = []
        inc_metrics = []
        for column in self.metric_cols:
            # ignore rank as an exc or inc metric
            if column == "mpi.rank":
                continue

            # add new column names to list of metrics if inc or inclusive in
            # old column names
            if "(inc)" in column or "inclusive" in column:
                if column in old_to_new:
                    column = old_to_new[column]
                inc_metrics.append(column)
            else:
                if column in old_to_new:
                    column = old_to_new[column]
                exc_metrics.append(column)

        with self.timer.phase("data frame"):
            # merge the metrics and node dataframes on the nid column
            dataframe = pd.merge(df_metrics, self.df_nodes, on="nid")

            # set the index to be a MultiIndex
            indices = ["node"]
            if "rank" in dataframe.columns:
                indices.append("rank")
            dataframe.set_index(indices, inplace=True)
            dataframe.sort_index(inplace=True)

        # set the default metric
        if self.default_metric is None:
            if "time (inc)" in dataframe.columns:
                self.default_metric = "time"
            elif "avg#inclusive#sum#time.duration" in dataframe.columns:
                self.default_metric = "avg#inclusive#sum#time.duration"
            elif len(inc_metrics) > 0:
                self.default_metric = inc_metrics[0]
            elif len(exc_metrics) > 0:
                self.default_metric = exc_metrics[0]

        metadata = self.filename_or_caliperreader.globals

        return hatchet.graphframe.GraphFrame(
            graph,
            dataframe,
            exc_metrics,
            inc_metrics,
            self.default_metric,
            metadata=metadata,
        )
