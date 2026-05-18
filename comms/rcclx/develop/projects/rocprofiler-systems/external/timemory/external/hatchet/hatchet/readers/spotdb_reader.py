# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd

import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.frame import Frame
from hatchet.util.timer import Timer


def _find_child_node(node, name):
    """Return child with given name from parent node"""
    for c in node.children:
        if c.frame.get("name") == name:
            return c
    return None


class SpotDatasetReader:
    """Reads a (single-run) dataset from SpotDB"""

    def __init__(self, regionprofile, metadata, attr_info):
        """Initialize SpotDataset reader

        Args:
            regionprofile (dict): Dict with region names to key:value record with
                metrics. Region names are hierarchical, separated with '/'.
                Example:
                { "a/b/c": { "metric": val, ... }, ... }

            metadata: (dict): Key-value run metadata for this dataset. Example:
                { "launchdate": 123456789, "figure_of_merit": 42.0 }

            attr_info (dict): Information about metric attributes. Contains, e.g.,
                type and alias info. This data is optional. Example:
                { "metric": { "type": "double", "alias": "The Metric", ... }, ... }
        """

        self.regionprofile = regionprofile
        self.attr_info = attr_info
        self.metadata = metadata
        self.df_data = []
        self.roots = {}
        self.metric_columns = set()

        self.timer = Timer()

    def create_graph(self):
        """Create the graph. Fills in df_data and metric_columns."""

        self.df_data[:] = []  # clear the list

        for pathstr, vals in self.regionprofile.items():
            # parse { "a/b/c": { "metric": val, ... }, ... } records
            if len(pathstr) == 0:
                continue

            path = pathstr.split("/")
            name = path[-1]
            node = self._create_node(path)

            metrics = {}
            for k, v in vals.items():
                info = self.attr_info.get(k, dict())
                colm = info.get("alias", k)
                type = info.get("type", "string")
                if "inclusive" in k:
                    colm += " (inc)"

                if type == "double":
                    metrics[colm] = float(v)
                elif type == "int" or type == "uint":
                    metrics[colm] = int(v)
                else:
                    metrics[colm] = v
                self.metric_columns.add(colm)

            self.df_data.append(dict({"name": name, "node": node}, **metrics))

    def read(self, default_metric="Total time"):
        """Create GraphFrame for the given Spot dataset."""

        with self.timer.phase("graph construction"):
            self.create_graph()

        graph = Graph(list(self.roots.values()))
        graph.enumerate_traverse()

        dataframe = pd.DataFrame(data=self.df_data)
        dataframe.set_index(["node"], inplace=True)

        exc_metrics = []
        inc_metrics = []
        for m in self.metric_columns:
            if "(inc)" in m:
                inc_metrics.append(m)
            else:
                exc_metrics.append(m)

        if default_metric not in dataframe.columns:
            if len(exc_metrics) > 0:
                default_metric = exc_metrics[0]
            elif len(inc_metrics) > 0:
                default_metric = inc_metrics[0]

        return hatchet.graphframe.GraphFrame(
            graph,
            dataframe,
            exc_metrics,
            inc_metrics,
            default_metric=default_metric,
            metadata=self.metadata,
        )

    def _create_node(self, path):
        parent = self.roots.get(path[0], None)
        if parent is None:
            parent = Node(Frame(name=path[0]))
            self.roots[path[0]] = parent

        node = parent
        for name in path[1:]:
            node = _find_child_node(parent, name)
            if node is None:
                node = Node(Frame(name=name), parent)
                parent.add_child(node)
            parent = node

        return node


class SpotDBReader:
    """Import multiple runs as graph frames from a SpotDB instance"""

    def __init__(self, db_key, list_of_ids=None, default_metric="Total time (inc)"):
        """Initialize SpotDBReader

        Args:
            db_key (str or SpotDB object): locator for SpotDB instance
                This can be a SpotDB object directly, or a locator for a spot
                database, which is a string with either
                    * A directory for .cali files,
                    * A .sqlite file name
                    * A SQL database URL (e.g., "mysql://hostname/db")

            list_of_ids: The list of run IDs to read from the database.
                If this is None, returns all runs.

            default_metric: Name of the default metric for the GraphFrames.
        """
        self.db_key = db_key
        self.list_of_ids = list_of_ids
        self.default_metric = default_metric

    def read(self):
        """Read given runs from SpotDB

        Returns:
            List of GraphFrames, one for each entry that was found
        """
        import spotdb

        if isinstance(self.db_key, str):
            db = spotdb.connect(self.db_key)
        else:
            db = self.db_key

        runs = self.list_of_ids or db.get_all_run_ids()

        regionprofiles = db.get_regionprofiles(runs)
        metadata = db.get_global_data(runs)
        attr_info = db.get_metric_attribute_metadata()

        result = []

        for run in runs:
            if run in regionprofiles:
                result.append(
                    SpotDatasetReader(
                        regionprofiles[run], metadata[run], attr_info
                    ).read(self.default_metric)
                )

        return result
