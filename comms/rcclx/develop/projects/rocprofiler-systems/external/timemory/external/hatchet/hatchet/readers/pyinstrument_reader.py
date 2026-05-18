# Copyright 2020-2023 University of Maryland and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import json

import pandas as pd

import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.frame import Frame


class PyinstrumentReader:
    def __init__(self, filename):
        self.pyinstrument_json_filename = filename
        self.graph_dict = {}
        self.list_roots = []
        self.node_dicts = []

    def create_graph(self):
        def parse_node_literal(child_dict, hparent):
            """Create node_dict for one node and then call the function
            recursively on all children."""

            hnode = Node(
                Frame({"name": child_dict["function"], "type": "function"}), hparent
            )

            child_node_dict = {
                "node": hnode,
                "name": child_dict["function"],
                "file": child_dict["file_path_short"],
                "line": child_dict["line_no"],
                "time": child_dict["time"],
                "time (inc)": child_dict["time"],
                "is_application_code": child_dict["is_application_code"],
            }

            hparent.add_child(hnode)
            self.node_dicts.append(child_node_dict)

            if "children" in child_dict:
                for child in child_dict["children"]:
                    # Pyinstrument's time metric actually stores inclusive time.
                    # To calculate exclusive time, we subtract the children's time
                    # from the parent's time.
                    child_node_dict["time"] -= child["time"]
                    parse_node_literal(child, hnode)

        # start with creating a node_dict for each root
        graph_root = Node(
            Frame(
                {"name": self.graph_dict["root_frame"]["function"], "type": "function"}
            ),
            None,
        )

        node_dict = {
            "node": graph_root,
            "name": self.graph_dict["root_frame"]["function"],
            "file": self.graph_dict["root_frame"]["file_path_short"],
            "line": self.graph_dict["root_frame"]["line_no"],
            "time": self.graph_dict["root_frame"]["time"],
            "time (inc)": self.graph_dict["root_frame"]["time"],
            "is_application_code": self.graph_dict["root_frame"]["is_application_code"],
        }

        self.node_dicts.append(node_dict)
        self.list_roots.append(graph_root)

        # call recursively on all children of root
        if "children" in self.graph_dict["root_frame"]:
            for child in self.graph_dict["root_frame"]["children"]:
                # Pyinstrument's time metric actually stores inclusive time.
                # To calculate exclusive time, we subtract the children's time
                # from the parent's time.
                node_dict["time"] -= child["time"]
                parse_node_literal(child, graph_root)

        graph = Graph(self.list_roots)
        graph.enumerate_traverse()

        return graph

    def read(self):
        with open(self.pyinstrument_json_filename) as pyinstrument_json:
            self.graph_dict = json.load(pyinstrument_json)

        graph = self.create_graph()

        dataframe = pd.DataFrame(data=self.node_dicts)
        dataframe.set_index(["node"], inplace=True)
        dataframe.sort_index(inplace=True)

        return hatchet.graphframe.GraphFrame(graph, dataframe, ["time"], ["time (inc)"])
