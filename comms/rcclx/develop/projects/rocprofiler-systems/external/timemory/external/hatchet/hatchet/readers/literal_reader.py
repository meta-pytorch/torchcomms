# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd

import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.frame import Frame


class LiteralReader:
    """Create a GraphFrame from a list of dictionaries.

    TODO: calculate inclusive metrics automatically.

    Example:

    .. code-block:: console

        dag_ldict = [
            {
                "frame": {"name": "A", "type": "function"},
                "metrics": {"time (inc)": 30.0, "time": 0.0},
                "children": [
                    {
                        "frame": {"name": "B",  "type": "function"},
                        "metrics": {"time (inc)": 11.0, "time": 5.0},
                        "children": [
                            {
                                "frame": {"name": "C", "type": "function"},
                                "metrics": {"time (inc)": 6.0, "time": 5.0},
                                "children": [
                                    {
                                        "frame": {"name": "D", "type": "function"},
                                        "metrics": {"time (inc)": 1.0, "time": 1.0},
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "frame": {"name": "E", "type": "function"},
                        "metrics": {"time (inc)": 19.0, "time": 10.0},
                        "children": [
                            {
                                "frame": {"name": "H", "type": "function"},
                                "metrics": {"time (inc)": 9.0, "time": 9.0}
                            }
                        ],
                    },
                ],
            }
        ]

    Return:
        (GraphFrame): graphframe containing data from dictionaries
    """

    def __init__(self, graph_dict):
        """Read from list of dictionaries.

        graph_dict (dict): List of dictionaries encoding nodes.
        """
        self.graph_dict = graph_dict

    def parse_node_literal(
        self, frame_to_node_dict, node_dicts, child_dict, hparent, seen_nids
    ):
        """Create node_dict for one node and then call the function
        recursively on all children.
        """

        # pull out _hatchet_nid if it exists
        # so it will not be inserted into
        # dataframe like a normal metric
        hnid = -1
        if "_hatchet_nid" in child_dict["metrics"]:
            hnid = child_dict["metrics"]["_hatchet_nid"]

        frame = Frame(child_dict["frame"])
        if hnid not in seen_nids:
            hnode = Node(frame, hparent, hnid=hnid)

            # depending on the node type, the name may not be in the frame
            node_name = child_dict["frame"].get("name")
            if not node_name:
                node_name = child_dict["name"]

            node_dict = dict(
                {"node": hnode, "name": node_name}, **child_dict["metrics"]
            )

            node_dicts.append(node_dict)
            frame_to_node_dict[frame] = hnode

            if hnid != -1:
                seen_nids.append(hnid)

        else:
            hnode = frame_to_node_dict.get(frame)

        hparent.add_child(hnode)

        if "children" in child_dict:
            for child in child_dict["children"]:
                self.parse_node_literal(
                    frame_to_node_dict, node_dicts, child, hnode, seen_nids
                )

    def read(self):
        list_roots = []
        node_dicts = []
        frame_to_node_dict = {}
        frame = None
        seen_nids = []
        hnid = -1

        # start with creating a node_dict for each root
        for i in range(len(self.graph_dict)):
            if "_hatchet_nid" in self.graph_dict[i]["metrics"]:
                hnid = self.graph_dict[i]["metrics"]["_hatchet_nid"]
                seen_nids.append(hnid)
            frame = Frame(self.graph_dict[i]["frame"])
            graph_root = Node(frame, None, hnid=hnid)

            # depending on the node type, the name may not be in the frame
            node_name = self.graph_dict[i]["frame"].get("name")
            if not node_name:
                node_name = self.graph_dict[i]["name"]

            node_dict = dict(
                {"node": graph_root, "name": node_name}, **self.graph_dict[i]["metrics"]
            )
            node_dicts.append(node_dict)

            list_roots.append(graph_root)
            frame_to_node_dict[frame] = graph_root

            # call recursively on all children of root
            if "children" in self.graph_dict[i]:
                for child in self.graph_dict[i]["children"]:
                    self.parse_node_literal(
                        frame_to_node_dict, node_dicts, child, graph_root, seen_nids
                    )

        graph = Graph(list_roots)

        # test if nids are already loaded
        if -1 in [n._hatchet_nid for n in graph.traverse()]:
            graph.enumerate_traverse()
        else:
            graph.enumerate_depth()

        exc_metrics = []
        inc_metrics = []
        for key in self.graph_dict[i]["metrics"].keys():
            if "(inc)" in key:
                inc_metrics.append(key)
            else:
                exc_metrics.append(key)

        dataframe = pd.DataFrame(data=node_dicts)
        dataframe.set_index(["node"], inplace=True)
        dataframe.sort_index(inplace=True)

        return hatchet.graphframe.GraphFrame(graph, dataframe, exc_metrics, inc_metrics)
