# Copyright 2022-2023 University of Oregon and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd

import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.frame import Frame
import glob
import json


class ApexReader:
    """Create a GraphFrame from a directory of APEX tasktree output JSON files.
    Example input:
    { "frame": {"name": "APEX MAIN", "type": "function", "rank": 0}, "metrics": {"time": 0.341264, "time (inc)": 0.341896, "min (inc)": 0.341896, "max (inc)": 0.341896, "sumsqr (inc)": 0.116893, "calls": 1.000000}, "children": [
     { "frame": {"name": "main", "type": "function", "rank": 0}, "metrics": {"time": 0.000059, "time (inc)": 0.000632, "min (inc)": 0.000632, "max (inc)": 0.000632, "sumsqr (inc)": 0.000000, "calls": 1.000000}, "children": [
      { "frame": {"name": "root_node", "type": "function", "rank": 0}, "metrics": {"time": 0.000217, "time (inc)": 0.000452, "min (inc)": 0.000452, "max (inc)": 0.000452, "sumsqr (inc)": 0.000000, "calls": 1.000000}, "children": [
       { "frame": {"name": "MPI_Recv", "type": "function", "rank": 0}, "metrics": {"time": 0.000027, "time (inc)": 0.000027, "min (inc)": 0.000002, "max (inc)": 0.000018, "sumsqr (inc)": 0.000000, "calls": 3.000000} },
       { "frame": {"name": "do_work", "type": "function", "rank": 0}, "metrics": {"time": 0.000111, "time (inc)": 0.000111, "min (inc)": 0.000111, "max (inc)": 0.000111, "sumsqr (inc)": 0.000000, "calls": 1.000000} },
       { "frame": {"name": "MPI_Send", "type": "function", "rank": 0}, "metrics": {"time": 0.000019, "time (inc)": 0.000019, "min (inc)": 0.000002, "max (inc)": 0.000008, "sumsqr (inc)": 0.000000, "calls": 6.000000} },
       { "frame": {"name": "get_next_work_item", "type": "function", "rank": 0}, "metrics": {"time": 0.000068, "time (inc)": 0.000079, "min (inc)": 0.000006, "max (inc)": 0.000031, "sumsqr (inc)": 0.000000, "calls": 8.000000}, "children": [
        { "frame": {"name": "get_work_items", "type": "function", "rank": 0}, "metrics": {"time": 0.000011, "time (inc)": 0.000011, "min (inc)": 0.000011, "max (inc)": 0.000011, "sumsqr (inc)": 0.000000, "calls": 1.000000} }
       ]
       }
      ]
      },
      { "frame": {"name": "MPI_Barrier", "type": "function", "rank": 0}, "metrics": {"time": 0.000120, "time (inc)": 0.000120, "min (inc)": 0.000017, "max (inc)": 0.000103, "sumsqr (inc)": 0.000000, "calls": 2.000000} }
     ]
     }
    ]
    }

    Return:
        (GraphFrame): graphframe containing data from dictionaries
    """

    def __init__(self, dirname):
        """Read from APEX tasktree json files in directory name

        dirname (string): Path to directory with tasktree.*.json files from APEX
        """
        self.warning = False
        self.dirname = dirname
        self.rank = 0
        self.ranks = []
        all_data = []
        files = sorted(glob.glob(dirname + "/tasktree.*.json"))
        for file in files:
            # get the rank number
            self.rank = file.split(".")[1]
            self.ranks.append(self.rank)
            # open text file in read mode
            with open(file) as json_file:
                data = json.load(json_file)
                all_data.append(data)
        self.graph_dict = all_data

    def parse_node_apex(
        self, frame_to_node_dict, node_dicts, child_dict, hparent, rank
    ):
        """Create node_dict for one node and then call the function
        recursively on all children.
        """

        if "rank" in child_dict["frame"]:
            rank = child_dict["frame"]["rank"]
            del child_dict["frame"]["rank"]
        if "type" not in child_dict["frame"]:
            child_dict["frame"]["type"] = "function"
        frame = Frame(child_dict["frame"])
        hnode = None
        found = False
        # See if the child exists already in the graph
        for child in hparent.children:
            if child.frame == frame:
                hnode = child
                found = True
                break
        # Doesn't exist, so add it
        if found is False:
            hnode = Node(frame, hparent, hnid=-1)

        # depending on the node type, the name may not be in the frame
        node_name = child_dict["frame"].get("name")
        if not node_name:
            node_name = child_dict["name"]
        node_dict = dict(
            {"node": hnode, "rank": rank, "name": node_name}, **child_dict["metrics"]
        )

        node_dicts.append(node_dict)
        frame_to_node_dict[frame] = hnode

        if not found:
            hparent.add_child(hnode)

        child_set = dict()
        if "children" in child_dict:
            for child in child_dict["children"]:
                # Validate the children names. We can potentially have duplicates when
                # we resolve OpenMP regions/loops that are outlined by compilers with
                # multiple loops. Each loop has a different address, but they resolve
                # to the same outlined function name, and with the same line number.
                # This can also happen if we resolve an address in a program or
                # library with no debug information, and there is no line number.
                # As a best effort, make the child name unique. It might not match
                # trees from multiple processes, though - the ordering might be different.
                c_name = child["frame"]["name"]
                if c_name in child_set.keys():
                    if not self.warning:
                        print(
                            rank,
                            ": WARNING! Duplicate child: ",
                            c_name,
                            "of parent",
                            child_dict["frame"]["name"],
                        )
                        self.warning = True
                    count = child_set[c_name] + 1
                    c_name = c_name + str(count)
                    child["frame"]["name"] = c_name
                    child_set[c_name] = count
                else:
                    child_set[c_name] = 1
                self.parse_node_apex(frame_to_node_dict, node_dicts, child, hnode, rank)

    def read(self):
        list_roots = []
        node_dicts = []
        frame_to_node_dict = {}
        frame = None
        top_node = None

        # start with creating a node_dict for each root
        for i in range(len(self.graph_dict)):
            rank = self.ranks[i]
            if "rank" in self.graph_dict[i]["frame"]:
                rank = self.graph_dict[i]["frame"]["rank"]
                del self.graph_dict[i]["frame"]["rank"]
            if "type" not in self.graph_dict[i]["frame"]:
                self.graph_dict[i]["frame"]["type"] = "function"
            frame = Frame(self.graph_dict[i]["frame"])
            graph_root = Node(frame, None, hnid=-1)
            if graph_root in list_roots:
                graph_root = top_node

            # depending on the node type, the name may not be in the frame
            node_name = self.graph_dict[i]["frame"].get("name")
            if not node_name:
                node_name = self.graph_dict[i]["name"]

            node_dict = dict(
                {"node": graph_root, "rank": rank, "name": node_name},
                **self.graph_dict[i]["metrics"],
            )
            node_dicts.append(node_dict)

            # Add the graph root to the list of roots, and save it
            # for the next tree to merge
            if graph_root not in list_roots:
                list_roots.append(graph_root)
                top_node = graph_root

            frame_to_node_dict[frame] = graph_root

            # call recursively on all children of root
            child_set = dict()
            if "children" in self.graph_dict[i]:
                for child in self.graph_dict[i]["children"]:
                    # Validate the children names. We can potentially have duplicates when
                    # we resolve OpenMP regions/loops that are outlined by compilers with
                    # multiple loops. Each loop has a different address, but they resolve
                    # to the same outlined function name, and with the same line number.
                    # This can also happen if we resolve an address in a program or
                    # library with no debug information, and there is no line number.
                    # As a best effort, make the child name unique. It might not match
                    # trees from multiple processes, though - the ordering might be different.
                    c_name = child["frame"]["name"]
                    if c_name in child_set.keys():
                        if not self.warning:
                            print(
                                rank,
                                ": WARNING! Duplicate child: ",
                                c_name,
                                "of parent",
                                node_name,
                            )
                            self.warning = True
                        count = child_set[c_name] + 1
                        c_name = c_name + str(count)
                        child["frame"]["name"] = c_name
                        child_set[c_name] = count
                    else:
                        child_set[c_name] = 1
                    self.parse_node_apex(
                        frame_to_node_dict, node_dicts, child, graph_root, rank
                    )

        graph = Graph(list_roots)
        graph.enumerate_traverse()

        exc_metrics = []
        inc_metrics = []
        for key in self.graph_dict[i]["metrics"].keys():
            if "(inc)" in key:
                inc_metrics.append(key)
            else:
                exc_metrics.append(key)

        dataframe = pd.DataFrame(data=node_dicts)

        # Make an index of all unique combinations of node, rank
        new_index = pd.MultiIndex.from_product(
            [dataframe["node"].unique(), dataframe["rank"].unique()],
            names=["node", "rank"],
        )
        dataframe.set_index(["node", "rank"], inplace=True)
        dataframe.sort_index(inplace=True)
        # For all missing values of [node, rank], add zero values for all columns
        dataframe = dataframe.reindex(new_index, fill_value=0)
        # Make our multi-index
        # Select rows where name is 0, we need to fix that.
        zeros = dataframe.loc[dataframe["name"] == 0]
        # For all columns with name '0', replace it with name from node.
        for row in zeros.itertuples(name="graphnodes"):
            name = row[0][0].frame["name"]
            index = row[0]
            dataframe.at[index, "name"] = name

        default_metric = "time (inc)"

        return hatchet.graphframe.GraphFrame(
            graph, dataframe, exc_metrics, inc_metrics, default_metric
        )
