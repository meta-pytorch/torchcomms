# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pstats
import sys
import pandas as pd


import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.frame import Frame


def print_incomptable_msg(stats_file):
    """
    Function which makes the syntax cleaner in Profiler.write_to_file().
    """
    errmsg = """\n Error: Incompatible pstats file ({})\n Please run your code in Python {} to read in this file. \n"""
    if sys.version_info[0] == 2:
        print(errmsg.format(stats_file, 3))
    if sys.version_info[0] == 3:
        print(errmsg.format(stats_file, 2.7))


class StatData:
    """Faux Enum for python"""

    NUMCALLS = 0
    NATIVECALLS = 1
    EXCTIME = 2
    INCTIME = 3
    SRCNODE = 4


class NameData:
    """Faux Enum for python"""

    FILE = 0
    LINE = 1
    FNCNAME = 2


class CProfileReader:
    def __init__(self, filename):
        self.pstats_file = filename

        self.name_to_hnode = {}
        self.name_to_dict = {}

    def _create_node_and_row(self, fn_data, fn_name, stats_dict):
        """
        Description: Takes a profiled function as specified in a pstats file
        and creates a node for it and adds a new line of metadata to our
        dataframe if it does not exist.
        """
        u_fn_name = "{}:{}:{}".format(
            fn_name,
            fn_data[NameData.FILE].split("/")[-1],
            fn_data[NameData.LINE],
        )
        fn_hnode = self.name_to_hnode.get(u_fn_name)

        if not fn_hnode:
            # create a node if it doesn't exist yet
            fn_hnode = Node(Frame({"type": "function", "name": fn_name}), None)
            self.name_to_hnode[u_fn_name] = fn_hnode

            # lookup stat data for source here
            fn_stats = stats_dict[fn_data]
            self._add_node_metadata(u_fn_name, fn_data, fn_stats, fn_hnode)

        return fn_hnode

    def _get_src(self, stat):
        """Gets the source/parent of our current desitnation node"""
        return stat[StatData.SRCNODE]

    def _add_node_metadata(self, stat_name, stat_module_data, stats, hnode):
        """Puts all the metadata associated with a node in a dictionary to insert into pandas."""
        node_dict = {
            "file": stat_module_data[NameData.FILE],
            "line": stat_module_data[NameData.LINE],
            "name": stat_module_data[NameData.FNCNAME],
            "numcalls": stats[StatData.NUMCALLS],
            "nativecalls": stats[StatData.NATIVECALLS],
            "time (inc)": stats[StatData.INCTIME],
            "time": stats[StatData.EXCTIME],
            "node": hnode,
        }
        self.name_to_dict[stat_name] = node_dict

    def create_graph(self):
        """Performs the creation of our node graph"""
        try:
            stats_dict = pstats.Stats(self.pstats_file).__dict__["stats"]
        except ValueError:
            print_incomptable_msg(self.pstats_file)
            raise
        list_roots = []

        # We iterate through each function/node in our stats dict
        for dst_module_data, dst_stats in stats_dict.items():
            dst_name = dst_module_data[NameData.FNCNAME]
            dst_hnode = self._create_node_and_row(dst_module_data, dst_name, stats_dict)

            # get all parents of our current destination node
            # create source nodes and link with destination node
            srcs = self._get_src(dst_stats)
            if srcs == {}:
                list_roots.append(dst_hnode)
            else:
                for src_module_data in srcs.keys():
                    src_name = src_module_data[NameData.FNCNAME]

                    if src_name is not None:
                        src_hnode = self._create_node_and_row(
                            src_module_data, src_name, stats_dict
                        )
                        dst_hnode.add_parent(src_hnode)
                        src_hnode.add_child(dst_hnode)

        return list_roots

    def read(self):
        roots = self.create_graph()
        graph = Graph(roots)
        graph.enumerate_traverse()

        dataframe = pd.DataFrame.from_dict(data=list(self.name_to_dict.values()))
        index = ["node"]
        dataframe.set_index(index, inplace=True)
        dataframe.sort_index(inplace=True)

        return hatchet.graphframe.GraphFrame(graph, dataframe, ["time"], ["time (inc)"])
