# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import matplotlib.cm
import matplotlib.colors


def trees_to_dot(roots, dataframe, metric, name, rank, thread, threshold):
    """Calls to_dot in turn for each tree in the graph/forest."""
    text = (
        "strict digraph {\n"
        "graph [bgcolor=transparent];\n"
        "node [penwidth=4, shape=circle];\n"
        "edge [penwidth=2];\n\n"
    )

    all_nodes = ""
    all_edges = ""

    # call to_dot for each root in the graph
    visited = []
    for root in roots:
        (nodes, edges) = to_dot(
            root, dataframe, metric, name, rank, thread, threshold, visited
        )
        all_nodes += nodes
        all_edges += edges

    text += all_nodes + "\n" + all_edges + "\n}\n"

    return text


def to_dot(hnode, dataframe, metric, name, rank, thread, threshold, visited):
    """Write to graphviz dot format."""
    colormap = matplotlib.cm.Reds
    min_time = dataframe[metric].min()
    max_time = dataframe[metric].max()

    def add_nodes_and_edges(hnode):
        # set dataframe index based on if rank is a part of the index
        if "rank" in dataframe.index.names and "thread" in dataframe.index.names:
            df_index = (hnode, rank, thread)
        elif "rank" in dataframe.index.names:
            df_index = (hnode, rank)
        elif "thread" in dataframe.index.names:
            df_index = (hnode, thread)
        else:
            df_index = hnode

        node_time = dataframe.loc[df_index, metric]
        node_name = dataframe.loc[df_index, name]
        node_id = hnode._hatchet_nid

        weight = (node_time - min_time) / (max_time - min_time)
        color = matplotlib.colors.rgb2hex(colormap(weight))

        # only display nodes whose metric is greater than some threshold
        if (node_time >= threshold * max_time) and (hnode not in visited):
            node_string = '"{0}" [color="{1}", label="{2}" shape=oval];\n'.format(
                node_id, color, node_name
            )
            edge_string = ""

            # only display those edges where child's metric is greater than
            # threshold
            children = []
            for child in hnode.children:
                if (
                    "rank" in dataframe.index.names
                    and "thread" in dataframe.index.names
                ):
                    df_index = (child, rank, thread)
                elif "rank" in dataframe.index.names:
                    df_index = (child, rank)
                elif "thread" in dataframe.index.names:
                    df_index = (child, thread)
                else:
                    df_index = child

                child_time = dataframe.loc[df_index, metric]
                if child_time >= threshold * max_time:
                    children.append(child)

            visited.append(hnode)
            for child in children:
                # add edges
                child_id = child._hatchet_nid

                edge_string += '"{0}" -> "{1}";\n'.format(node_id, child_id)
                (nodes, edges) = add_nodes_and_edges(child)
                node_string += nodes
                edge_string += edges
        else:
            node_string = ""
            edge_string = ""

        return (node_string, edge_string)

    # call add_nodes_and_edges on the root
    return add_nodes_and_edges(hnode)
