# Copyright 2021-2023 University of Maryland and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import re
import os
import glob
import pandas as pd
import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.frame import Frame


class TAUReader:
    """Read in a profile generated using TAU."""

    def __init__(self, dirname):
        self.dirname = dirname
        self.node_dicts = []
        self.callpath_to_node = {}
        self.rank_thread_to_data = {}
        self.filepath_to_data = {}
        self.inc_metrics = []
        self.exc_metrics = []
        self.columns = []
        self.multiple_ranks = False
        self.multiple_threads = False

    def create_node_dict(
        self,
        node,
        columns,
        metric_values,
        name,
        filename,
        module,
        start_line,
        end_line,
        rank,
        thread,
    ):
        node_dict = {
            "node": node,
            "rank": rank,
            "thread": thread,
            "name": name,
            "file": filename,
            "module": module,
            "line": int(start_line),
            "end_line": int(end_line),
        }
        for i in range(len(metric_values)):
            node_dict[columns[i + 1]] = metric_values[i]
        return node_dict

    def create_graph(self):
        def _get_name_file_module(is_parent, node_info, symbol):
            """This function gets the name, file and module information
            for a node using the corresponding line in the output file.
            Example line: [UNWIND] <file> [@] <name> [{<file_or_module>} {<line>}]
            There are several line formats in TAU and this function gets
            the node information considering all these formats for which
            examples are given below.
            """
            name, file, module = None, None, None
            # There are several different formats in TAU outputs.
            # There might be file, line, and module information.
            # The following if-else block covers all possible output
            # formats. Example formats are given in comments.
            if symbol == " [@]":
                # Check if there is a [@] symbol.
                # Example: [UNWIND] <file> [@] <name> [{<file_or_module>} {<line>}]
                # Example: [UNWIND] <file> [@] <name> <module>
                node_info = node_info.split(symbol)
                # We don't need file and module information if it's a parent node.
                if not is_parent:
                    file_or_module = node_info[0].split()[1].strip("][")
                    # We put ".so" and ".S" in the module column.
                    if ".so" in node_info[0] or ".S" in node_info[0]:
                        module = file_or_module
                    else:
                        file = file_or_module

                    name = node_info[0].split()[0] + " " + file_or_module
                else:
                    # We just need to take name if it is a parent
                    name = node_info[0]
            elif symbol == " C ":
                # Check if there is a C symbol.
                # "C" symbol means it's a C function.
                node_info = node_info.split(symbol)
                name = node_info[0]
                # We don't need file and module information if it's a parent node.
                if not is_parent:
                    if "[{" in node_info[1]:
                        # Example: <name> C [{<file>} {<line>}]
                        node_info = node_info[1].split()
                        file = node_info[0].strip("}[{")
            else:
                if "[{" in node_info:
                    # If there isn't "C" or "[@]"
                    # Example: [<type>] <name> [{} {}]
                    node_info = node_info.split(" [{")
                    name = node_info[0]
                    # We don't need file and module information if it's a parent node.
                    if not is_parent:
                        if ".so" in node_info[1]:
                            module = node_info[1].split()[0].strip("}{")
                        else:
                            file = node_info[1].split()[0].strip("}{")
                else:
                    # Example 1: [<type>] <name> <module>
                    # Example 2: [<type>] <name>
                    # Example 3: <name>
                    name = node_info
                    node_info = node_info.split()
                    # We need to take module information from the first example.
                    # Another example is "[CONTEXT] .TAU application" which contradicts
                    # with the first example. So we check if there is "\" symbol which
                    # will show the module information in this case.
                    if len(node_info) == 3 and "/" in name:
                        name = node_info[0] + " " + node_info[1]
                        # We don't need file and module information if it's a parent node.
                        if not is_parent:
                            module = node_info[2]
            return [name, file, module]

        def _get_line_numbers(node_info):
            start_line, end_line = 0, 0
            # There should be "[{}]" symbols if there is line number information.
            if "[{" in node_info:
                tmp_module_or_file_line = (
                    re.search(r"\{.*\}\]", node_info).group(0).split(" {")
                )

                line_numbers = tmp_module_or_file_line[1].strip("}]").replace("{", "")
                start_line = line_numbers
                if "-" in line_numbers:
                    # Sometimes there is "-" between start line and end line
                    # Example: {341,1}-{396,1}
                    line_numbers = line_numbers.split("-")
                    start_line = line_numbers[0].split(",")[0]
                    end_line = line_numbers[1].split(",")[0]
                else:
                    if "," in line_numbers:
                        # Sometimes we don't have "-".
                        # Example: {15,0}
                        start_line = line_numbers.split(",")[0]
                        end_line = line_numbers.split(",")[1]
            else:
                # Some [UNWIND] nodes have the following formats.
                # Example 1: [UNWIND] <file>.<line_number> [@]
                # Example 2: [UNWIND] [<file>.<line_number>] [@]
                try:
                    start_line = int(node_info.split(".")[-1].split()[0].strip("]"))
                except ValueError:
                    pass
            return [start_line, end_line]

        def _create_parent(child_node, parent_callpath):
            """In TAU output, sometimes we see a node as a parent
            in the callpath before we see it as a leaf node. In
            this case, we need to create a hatchet node for the parent.

            We can't create a node_dict for the parent because we don't
            know its metric values when we first see it in a callpath.

            Example: a => b => c "<c_metric_values>"
            Here, if we haven't seen 'b' before, we should create it when we
            create 'c'.

            This function recursively creates parent nodes in a callpath
            until it reaches the already existing parent in that callpath.
            """
            parent_node = self.callpath_to_node.get(parent_callpath)

            # Return if arrives to the parent
            # Else create a parent and add parent/child
            if parent_node is not None:
                parent_node.add_child(child_node)
                child_node.add_parent(parent_node)
                return
            else:
                grand_parent_callpath = parent_callpath[:-1]
                parent_info = parent_callpath[-1]
                parent_name = ""

                if " C " in parent_info:
                    parent_name = _get_name_file_module(True, parent_info, " C ")[0]
                elif " [@] " in parent_info:
                    parent_name = _get_name_file_module(True, parent_info, " [@]")[0]
                else:
                    parent_name = _get_name_file_module(True, parent_info, "")[0]

                parent_node = Node(
                    Frame({"type": "function", "name": parent_name}), None
                )
                self.callpath_to_node[parent_callpath] = parent_node

                parent_node.add_child(child_node)
                child_node.add_parent(parent_node)
                _create_parent(parent_node, grand_parent_callpath)

        def _construct_column_list(first_rank_filenames):
            """This function constructs columns, exc_metrics, and
            inc_metrics using all metric files of a rank. It gets the
            all metric files of a rank as a tuple and only loads the
            second line (metadata) of these files.
            """
            columns = []
            for file_index in range(len(first_rank_filenames)):
                with open(first_rank_filenames[file_index], "r") as f:
                    # Get the metric name from the first line (..._TIME)
                    # if it doesn't exist in the second line:
                    # "192 templated_functions_MULTI_TIME"
                    first_line = next(f)
                    # No need to check if the metadata is the same for all metric files.
                    metadata = next(f)

                    # Get first three columns from # Name Calls Subrs Excl Incl ProfileCalls #
                    # ProfileCalls is removed since it is is typically set to 0 and not used.
                    # We only do this once since these column names are the same for all files.
                    if file_index == 0:
                        columns.extend(
                            re.match(r"\#\s(.*)\s\#", metadata).group(1).split(" ")[:-3]
                        )

                    name_metadata = re.search(r"<value>(.*?)<\/value>", metadata)
                    if name_metadata is None:
                        # Get metric from the first line if it doesn't exist in the
                        # second line (metadata).
                        metric_name = first_line.split("_")[-1][:-1]
                    else:
                        # Get metric from the second line (metadata).
                        metric_name = name_metadata.group(1)

                    # TODO: Decide if Calls and Subrs should be inc or exc metrics
                    if metric_name == "CPU_TIME" or metric_name == "TIME":
                        metric_name = "time"
                    elif metric_name == "Name":
                        metric_name == "name"
                    columns.extend([metric_name, metric_name + " (inc)"])
                    self.exc_metrics.append(metric_name)
                    self.inc_metrics.append(metric_name + " (inc)")
            return columns

        # dirpath -> returns path of a directory, string
        # dirnames -> returns directory names, list
        # files -> returns filenames in a directory, list
        profile_filenames = []
        for dirpath, dirnames, files in os.walk(self.dirname):
            profiles_in_dir = glob.glob(dirpath + "/profile.*")
            if profiles_in_dir:
                # sort input files in each directory in the same order
                profile_filenames.append(sorted(profiles_in_dir))

        # Store all files in a list of tuples.
        # Each tuple stores all the metric files of a rank.
        # We process one rank at a time.
        # Example: [(metric1/profile.x.0.0, metric2/profile.x.0.0), ...]
        profile_filenames = list(zip(*profile_filenames))

        # Get column information from the metric files of a rank.
        self.columns = _construct_column_list(profile_filenames[0])

        list_roots = []
        prev_rank, prev_thread = 0, 0
        # Example filenames_per_rank: (metric1/profile.x.0.0 ...)
        for filenames_per_rank in profile_filenames:
            file_info = filenames_per_rank[0].split(".")
            rank, thread = int(file_info[-3]), int(file_info[-1])
            if not self.multiple_ranks:
                self.multiple_ranks = True if rank != prev_rank else False
            if not self.multiple_threads:
                self.multiple_threads = True if thread != prev_thread else False

            # Load all files represent a different metric for a rank or a thread.
            # If there are 2 metrics, load metric1\profile.x.0.0 and metric2\profile.x.0.0
            file_data = []
            for f_index in range(len(filenames_per_rank)):
                # Store the lines after metadata.
                file_data.append(open(filenames_per_rank[f_index], "r").readlines()[2:])

            # Get the root information from only the first file to compare them
            # with others.
            # Example: ".TAU application" 1 1 272 15755429 0 GROUP="TAU_DEFAULT"
            root_line = re.match(r"\"(.*)\"\s(.*)\sG", file_data[0][0])
            root_name = root_line.group(1).strip(" ")
            # convert it to a tuple to use it as a key in callpath_to_node dictionary
            root_callpath = tuple([root_name])
            root_values = list(map(int, root_line.group(2).split(" ")[:-1]))

            # After first profile.0.0.0, only get Excl and Incl metric values
            # from other files since other columns will be the same.
            # We assume each metric file of a rank has the same root.
            first_file_root_name = re.search(r"\"(.*?)\"", file_data[0][0]).group(1)
            for f_index in range(1, len(file_data)):
                root_name = re.search(r"\"(.*?)\"", file_data[f_index][0]).group(1)
                # Below assert statement throws an error if the roots are not the
                # same for different metric files.
                # TODO: We need to find a solution if this throws an error.
                assert first_file_root_name == root_name, (
                    "Metric files for a rank has different roots.\n"
                    + "File: "
                    + filenames_per_rank[f_index]
                    + "\nLine: 2"
                )
                root_line = re.match(r"\"(.*)\"\s(.*)\sG", file_data[f_index][0])
                root_values.extend(list(map(int, root_line.group(2).split(" ")[2:4])))

            # Check if the root exists in other ranks.
            # Note that we assume the root is the same for all metric files of a rank.
            if root_callpath not in self.callpath_to_node:
                # Create the root node since it doesn't exist
                root_node = Node(Frame({"name": root_name, "type": "function"}), None)

                # Store callpaths to identify nodes
                self.callpath_to_node[root_callpath] = root_node
                list_roots.append(root_node)
            else:
                # Don't create a new node since it is created earlier
                root_node = self.callpath_to_node.get(root_callpath)

            node_dict = self.create_node_dict(
                root_node,
                self.columns,
                root_values,
                root_name,
                None,
                None,
                0,
                0,
                rank,
                thread,
            )
            self.node_dicts.append(node_dict)

            # Start from the line after root.
            # Iterate over only the first metric file of a rank
            # since the lines should be exactly the same across
            # all metric files of a rank.
            # Uses the same "line_index" for other metric files of a rank.
            for line_index in range(1, len(file_data[0])):
                line = file_data[0][line_index]
                metric_values = []
                # We only parse the lines that has "=>" symbol which shows the callpath info.
                # We just skip the other lines.
                if "=>" in line:
                    # Example: ".TAU application  => foo()  => bar()" 31 0 155019 155019 0 GROUP="TAU_SAMPLE|TAU_CALLPATH"
                    callpath_line_regex = re.match(r"\"(.*)\"\s(.*)\sG", line)
                    # callpath: ".TAU application  => foo()  => bar()"
                    callpath = [
                        name.strip(" ")
                        for name in callpath_line_regex.group(1).split("=>")
                    ]

                    # Example leaf_name: StrToInt [{lulesh-util.cc} {13,1}-{29,1}]
                    leaf_name = callpath[-1]
                    callpath = tuple(callpath)
                    parent_callpath = callpath[:-1]
                    # Don't include the value for ProfileCalls.
                    # metric_values: 31 0 155019 155019
                    metric_values = list(
                        map(float, callpath_line_regex.group(2).split(" ")[:-1])
                    )

                    # Get start and end line information
                    leaf_line_numbers = _get_line_numbers(leaf_name)
                    # Get name, file, and module information using the leaf name
                    # and the symbol on it
                    if " C " in leaf_name:
                        leaf_name_file_module = _get_name_file_module(
                            False, leaf_name, " C "
                        )
                    elif " [@]" in leaf_name:
                        leaf_name_file_module = _get_name_file_module(
                            False, leaf_name, " [@]"
                        )
                    else:
                        leaf_name_file_module = _get_name_file_module(
                            False, leaf_name, ""
                        )

                    # Example: ".TAU application  => foo()  => bar()" 31 0 155019..."
                    first_file_callpath_line = re.search(
                        r"\"(.*?)\"", file_data[0][line_index]
                    ).group(1)
                    # After first profile.x.0.0, only get Excl and Incl metric values
                    # from other files.
                    for f_index in range(1, len(file_data)):
                        other_file_callpath_line = re.search(
                            r"\"(.*?)\"", file_data[f_index][line_index]
                        ).group(1)
                        # We assume metric files of a rank should have the exact same lines.
                        # Only difference should be the Incl and Excl metric values.
                        # TODO: We should find a solution if this raises an error.
                        assert first_file_callpath_line == other_file_callpath_line, (
                            "Lines across metric files for a rank are not the same.\n"
                            + "File: "
                            + filenames_per_rank[f_index]
                            + "\nLine: "
                            + str(line_index + 3)
                        )
                        # Get the information from the same line in each file. "line_index".
                        callpath_line_regex = re.match(
                            r"\"(.*)\"\s(.*)\sG", file_data[f_index][line_index]
                        )
                        metric_values.extend(
                            map(float, callpath_line_regex.group(2).split(" ")[2:4])
                        )

                    leaf_node = self.callpath_to_node.get(callpath)
                    # Check if that node is created earlier
                    if leaf_node is None:
                        # Create the node since it doesn't exist
                        leaf_node = Node(
                            Frame(
                                {"type": "function", "name": leaf_name_file_module[0]}
                            ),
                            None,
                        )
                        self.callpath_to_node[callpath] = leaf_node

                        # Get its parent from its callpath.
                        parent_node = self.callpath_to_node.get(parent_callpath)
                        if parent_node is None:
                            # Create parent if it doesn't exist.
                            _create_parent(leaf_node, parent_callpath)
                        else:
                            parent_node.add_child(leaf_node)
                            leaf_node.add_parent(parent_node)

                    node_dict = self.create_node_dict(
                        leaf_node,
                        self.columns,
                        metric_values,
                        # name
                        leaf_name_file_module[0],
                        # file
                        leaf_name_file_module[1],
                        # module
                        leaf_name_file_module[2],
                        # start line
                        leaf_line_numbers[0],
                        # end line
                        leaf_line_numbers[1],
                        rank,
                        thread,
                    )

                    self.node_dicts.append(node_dict)

        return list_roots

    def read(self):
        """Read the TAU profile file to extract the calling context tree."""
        # Add all nodes and roots.
        roots = self.create_graph()
        # Create a graph object once all nodes have been added.
        graph = Graph(roots)
        graph.enumerate_traverse()

        dataframe = pd.DataFrame.from_dict(data=self.node_dicts)

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
        num_of_indices = len(indices)
        if num_of_indices > 1:
            if num_of_indices == 2:
                dataframe = dataframe.unstack()
            elif num_of_indices == 3:
                dataframe = dataframe.unstack().unstack()
            for idx, row in dataframe.iterrows():
                # There is always a valid name for an index.
                # Take that valid name and assign to other ranks/rows.
                name = row["name"][row["name"].first_valid_index()]
                dataframe.loc[idx, "name"] = name

                # Sometimes there is no file information.
                if row["file"].first_valid_index() is not None:
                    file = row["file"][row["file"].first_valid_index()]
                    dataframe.loc[idx, "file"] = file

                # Sometimes there is no module information.
                if row["module"].first_valid_index() is not None:
                    module = row["module"][row["module"].first_valid_index()]
                    dataframe.loc[idx, "module"] = module

                # Fill the rest with 0
                dataframe.fillna(0, inplace=True)

            # Stack the dataframe
            if num_of_indices == 2:
                dataframe = dataframe.stack()
            elif num_of_indices == 3:
                dataframe = dataframe.stack().stack()

        default_metric = "time (inc)"
        dataframe = dataframe.astype({"line": int, "end_line": int})
        return hatchet.graphframe.GraphFrame(
            graph, dataframe, self.exc_metrics, self.inc_metrics, default_metric
        )
