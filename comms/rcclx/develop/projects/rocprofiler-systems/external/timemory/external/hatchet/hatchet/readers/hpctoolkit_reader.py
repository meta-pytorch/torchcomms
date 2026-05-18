# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import glob
import struct
import re
import os
import traceback

import numpy as np
import pandas as pd
import multiprocessing as mp
import multiprocessing.sharedctypes

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# cython imports
try:
    import hatchet.cython_modules.libs.reader_modules as _crm
except ImportError:
    print("-" * 80)
    print(
        """Error: Shared object (.so) not found for cython module.\n\tPlease run install.sh from the hatchet root directory to build modules."""
    )
    print("-" * 80)
    traceback.print_exc()
    raise


import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.util.timer import Timer
from hatchet.frame import Frame


src_file = 0


def init_shared_array(buf_):
    """Initialize shared array."""
    global shared_metrics
    shared_metrics = buf_


def read_metricdb_file(args):
    """Read a single metricdb file into a 1D array."""
    (
        filename,
        num_nodes,
        num_threads_per_rank,
        num_cpu_threads_per_rank,
        num_metrics,
        shape,
    ) = args
    rank = int(
        re.search(r"\-(\d+)\-(\d+)\-([\w\d]+)\-(\d+)\-\d.metric-db$", filename).group(1)
    )
    thread = int(
        re.search(r"\-(\d+)\-(\d+)\-([\w\d]+)\-(\d+)\-\d.metric-db$", filename).group(2)
    )

    with open(filename, "rb") as metricdb:
        metricdb.seek(32)
        arr1d = np.fromfile(
            metricdb, dtype=np.dtype(">f8"), count=num_nodes * num_metrics
        )

    arr = np.frombuffer(shared_metrics).reshape(shape)

    # copy the data in the right place in the larger 2D array of metrics
    if thread < 500:
        rank_offset = (rank * num_threads_per_rank + thread) * num_nodes
    else:
        # GPU streams in hpctoolkit 2021.05.15 start at thread id 500
        rank_offset = (
            rank * num_threads_per_rank + num_cpu_threads_per_rank + (thread - 500)
        ) * num_nodes

    arr[rank_offset : rank_offset + num_nodes, :num_metrics].flat = arr1d.flat
    arr[rank_offset : rank_offset + num_nodes, num_metrics] = range(1, num_nodes + 1)
    arr[rank_offset : rank_offset + num_nodes, num_metrics + 1] = rank
    arr[rank_offset : rank_offset + num_nodes, num_metrics + 2] = thread


class HPCToolkitReader:
    """Read in the various sections of an HPCToolkit experiment.xml file and
    metric-db files.
    """

    def __init__(self, dir_name):
        # this is the name of the HPCToolkit database directory. The directory
        # contains an experiment.xml and some metric-db files
        self.dir_name = dir_name

        root = ET.parse(self.dir_name + "/experiment.xml").getroot()
        self.loadmodule_table = next(root.iter("LoadModuleTable"))
        self.file_table = next(root.iter("FileTable"))
        self.procedure_table = next(root.iter("ProcedureTable"))
        self.metricdb_table = next(root.iter("MetricDBTable"))
        self.callpath_profile = next(root.iter("SecCallPathProfileData"))

        # For a parallel run, there should be one metric-db file per MPI
        # process
        metricdb_files = glob.glob(self.dir_name + "/*.metric-db")
        self.num_metricdb_files = len(metricdb_files)

        # We need to know how many threads per rank there are. This counts the
        # number of thread 0 metric-db files (i.e., number of ranks), then
        # uses this as the divisor to the total number of metric-db files.
        metricdb_numranks_files = glob.glob(self.dir_name + "/*-000-*.metric-db")
        self.num_ranks = len(metricdb_numranks_files)
        self.num_threads_per_rank = int(
            self.num_metricdb_files / len(metricdb_numranks_files)
        )

        self.num_cpu_threads_per_rank = self.count_cpu_threads_per_rank()

        # Read one metric-db file to extract the number of nodes in the CCT
        # and the number of metrics
        with open(metricdb_files[0], "rb") as metricdb:
            metricdb.read(18)  # skip tag
            metricdb.read(5)  # skip version TODO: should we?
            endian = metricdb.read(1)

            if endian == b"b":
                self.num_nodes = struct.unpack(">i", metricdb.read(4))[0]
                self.num_metrics = struct.unpack(">i", metricdb.read(4))[0]
            else:
                raise ValueError(
                    "HPCToolkitReader doesn't support endian '%s'" % endian
                )

        self.load_modules = {}
        self.src_files = {}
        self.procedure_names = {}
        self.metric_names = {}

        # this list of dicts will hold all the node information such as
        # procedure name, load module, filename, etc. for all the nodes
        self.node_dicts = []

        self.timer = Timer()

    def fill_tables(self):
        """Read certain sections of the experiment.xml file to create dicts of load
        modules, src_files, procedure_names, and metric_names.
        """
        for loadm in (self.loadmodule_table).iter("LoadModule"):
            self.load_modules[loadm.get("i")] = loadm.get("n")

        for filename in (self.file_table).iter("File"):
            self.src_files[filename.get("i")] = filename.get("n")

        for procedure in (self.procedure_table).iter("Procedure"):
            self.procedure_names[procedure.get("i")] = procedure.get("n")

        # store the keys as ints because we sort on keys later
        for metric in (self.metricdb_table).iter("MetricDB"):
            self.metric_names[int(metric.get("i"))] = metric.get("n")

        return (
            self.load_modules,
            self.src_files,
            self.procedure_names,
            self.metric_names,
        )

    def read_all_metricdb_files(self):
        """Read all the metric-db files and create a dataframe with num_nodes X
        num_metricdb_files rows and num_metrics columns. Three additional columns
        store the node id, MPI process rank, and thread id (if applicable).
        """
        metricdb_files = glob.glob(self.dir_name + "/*.metric-db")
        metricdb_files.sort()

        # All the metric data per node and per process is read into the metrics
        # array below. The three additional columns are for storing the implicit
        # node id (nid), MPI process rank, and thread id (if applicable).
        shape = [self.num_nodes * self.num_metricdb_files, self.num_metrics + 3]
        size = int(np.prod(shape))

        # shared memory buffer for multiprocessing
        shared_buffer = mp.sharedctypes.RawArray("d", size)

        pool = mp.Pool(initializer=init_shared_array, initargs=(shared_buffer,))
        self.metrics = np.frombuffer(shared_buffer).reshape(shape)
        args = [
            (
                filename,
                self.num_nodes,
                self.num_threads_per_rank,
                self.num_cpu_threads_per_rank,
                self.num_metrics,
                shape,
            )
            for filename in metricdb_files
        ]
        try:
            pool.map(read_metricdb_file, args)
        finally:
            pool.close()

        # once all files have been read, create a dataframe of metrics
        metric_names = [
            self.metric_names[key] for key in sorted(self.metric_names.keys())
        ]
        for idx, name in enumerate(metric_names):
            if name == "CPUTIME (usec) (E)" or name == "CPUTIME (sec) (E)":
                metric_names[idx] = "time"
            if name == "CPUTIME (usec) (I)" or name == "CPUTIME (sec) (I)":
                metric_names[idx] = "time (inc)"

        self.metric_columns = metric_names
        df_columns = self.metric_columns + ["nid", "rank", "thread"]
        self.df_metrics = pd.DataFrame(self.metrics, columns=df_columns)
        self.df_metrics["nid"] = self.df_metrics["nid"].astype(int, copy=False)
        self.df_metrics["rank"] = self.df_metrics["rank"].astype(int, copy=False)
        self.df_metrics["thread"] = self.df_metrics["thread"].astype(int, copy=False)

        # if number of threads per rank is 1, we do not need to keep the thread ID column
        if self.num_threads_per_rank == 1:
            del self.df_metrics["thread"]

        # used to speedup parse_xml_node
        self.np_metrics = self.df_metrics[self.metric_columns].values

        # getting the number of execution threads for our stride in
        # subtract_exclusive_metric_vals/ num nodes is already calculated
        self.total_execution_threads = self.num_threads_per_rank * self.num_ranks

    def read(self):
        """Read the experiment.xml file to extract the calling context tree and create
        a dataframe out of it. Then merge the two dataframes to create the final
        dataframe.

        Return:
            (GraphFrame): new GraphFrame with HPCToolkit data.
        """
        with self.timer.phase("fill tables"):
            self.fill_tables()

        with self.timer.phase("read metric db"):
            self.read_all_metricdb_files()

        list_roots = []

        # parse the ElementTree to generate a calling context tree
        for root in self.callpath_profile.findall("PF"):
            global src_file

            nid = int(root.get("i"))
            src_file = root.get("f")

            # start with the root and create the callpath and node for the root
            # also a corresponding node_dict to be inserted into the dataframe
            graph_root = Node(
                Frame(
                    {"type": "function", "name": self.procedure_names[root.get("n")]}
                ),
                None,
            )
            node_dict = self.create_node_dict(
                nid,
                graph_root,
                self.procedure_names[root.get("n")],
                "PF",
                self.src_files[src_file],
                int(root.get("l")),
                self.load_modules[root.get("lm")],
            )

            self.node_dicts.append(node_dict)
            list_roots.append(graph_root)

            # start graph construction at the root
            with self.timer.phase("graph construction"):
                self.parse_xml_children(root, graph_root)

            # put updated metrics back in dataframe
            for i, column in enumerate(self.metric_columns):
                if "(inc)" not in column and "(I)" not in column:
                    self.df_metrics[column] = self.np_metrics.T[i]

        with self.timer.phase("graph construction"):
            graph = Graph(list_roots)
            graph.enumerate_traverse()

        # create a dataframe for all the nodes in the graph
        self.df_nodes = pd.DataFrame.from_dict(data=self.node_dicts)

        # merge the metrics and node dataframes
        with self.timer.phase("data frame"):
            dataframe = pd.merge(self.df_metrics, self.df_nodes, on="nid")

            # set the index to be a MultiIndex
            if self.num_threads_per_rank > 1:
                indices = ["node", "rank", "thread"]
            # if number of threads per rank is 1, do not make thread an index
            elif self.num_threads_per_rank == 1:
                indices = ["node", "rank"]
            dataframe.set_index(indices, inplace=True)
            dataframe.sort_index(inplace=True)

        # create list of exclusive and inclusive metric columns
        exc_metrics = []
        inc_metrics = []
        for column in self.metric_columns:
            if "(inc)" in column or "(I)" in column:
                inc_metrics.append(column)
            else:
                exc_metrics.append(column)

        return hatchet.graphframe.GraphFrame(graph, dataframe, exc_metrics, inc_metrics)

    def parse_xml_children(self, xml_node, hnode):
        """Parses all children of an XML node."""
        for xml_child in xml_node:
            if xml_child.tag != "M":
                nid = int(xml_node.get("i"))
                line = int(xml_node.get("l"))
                self.parse_xml_node(xml_child, nid, line, hnode)

    def parse_xml_node(self, xml_node, parent_nid, parent_line, hparent):
        """Parses an XML node and its children recursively."""
        nid = int(xml_node.get("i"))

        global src_file
        xml_tag = xml_node.tag

        if xml_tag == "PF" or xml_tag == "Pr":
            # procedure
            name = self.procedure_names[xml_node.get("n")]
            if parent_line != 0:
                name = str(parent_line) + ":" + name
            src_file = xml_node.get("f")
            line = int(xml_node.get("l"))

            hnode = Node(Frame({"type": "function", "name": name}), hparent)
            node_dict = self.create_node_dict(
                nid,
                hnode,
                name,
                xml_tag,
                self.src_files[src_file],
                line,
                self.load_modules[xml_node.get("lm")],
            )

        elif xml_tag == "L":
            # loop
            src_file = xml_node.get("f")
            line = int(xml_node.get("l"))
            name = (
                "Loop@" + os.path.basename(self.src_files[src_file]) + ":" + str(line)
            )

            hnode = Node(
                Frame({"type": "loop", "file": self.src_files[src_file], "line": line}),
                hparent,
            )
            node_dict = self.create_node_dict(
                nid, hnode, name, xml_tag, self.src_files[src_file], line, None
            )

        elif xml_tag == "S":
            # statement
            line = int(xml_node.get("l"))
            # this might not be required for resolving conflicts
            name = os.path.basename(self.src_files[src_file]) + ":" + str(line)

            hnode = Node(
                Frame(
                    {
                        "type": "statement",
                        "file": self.src_files[src_file],
                        "line": line,
                    }
                ),
                hparent,
            )
            node_dict = self.create_node_dict(
                nid, hnode, name, xml_tag, self.src_files[src_file], line, None
            )

            # when we reach statement nodes, we subtract their exclusive
            # metric values from the parent's values
            for i, column in enumerate(self.metric_columns):
                if "(inc)" not in column and "(I)" not in column:
                    _crm.subtract_exclusive_metric_vals(
                        nid,
                        parent_nid,
                        self.np_metrics.T[i],
                        self.total_execution_threads,
                        self.num_nodes,
                    )

        if xml_tag == "C" or (
            xml_tag == "Pr" and self.procedure_names[xml_node.get("n")] == ""
        ):
            # do not add a node to the graph if the xml_tag is a callsite
            # or if its a procedure with no name
            # for Prs, the preceding Pr has the calling line number and for
            # PFs, the preceding C has the line number
            line = int(xml_node.get("l"))
            self.parse_xml_children(xml_node, hparent)
        else:
            self.node_dicts.append(node_dict)
            hparent.add_child(hnode)
            self.parse_xml_children(xml_node, hnode)

    def create_node_dict(self, nid, hnode, name, node_type, src_file, line, module):
        """Create a dict with all the node attributes."""
        node_dict = {
            "nid": nid,
            "name": name,
            "type": node_type,
            "file": src_file,
            "line": line,
            "module": module,
            "node": hnode,
        }

        return node_dict

    def count_cpu_threads_per_rank(self):
        metricdb_files = glob.glob(self.dir_name + "/*.metric-db")
        cpu_thread_ids = set()

        for filename in metricdb_files:
            thread = int(
                re.search(
                    r"\-(\d+)\-(\d+)\-([\w\d]+)\-(\d+)\-\d.metric-db$", filename
                ).group(2)
            )
            if thread < 500:
                cpu_thread_ids.add(thread)

        return len(cpu_thread_ids)
