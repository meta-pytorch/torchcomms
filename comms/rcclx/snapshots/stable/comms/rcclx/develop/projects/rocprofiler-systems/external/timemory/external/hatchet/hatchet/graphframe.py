# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import sys
import traceback
import os

from collections import defaultdict

import pandas as pd
import numpy as np
import multiprocess as mp

from .node import Node
from .graph import Graph
from .frame import Frame
from .query import AbstractQuery, QueryMatcher, CypherQuery
from .external.console import ConsoleRenderer
from .util.dot import trees_to_dot
from .util.logger import Logger
from .util.deprecated import deprecated_params
from .chopper import Chopper

try:
    from .cython_modules.libs import graphframe_modules as _gfm_cy
except ImportError:
    print("-" * 80)
    print(
        """Error: Shared object (.so) not found for cython module.\n\tPlease run install.sh from the hatchet root directory to build modules."""
    )
    print("-" * 80)
    traceback.print_exc()
    raise


def parallel_apply(filter_function, dataframe, queue):
    """A function called in parallel, which does a pandas apply on part of a
    dataframe and returns the results via multiprocessing queue function."""
    filtered_rows = dataframe.apply(filter_function, axis=1)
    filtered_df = dataframe[filtered_rows]
    queue.put(filtered_df)


class GraphFrame:
    """An input dataset is read into an object of this type, which includes a graph
    and a dataframe.
    """

    def __init__(
        self,
        graph,
        dataframe,
        exc_metrics,
        inc_metrics,
        default_metric="time",
        metadata=None,
        attributes=None,
    ):
        """Create a new GraphFrame from a graph and a dataframe.

        Likely, you do not want to use this function.

        See ``from_hpctoolkit``, ``from_caliper``, ``from_gprof_dot``, and
        other reader methods for easier ways to create a ``GraphFrame``.

        Arguments:
             graph (Graph): Graph of nodes in this GraphFrame.
             dataframe (DataFrame): Pandas DataFrame indexed by Nodes
                 from the graph, and potentially other indexes.
             exc_metrics: list of names of exclusive metrics in the dataframe.
             inc_metrics: list of names of inclusive metrics in the dataframe.
             default_metric (str): default column to use if one is needed but
                 not explicitly specified
             metadata (dict): information about the dataframe, such as the naming
                 convention for inclusive/exclusive metrics
             attributes (dict): a dictionary of names + values used by readers to
                 extend the Graphframe object. For example, if the underlying reader
                 needs to provide important and easily accessible information or
                 the reader needs to provide an additional function(s) for
                 manipulating/verifying the data.
        """
        if graph is None:
            raise ValueError("GraphFrame() requires a Graph")
        if dataframe is None:
            raise ValueError("GraphFrame() requires a DataFrame")
        if exc_metrics is None and inc_metrics is None:
            raise ValueError(
                "GraphFrame() requires atleast one exclusive or inclusive metric"
            )

        if "node" not in list(dataframe.index.names):
            raise ValueError(
                "DataFrames passed to GraphFrame() must have an index called 'node'."
            )

        self.graph = graph
        self.dataframe = dataframe
        self.exc_metrics = [] if exc_metrics is None else exc_metrics
        self.inc_metrics = [] if inc_metrics is None else inc_metrics
        self.default_metric = default_metric
        self.metadata = {} if metadata is None else metadata
        if "hatchet_inclusive_suffix" not in self.metadata:
            self.metadata["hatchet_inclusive_suffix"] = " (inc)"
        if "hatchet_exclusive_suffix" not in self.metadata:
            self.metadata["hatchet_exclusive_suffix"] = " (exc)"

        try:
            attributes = dict(attributes) if attributes is not None else {}
        except TypeError as e:
            print(
                "graphframe attributes argument must be convertable to dict: {}".format(
                    e
                )
            )
            raise

        # create attributes
        self.attributes = attributes.keys()
        for x, y in attributes.items():
            if hasattr(self, x):
                raise ValueError(
                    "cannot create graphframe attribute '{}' because it already exists".format(
                        x
                    )
                )
            setattr(self, x, y)

    @staticmethod
    @Logger.loggable
    def from_hpctoolkit(dirname):
        """Read an HPCToolkit database directory into a new GraphFrame.

        Arguments:
            dirname (str): parent directory of an HPCToolkit
                experiment.xml file

        Returns:
            (GraphFrame): new GraphFrame containing HPCToolkit profile data
        """
        # import this lazily to avoid circular dependencies
        from .readers.hpctoolkit_reader import HPCToolkitReader
        from .readers.hpctoolkit_v4_reader import HPCToolkitV4Reader

        if "experiment.xml" in os.listdir(dirname):
            return HPCToolkitReader(dirname).read()
        else:
            return HPCToolkitV4Reader(dirname).read()

    @staticmethod
    @Logger.loggable
    def from_caliper(filename_or_stream, query=None):
        """Read in a Caliper .cali or .json file.
        Args:
            filename_or_stream (str or file-like): name of a Caliper output
                file in `.cali` or JSON-split format, or an open file object
                to read one
            query (str): cali-query in CalQL format
        """
        # import this lazily to avoid circular dependencies
        from .readers.caliper_reader import CaliperReader

        return CaliperReader(filename_or_stream, query).read()

    @staticmethod
    @Logger.loggable
    def from_caliperreader(filename_or_caliperreader):
        """Read in a native Caliper `cali' file using Caliper's python reader.

        Args:
            filename_or_caliperreader (str or CaliperReader): name of a Caliper
                output file in `.cali` format, or a CaliperReader object
        """
        # import this lazily to avoid circular dependencies
        from .readers.caliper_native_reader import CaliperNativeReader

        return CaliperNativeReader(filename_or_caliperreader).read()

    @staticmethod
    @Logger.loggable
    def from_spotdb(db_key, list_of_ids=None):
        """Read multiple graph frames from a SpotDB instance

        Args:
            db_key (str or SpotDB object): locator for SpotDB instance
                This can be a SpotDB object directly, or a locator for a spot
                database, which is a string with either
                    * A directory for .cali files,
                    * A .sqlite file name
                    * A SQL database URL (e.g., "mysql://hostname/db")

            list_of_ids: The list of run IDs to read from the database.
                If this is None, returns all runs.

        Returns:
            A list of graphframes, one for each requested run that was found
        """

        from .readers.spotdb_reader import SpotDBReader

        return SpotDBReader(db_key, list_of_ids).read()

    @staticmethod
    @Logger.loggable
    def from_gprof_dot(filename):
        """Read in a DOT file generated by gprof2dot."""
        # import this lazily to avoid circular dependencies
        from .readers.gprof_dot_reader import GprofDotReader

        return GprofDotReader(filename).read()

    @staticmethod
    @Logger.loggable
    def from_cprofile(filename):
        """Read in a pstats/prof file generated using python's cProfile."""
        # import this lazily to avoid circular dependencies
        from .readers.cprofile_reader import CProfileReader

        return CProfileReader(filename).read()

    @staticmethod
    @Logger.loggable
    def from_pyinstrument(filename):
        """Read in a JSON file generated using Pyinstrument."""
        # import this lazily to avoid circular dependencies
        from .readers.pyinstrument_reader import PyinstrumentReader

        return PyinstrumentReader(filename).read()

    @staticmethod
    @Logger.loggable
    def from_tau(dirname):
        """Read in a profile generated using TAU."""
        # import this lazily to avoid circular dependencies
        from .readers.tau_reader import TAUReader

        return TAUReader(dirname).read()

    @staticmethod
    @Logger.loggable
    def from_scorep(filename):
        """Read in a profile generated using Score-P."""
        # import this lazily to avoid circular dependencies
        from .readers.scorep_reader import ScorePReader

        return ScorePReader(filename).read()

    @staticmethod
    @Logger.loggable
    def from_timemory(input=None, select=None, **_kwargs):
        """Read in timemory data.

        Links:
            https://github.com/NERSC/timemory
            https://timemory.readthedocs.io

        Arguments:
            input (str or file-stream or dict or None):
                Valid argument types are:

                1. Filename for a timemory JSON tree file
                2. Open file stream to one of these files
                3. Dictionary from timemory JSON tree


                Currently, timemory supports two JSON layouts: flat and tree.
                The former is a 1D-array representation of the hierarchy which
                represents the hierarchy via indentation schemes in the labels
                and is not compatible with hatchet. The latter is a hierarchical
                representation of the data and is the required JSON layout when
                using hatchet. Timemory JSON tree files typically have the
                extension ".tree.json".

                If input is None, this assumes that timemory has been recording
                data within the application that is using hatchet. In this
                situation, this method will attempt to import the data directly
                from timemory.

                At the time of this writing, the direct data import will:

                1. Stop any currently collecting components
                2. Aggregate child thread data of the calling thread
                3. Clear all data on the child threads
                4. Aggregate the data from any MPI and/or UPC++ ranks.


                Thus, if MPI or UPC++ is used, every rank must call this routine.
                The zeroth rank will have the aggregation and all the other
                non-zero ranks will only have the rank-specific data.

                Whether or not the per-thread and per-rank data itself is
                combined is controlled by the `collapse_threads` and
                `collapse_processes` attributes in the `timemory.settings`
                submodule.

                In the C++ API, it is possible for only #1 to be applied and data
                can be obtained for an individual thread and/or rank without
                aggregation. This is not currently available to Python, however,
                it can be made available upon request via a GitHub Issue.

            select (list of str):
                A list of strings which match the component enumeration names, e.g. ["cpu_clock"].

            per_thread (boolean):
                Ensures that when applying filters to the graphframe, frames with
                identical name/file/line/etc. info but from different threads are
                not combined

            per_rank (boolean):
                Ensures that when applying filters to the graphframe, frames with
                identical name/file/line/etc. info but from different ranks are
                not combined

        """
        from .readers.timemory_reader import TimemoryReader

        if input is not None:
            try:
                return TimemoryReader(input, select, **_kwargs).read()
            except IOError:
                pass
        else:
            try:
                import timemory

                TimemoryReader(timemory.get(hierarchy=True), select, **_kwargs).read()
            except ImportError:
                print(
                    "Error! timemory could not be imported. Provide filename, file stream, or dict."
                )
                raise

    @staticmethod
    @Logger.loggable
    def from_literal(graph_dict):
        """Create a GraphFrame from a list of dictionaries."""
        # import this lazily to avoid circular dependencies
        from .readers.literal_reader import LiteralReader

        return LiteralReader(graph_dict).read()

    @staticmethod
    @Logger.loggable
    def from_apex(dirname):
        """Create a GraphFrame from a list of dictionaries."""
        # import this lazily to avoid circular dependencies
        from .readers.apex_reader import ApexReader

        return ApexReader(dirname).read()

    @staticmethod
    @Logger.loggable
    def from_perfetto(file_or_files, **kwargs):
        """Create a GraphFrame from perfetto files"""
        # import this lazily to avoid circular dependencies
        from .readers.perfetto_reader import PerfettoReader
        import glob
        import os

        files = []
        filename_list = (
            file_or_files
            if isinstance(file_or_files, (list, tuple))
            else [file_or_files]
        )

        for filename in filename_list:
            if os.path.exists(filename) and os.path.isdir(filename):
                filename = os.path.join(filename, "*.proto")

            if not os.path.exists(filename):
                files += glob.glob(
                    filename,
                    recursive=kwargs["recursive"] if "recursive" in kwargs else False,
                )
            else:
                files += [filename]

        if len(files) == 0:
            raise ValueError(
                "No omnitrace perfetto files found in '{}'".format(filename_list)
            )

        return PerfettoReader(files, **kwargs).read(**kwargs)

    @staticmethod
    @Logger.loggable
    def from_omnitrace(file_or_files, **kwargs):
        """In the future, this should support both omnitrace perfetto traces and timemory JSON files"""
        return GraphFrame.from_perfetto(file_or_files, **kwargs)

    @staticmethod
    @Logger.loggable
    def from_lists(*lists):
        """Make a simple GraphFrame from lists.

        This creates a Graph from lists (see ``Graph.from_lists()``) and uses
        it as the index for a new GraphFrame. Every node in the new graph has
        exclusive time of 1 and inclusive time is computed automatically.

        """
        graph = Graph.from_lists(*lists)
        graph.enumerate_traverse()

        df = pd.DataFrame({"node": list(graph.traverse())})
        df["time"] = [1.0] * len(graph)
        df["name"] = [n.frame["name"] for n in graph.traverse()]
        df.set_index(["node"], inplace=True)
        df.sort_index(inplace=True)

        gf = GraphFrame(graph, df, ["time"], [])
        gf.calculate_inclusive_metrics()
        return gf

    @staticmethod
    @Logger.loggable
    def from_hdf(filename, **kwargs):
        # import this lazily to avoid circular dependencies
        from .readers.hdf5_reader import HDF5Reader

        return HDF5Reader(filename).read(**kwargs)

    @Logger.loggable
    def to_hdf(self, filename, key="hatchet_graphframe", **kwargs):
        # import this lazily to avoid circular dependencies
        from .writers.hdf5_writer import HDF5Writer

        HDF5Writer(filename).write(self, key=key, **kwargs)

    @Logger.loggable
    def update_metadata(self, num_processes=None, num_threads=None, metadata=None):
        """Update a GraphFrame object's metadata."""
        if num_processes is not None:
            if not isinstance(num_processes, int):
                raise TypeError(
                    "The number of processing elements must be of type integer."
                )
            self.metadata["num_processes"] = num_processes

        if num_threads is not None:
            if not isinstance(num_threads, int):
                raise TypeError("The number of threads must be of type integer.")
            self.metadata["num_threads"] = num_threads

        if metadata is not None:
            self.metadata.update(metadata)

        return self

    @Logger.loggable
    def copy(self):
        """Return a shallow copy of the graphframe.

        This copies the DataFrame, but the Graph is shared between self and
        the new GraphFrame.
        """
        return GraphFrame(
            self.graph,
            self.dataframe.copy(),
            list(self.exc_metrics),
            list(self.inc_metrics),
            self.default_metric,
            dict(self.metadata),
            attributes=dict([[x, getattr(self, x)] for x in self.attributes]),
        )

    @Logger.loggable
    def deepcopy(self):
        """Return a copy of the graphframe."""
        node_clone = {}
        graph_copy = self.graph.copy(node_clone)
        dataframe_copy = self.dataframe.copy()

        index_names = dataframe_copy.index.names
        dataframe_copy.reset_index(inplace=True)

        dataframe_copy["node"] = dataframe_copy["node"].apply(lambda x: node_clone[x])

        dataframe_copy.set_index(index_names, inplace=True)

        return GraphFrame(
            graph_copy,
            dataframe_copy,
            list(self.exc_metrics),
            list(self.inc_metrics),
            self.default_metric,
            dict(self.metadata),
            attributes=dict([[x, getattr(self, x)] for x in self.attributes]),
        )

    def drop_index_levels(self, function=np.mean):
        """Drop all index levels but `node`."""
        index_names = list(self.dataframe.index.names)
        index_names.remove("node")

        # create dict that stores aggregation function for each column
        agg_dict = {}
        for col in self.dataframe.columns.tolist():
            if col in self.exc_metrics + self.inc_metrics:
                agg_dict[col] = function
            else:
                agg_dict[col] = lambda x: x.iloc[0]

        # perform a groupby to merge nodes that just differ in index columns
        self.dataframe.reset_index(level="node", inplace=True)
        agg_df = self.dataframe.groupby("node").agg(agg_dict)

        self.dataframe = agg_df

    @Logger.loggable
    def filter(self, filter_obj, squash=True, num_procs=mp.cpu_count()):
        """Filter the dataframe using a user-supplied function.

        Note: Operates in parallel on user-supplied lambda functions.

        Arguments:
            filter_obj (callable, list, or QueryMatcher): the filter to apply to the GraphFrame.
            squash (boolean, optional): if True, automatically call squash for the user.
        """
        dataframe_copy = self.dataframe.copy()

        index_names = self.dataframe.index.names
        dataframe_copy.reset_index(inplace=True)

        filtered_df = None

        if callable(filter_obj):
            # applying pandas filter using the callable function
            if num_procs > 1:
                # perform filter in parallel (default)
                queue = mp.Queue()
                processes = []
                returned_frames = []
                subframes = np.array_split(dataframe_copy, num_procs)

                # Manually create a number of processes equal to the number of
                # logical cpus available
                for pid in range(num_procs):
                    process = mp.Process(
                        target=parallel_apply,
                        args=(filter_obj, subframes[pid], queue),
                    )
                    process.start()
                    processes.append(process)

                # Stores filtered subframes in a list: 'returned_frames', for
                # pandas concatenation. This intermediary list is used because
                # pandas concat is faster when called only once on a list of
                # dataframes, than when called multiple times appending onto a
                # frame of increasing size.
                for pid in range(num_procs):
                    returned_frames.append(queue.get())

                for proc in processes:
                    proc.join()

                filtered_df = pd.concat(returned_frames)

            else:
                # perform filter sequentiually if num_procs = 1
                filtered_rows = dataframe_copy.apply(filter_obj, axis=1)
                filtered_df = dataframe_copy[filtered_rows]

        elif isinstance(filter_obj, (list, str)) or issubclass(
            type(filter_obj), AbstractQuery
        ):
            # use a callpath query to apply the filter
            query = filter_obj
            if isinstance(filter_obj, list):
                query = QueryMatcher(filter_obj)
            elif isinstance(filter_obj, str):
                query = CypherQuery(filter_obj)
            query_matches = query.apply(self)
            # match_set = list(set().union(*query_matches))
            # filtered_df = dataframe_copy.loc[dataframe_copy["node"].isin(match_set)]
            filtered_df = dataframe_copy.loc[dataframe_copy["node"].isin(query_matches)]
        else:
            raise InvalidFilter(
                "The argument passed to filter must be a callable, a query path list, or a QueryMatcher object."
            )

        if filtered_df.shape[0] == 0:
            raise EmptyFilter(
                "The provided filter would have produced an empty GraphFrame."
            )

        filtered_df.set_index(index_names, inplace=True)

        filtered_gf = GraphFrame(
            self.graph,
            filtered_df,
            list(self.exc_metrics),
            list(self.inc_metrics),
            self.default_metric,
            dict(self.metadata),
            attributes=dict([[x, getattr(self, x)] for x in self.attributes]),
        )

        if squash:
            return filtered_gf.squash()
        return filtered_gf

    @Logger.loggable
    def squash(self):
        """Rewrite the Graph to include only nodes present in the DataFrame's rows.

        This can be used to simplify the Graph, or to normalize Graph
        indexes between two GraphFrames.
        """
        index_names = self.dataframe.index.names
        self.dataframe.reset_index(inplace=True)

        # create new nodes for each unique node in the old dataframe
        old_to_new = {n: n.copy() for n in set(self.dataframe["node"])}
        for i in old_to_new:
            old_to_new[i]._hatchet_nid = i._hatchet_nid

        # Maintain sets of connections to make for each old node.
        # Start with old -> new mapping and update as we traverse subgraphs.
        connections = defaultdict(lambda: set())
        connections.update({k: {v} for k, v in old_to_new.items()})

        new_roots = []  # list of new roots

        # connect new nodes to children according to transitive
        # relationships in the old graph.
        def rewire(node, new_parent, visited):
            # make all transitive connections for the node we're visiting
            for n in connections[node]:
                if new_parent:
                    # there is a parent in the new graph; connect it
                    if n not in new_parent.children:
                        new_parent.add_child(n)
                        n.add_parent(new_parent)

                elif n not in new_roots:
                    # this is a new root
                    new_roots.append(n)

            new_node = old_to_new.get(node)
            transitive = set()
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    transitive |= rewire(child, new_node or new_parent, visited)

            if new_node:
                # since new_node exists in the squashed graph, we only
                # need to connect new_node
                return {new_node}
            else:
                # connect parents to the first transitively reachable
                # new_nodes of nodes we're removing with this squash
                connections[node] |= transitive
                return connections[node]

        # run rewire for each root and make a new graph
        visited = set()
        for root in self.graph.roots:
            rewire(root, None, visited)
        graph = Graph(new_roots)
        graph.enumerate_traverse()

        # reindex new dataframe with new nodes
        df = self.dataframe.copy()
        df["node"] = df["node"].apply(lambda x: old_to_new[x])

        # at this point, the graph is potentially invalid, as some nodes
        # may have children with identical frames.
        merges = graph.normalize()
        df["node"] = df["node"].apply(lambda n: merges.get(n, n))

        self.dataframe.set_index(index_names, inplace=True)
        df.set_index(index_names, inplace=True)
        # create dict that stores aggregation function for each column
        agg_dict = {}
        for col in df.columns.tolist():
            if col in self.exc_metrics + self.inc_metrics:
                # use min_count=1 (default is 0) here, so sum of an all-NA
                # series is NaN, not 0
                # when min_count=1, sum([NaN, NaN)] = NaN
                # when min_count=0, sum([NaN, NaN)] = 0
                agg_dict[col] = lambda x: x.sum(min_count=1)
            else:
                agg_dict[col] = lambda x: x.iloc[0]

        # perform a groupby to merge nodes with the same callpath
        agg_df = df.groupby(index_names).agg(agg_dict)
        agg_df.sort_index(inplace=True)

        # put it all together
        new_gf = GraphFrame(
            graph,
            agg_df,
            list(self.exc_metrics),
            list(self.inc_metrics),
            self.default_metric,
            dict(self.metadata),
            attributes=dict([[x, getattr(self, x)] for x in self.attributes]),
        )
        new_gf.calculate_inclusive_metrics()
        return new_gf

    def _init_sum_columns(self, columns, out_columns):
        """Helper function for subtree_sum and subgraph_sum."""
        if out_columns is None:
            out_columns = columns
        else:
            # init out columns with input columns in case they are not there.
            for col, out in zip(columns, out_columns):
                self.dataframe[out] = self.dataframe[col]

        if len(columns) != len(out_columns):
            raise ValueError("columns out_columns must be the same length!")

        return out_columns

    def subtree_sum(
        self, columns, out_columns=None, function=lambda x: x.sum(min_count=1)
    ):
        """Compute sum of elements in subtrees.  Valid only for trees.

        For each row in the graph, ``out_columns`` will contain the
        element-wise sum of all values in ``columns`` for that row's node
        and all of its descendants.

        This algorithm will multiply count nodes with in-degree higher
        than one -- i.e., it is only correct for trees.  Prefer using
        ``subgraph_sum`` (which calls ``subtree_sum`` if it can), unless
        you have a good reason not to.

        Arguments:
            columns (list of str): names of columns to sum (default: all columns)
            out_columns (list of str): names of columns to store results
                (default: in place)
            function (callable): associative operator used to sum
                elements, sum of an all-NA series is NaN (default: sum(min_count=1))
        """
        out_columns = self._init_sum_columns(columns, out_columns)

        # sum over the output columns
        for node in self.graph.traverse(order="post"):
            if node.children:
                # TODO: need a better way of aggregating inclusive metrics when
                # TODO: there is a multi-index
                try:
                    is_multi_index = isinstance(
                        self.dataframe.index, pd.core.index.MultiIndex
                    )
                except AttributeError:
                    is_multi_index = isinstance(self.dataframe.index, pd.MultiIndex)

                if is_multi_index:
                    for rank_thread in self.dataframe.loc[
                        (node), out_columns
                    ].index.unique():
                        # rank_thread is either rank or a tuple of (rank, thread).
                        # We check if rank_thread is a tuple and if it is, we
                        # create a tuple of (node, rank, thread). If not, we create
                        # a tuple of (node, rank).
                        if isinstance(rank_thread, tuple):
                            df_index1 = (node,) + rank_thread
                            df_index2 = ([node] + node.children,) + rank_thread
                        else:
                            df_index1 = (node, rank_thread)
                            df_index2 = ([node] + node.children, rank_thread)

                        for col in out_columns:
                            self.dataframe.loc[df_index1, col] = function(
                                self.dataframe.loc[df_index2, col]
                            )
                else:
                    for col in out_columns:
                        self.dataframe.loc[node, col] = function(
                            self.dataframe.loc[[node] + node.children, col]
                        )

    def subgraph_sum(
        self, columns, out_columns=None, function=lambda x: x.sum(min_count=1)
    ):
        """Compute sum of elements in subgraphs.

        For each row in the graph, ``out_columns`` will contain the
        element-wise sum of all values in ``columns`` for that row's node
        and all of its descendants.

        This algorithm is worst-case quadratic in the size of the graph,
        so we try to call ``subtree_sum`` if we can.  In general, there
        is not a particularly efficient algorithm known for subgraph
        sums, so this does about as well as we know how.

        Arguments:
            columns (list of str):  names of columns to sum (default: all columns)
            out_columns (list of str): names of columns to store results
                (default: in place)
            function (callable): associative operator used to sum
                elements, sum of an all-NA series is NaN (default: sum(min_count=1))
        """
        if self.graph.is_tree():
            self.subtree_sum(columns, out_columns, function)
            return

        out_columns = self._init_sum_columns(columns, out_columns)
        for node in self.graph.traverse():
            subgraph_nodes = list(node.traverse())
            # TODO: need a better way of aggregating inclusive metrics when
            # TODO: there is a multi-index
            try:
                is_multi_index = isinstance(
                    self.dataframe.index, pd.core.index.MultiIndex
                )
            except AttributeError:
                is_multi_index = isinstance(self.dataframe.index, pd.MultiIndex)

            if is_multi_index:
                for rank_thread in self.dataframe.loc[
                    (node), out_columns
                ].index.unique():
                    # rank_thread is either rank or a tuple of (rank, thread).
                    # We check if rank_thread is a tuple and if it is, we
                    # create a tuple of (node, rank, thread). If not, we create
                    # a tuple of (node, rank).
                    if isinstance(rank_thread, tuple):
                        df_index1 = (node,) + rank_thread
                        df_index2 = (subgraph_nodes,) + rank_thread
                    else:
                        df_index1 = (node, rank_thread)
                        df_index2 = (subgraph_nodes, rank_thread)

                    for col in out_columns:
                        self.dataframe.loc[df_index1, col] = [
                            function(self.dataframe.loc[df_index2, col])
                        ]
            else:
                # TODO: if you take the list constructor away from the
                # TODO: assignment below, this assignment gives NaNs. Why?
                self.dataframe.loc[(node), out_columns] = list(
                    function(self.dataframe.loc[(subgraph_nodes), columns])
                )

    def calculate_inclusive_metrics(self):
        """Update inclusive columns (typically after operations that rewire the
        graph.
        """
        # we should update inc metric only if exc metric exist
        if not self.exc_metrics:
            return

        self.inc_metrics = [
            "%s%s" % (s, self.metadata["hatchet_inclusive_suffix"])
            for s in self.exc_metrics
        ]
        self.subgraph_sum(self.exc_metrics, self.inc_metrics)

    @Logger.loggable
    def calculate_exclusive_metrics(self, columns=None):
        """Calculates exclusive metrics using the corresponding inclusive metrics.

        Computed for all inclusive metrics if columns=None.

        Raise error if the given column is not in inc_metrics.
        If the given column is inclusive:
            If ' (inc)' is in the given column, name of the new column will be
        the given column without ' (inc)' at the end.
            If ' (inc)' is not in the given column, name of the new column will be
        the given column with ' (exc)' at the end.
        """

        # check if the columns parameter is None
        if columns is not None:
            # make it a list if it's not None and not a list
            if not isinstance(columns, list):
                columns = [columns]

            # check if the given columns are in inc_metrics.
            for column in columns:
                assert (
                    column in self.inc_metrics
                ), "{} does not exist in the graphframe.inc_metrics.".format(column)
        else:
            # if the user doesn't provide any columns, calculate
            # exclusive metric for all inclusive metrics.
            columns = [column for column in self.inc_metrics]

        # create the new columns and add
        # them to the exc_metrics.
        inc_exc_pairs = []
        new_data = {}
        for column in columns:
            # name the new column removing 'hatchet_inc_suffix' from
            # the inclusive column
            new_column = ""
            if self.metadata["hatchet_inclusive_suffix"] in column:
                new_column = column.replace(
                    self.metadata["hatchet_inclusive_suffix"], ""
                )
            # add 'hatchet_ex_suffix' if 'hatchet_inc_suffix' doesn't exists.
            else:
                new_column = column + self.metadata["hatchet_exclusive_suffix"]

            if new_column not in self.exc_metrics:
                self.exc_metrics.append(new_column)

            # keep the columns as a list of (inc_metric, exc_metric)
            inc_exc_pairs.append((column, new_column))
            # create a dict for the new data
            new_data[new_column] = {}

        # compute exclusive metrics by travering the graph
        for node in self.graph.traverse():
            for inc, exc in inc_exc_pairs:
                # if the dataframe index is a MultiIndex
                if isinstance(self.dataframe.index, pd.MultiIndex):
                    for idx in self.dataframe.loc[node].index:
                        node_index_tuple = None
                        # if 1D multiindex
                        if isinstance(idx, int):
                            idx = [idx]
                        node_index_tuple = tuple([node]) + tuple(idx)

                        # calculate exclusive metric.
                        child_inc_sum = 0
                        for child in node.children:
                            child_index_tuple = tuple([child]) + tuple(idx)
                            child_inc_sum += self.dataframe.loc[child_index_tuple][inc]
                        exc_value = (
                            self.dataframe.loc[node_index_tuple][inc] - child_inc_sum
                        )
                        # store the new value.
                        new_data[exc][node_index_tuple] = exc_value
                # if not a MultiIndex
                else:
                    # calculate exclusive metric.
                    child_inc_sum = 0
                    for child in node.children:
                        child_inc_sum += self.dataframe.loc[child][inc]
                    exc_value = self.dataframe.loc[node][inc] - child_inc_sum
                    # store the new value.
                    new_data[exc][node] = exc_value

                # create series for each exc column.
                new_data[exc] = pd.Series(data=new_data[exc])

        # add all new exc columns to the dataframe at once.
        self.dataframe = self.dataframe.assign(**new_data)

    @Logger.loggable
    def show_metric_columns(self):
        """Returns a list of dataframe column labels."""
        return list(self.exc_metrics + self.inc_metrics)

    def unify(self, other):
        """Returns a unified graphframe.

        Ensure self and other have the same graph and same node IDs. This may
        change the node IDs in the dataframe.

        Update the graphs in the graphframe if they differ.
        """
        if self.graph is other.graph:
            return

        node_map = {}
        union_graph = self.graph.union(other.graph, node_map)

        self_index_names = self.dataframe.index.names
        other_index_names = other.dataframe.index.names

        self.dataframe.reset_index(inplace=True)
        other.dataframe.reset_index(inplace=True)

        self.dataframe["node"] = self.dataframe["node"].apply(lambda x: node_map[id(x)])
        other.dataframe["node"] = other.dataframe["node"].apply(
            lambda x: node_map[id(x)]
        )

        # add missing rows to copy of self's dataframe in preparation for
        # operation
        self._insert_missing_rows(other)

        self.dataframe.set_index(self_index_names, inplace=True, drop=True)
        other.dataframe.set_index(other_index_names, inplace=True, drop=True)

        self.graph = union_graph
        other.graph = union_graph

    @deprecated_params(
        metric="metric_column",
        name="name_column",
        expand_names="expand_name",
        context="context_column",
        invert_colors="invert_colormap",
    )
    @Logger.loggable
    def tree(
        self,
        metric_column=None,
        precision=3,
        name_column="name",
        expand_name=False,
        context_column="file",
        rank=0,
        thread=0,
        depth=10000,
        highlight_name=False,
        colormap="RdYlGn",
        invert_colormap=False,
    ):
        """Format this graphframe as a tree and return the resulting string."""
        color = sys.stdout.isatty()
        shell = None
        if metric_column is None:
            metric_column = self.default_metric

        if color is False:
            try:
                import IPython

                shell = IPython.get_ipython().__class__.__name__
            except ImportError:
                pass
            # Test if running in a Jupyter notebook or qtconsole
            if shell == "ZMQInteractiveShell":
                color = True

        if sys.version_info.major == 2:
            unicode = False
        elif sys.version_info.major == 3:
            unicode = True

        return ConsoleRenderer(unicode=unicode, color=color).render(
            self.graph.roots,
            self.dataframe,
            metric_column=metric_column,
            precision=precision,
            name_column=name_column,
            expand_name=expand_name,
            context_column=context_column,
            rank=rank,
            thread=thread,
            depth=depth,
            highlight_name=highlight_name,
            colormap=colormap,
            invert_colormap=invert_colormap,
        )

    @Logger.loggable
    def to_dot(self, metric=None, name="name", rank=0, thread=0, threshold=0.0):
        """Write the graph in the graphviz dot format:
        https://www.graphviz.org/doc/info/lang.html
        """
        if metric is None:
            metric = self.default_metric
        return trees_to_dot(
            self.graph.roots, self.dataframe, metric, name, rank, thread, threshold
        )

    @Logger.loggable
    def to_flamegraph(self, metric=None, name="name", rank=0, thread=0, threshold=0.0):
        """Write the graph in the folded stack output required by FlameGraph
        http://www.brendangregg.com/flamegraphs.html
        """
        folded_stack = ""
        if metric is None:
            metric = self.default_metric

        for root in self.graph.roots:
            for hnode in root.traverse():
                callpath = hnode.path()
                for i in range(0, len(callpath) - 1):
                    if (
                        "rank" in self.dataframe.index.names
                        and "thread" in self.dataframe.index.names
                    ):
                        df_index = (callpath[i], rank, thread)
                    elif "rank" in self.dataframe.index.names:
                        df_index = (callpath[i], rank)
                    elif "thread" in self.dataframe.index.names:
                        df_index = (callpath[i], thread)
                    else:
                        df_index = callpath[i]
                    folded_stack = (
                        folded_stack + str(self.dataframe.loc[df_index, "name"]) + "; "
                    )

                if (
                    "rank" in self.dataframe.index.names
                    and "thread" in self.dataframe.index.names
                ):
                    df_index = (callpath[-1], rank, thread)
                elif "rank" in self.dataframe.index.names:
                    df_index = (callpath[-1], rank)
                elif "thread" in self.dataframe.index.names:
                    df_index = (callpath[-1], thread)
                else:
                    df_index = callpath[-1]
                folded_stack = (
                    folded_stack + str(self.dataframe.loc[df_index, "name"]) + " "
                )

                # set dataframe index based on if rank and thread are part of the index
                if (
                    "rank" in self.dataframe.index.names
                    and "thread" in self.dataframe.index.names
                ):
                    df_index = (hnode, rank, thread)
                elif "rank" in self.dataframe.index.names:
                    df_index = (hnode, rank)
                elif "thread" in self.dataframe.index.names:
                    df_index = (hnode, thread)
                else:
                    df_index = hnode

                folded_stack = (
                    folded_stack + str(self.dataframe.loc[df_index, metric]) + "\n"
                )

        return folded_stack

    @Logger.loggable
    def to_literal(self, name="name", rank=0, thread=0, cat_columns=[]):
        """Format this graph as a list of dictionaries for Roundtrip
        visualizations.
        """
        graph_literal = []
        visited = []

        def _get_df_index(hnode):
            if (
                "rank" in self.dataframe.index.names
                and "thread" in self.dataframe.index.names
            ):
                df_index = (hnode, rank, thread)
            elif "rank" in self.dataframe.index.names:
                df_index = (hnode, rank)
            elif "thread" in self.dataframe.index.names:
                df_index = (hnode, thread)
            else:
                df_index = hnode

            return df_index

        def metrics_to_dict(df_index):
            metrics_dict = {}
            for m in sorted(self.inc_metrics + self.exc_metrics):
                node_metric_val = self.dataframe.loc[df_index, m]
                metrics_dict[m] = node_metric_val

            return metrics_dict

        def attributes_to_dict(df_index):
            valid_columns = [
                col for col in cat_columns if col in self.dataframe.columns
            ]

            attributes_dict = {}
            for m in sorted(valid_columns):
                node_attr_val = self.dataframe.loc[df_index, m]
                attributes_dict[m] = node_attr_val

            return attributes_dict

        def add_nodes(hnode):
            df_index = _get_df_index(hnode)

            node_dict = {}

            node_name = self.dataframe.loc[df_index, name]

            node_dict["name"] = node_name
            node_dict["frame"] = hnode.frame.attrs
            node_dict["metrics"] = metrics_to_dict(df_index)
            node_dict["metrics"]["_hatchet_nid"] = hnode._hatchet_nid
            node_dict["attributes"] = attributes_to_dict(df_index)

            if hnode.children and hnode not in visited:
                visited.append(hnode)
                node_dict["children"] = []

                for child in sorted(hnode.children, key=lambda n: n.frame):
                    node_dict["children"].append(add_nodes(child))

            return node_dict

        for root in sorted(self.graph.roots, key=lambda n: n.frame):
            graph_literal.append(add_nodes(root))

        return graph_literal

    def _operator(self, other, op):
        """Generic function to apply operator to two dataframes and store
        result in self.

        Arguments:
            self (graphframe): self's graphframe
            other (graphframe): other's graphframe
            op (operator): pandas arithmetic operator

        Return:
            (GraphFrame): self's graphframe modified
        """
        # unioned set of self and other exclusive and inclusive metrics
        all_metrics = list(
            set().union(
                self.exc_metrics, self.inc_metrics, other.exc_metrics, other.inc_metrics
            )
        )

        self.dataframe.update(op(other.dataframe[all_metrics]))

        return self

    def _insert_missing_rows(self, other):
        """Helper function to add rows that exist in other, but not in self.

        This returns a graphframe with a modified dataframe. The new rows will
        contain zeros for numeric columns.

        Return:
            (GraphFrame): self's modified graphframe
        """
        all_metrics = list(
            set().union(
                self.exc_metrics, self.inc_metrics, other.exc_metrics, other.inc_metrics
            )
        )

        # make two 2D nparrays arrays with two columns:
        # 1) the hashed value of a node and 2) a numerical index
        # Many operations are stacked here to reduce the need for storing
        # large intermediary datasets
        self_hsh_ndx = np.vstack(
            (
                np.array(
                    [x.__hash__() for x in self.dataframe["node"]], dtype=np.uint64
                ),
                self.dataframe.index.values.astype(np.uint64),
            )
        ).T
        other_hsh_ndx = np.vstack(
            (
                np.array(
                    [x.__hash__() for x in other.dataframe["node"]], dtype=np.uint64
                ),
                other.dataframe.index.values.astype(np.uint64),
            )
        ).T

        # sort our 2D arrays by hashed node value so a binary search can be used
        # in the cython function fast_not_isin
        self_hsh_ndx_sorted = self_hsh_ndx[self_hsh_ndx[:, 0].argsort()]
        other_hsh_ndx_sorted = other_hsh_ndx[other_hsh_ndx[:, 0].argsort()]

        # get nodes that exist in other, but not in self, set metric columns to 0 for
        # these rows
        other_not_in_self = other.dataframe[
            _gfm_cy.fast_not_isin(
                other_hsh_ndx_sorted,
                self_hsh_ndx_sorted,
                other_hsh_ndx_sorted.shape[0],
                self_hsh_ndx_sorted.shape[0],
            )
        ]
        # get nodes that exist in self, but not in other
        self_not_in_other = self.dataframe[
            _gfm_cy.fast_not_isin(
                self_hsh_ndx_sorted,
                other_hsh_ndx_sorted,
                self_hsh_ndx_sorted.shape[0],
                other_hsh_ndx_sorted.shape[0],
            )
        ]

        # if there are missing nodes in either self or other, add a new column
        # called _missing_node
        if not self_not_in_other.empty:
            self.dataframe = self.dataframe.assign(
                _missing_node=np.zeros(len(self.dataframe), dtype=np.short)
            )
        if not other_not_in_self.empty:
            # initialize with 2 to save filling in later
            other_not_in_self = other_not_in_self.assign(
                _missing_node=[int(2) for x in range(len(other_not_in_self))]
            )

            # add a new column to self if other has nodes not in self
            if self_not_in_other.empty:
                self.dataframe["_missing_node"] = np.zeros(
                    len(self.dataframe), dtype=np.short
                )

        # get lengths to pass into
        onis_len = len(other_not_in_self)
        snio_len = len(self_not_in_other)

        # case where self is a superset of other
        if snio_len != 0:
            self_missing_node = self.dataframe["_missing_node"].values
            snio_indices = self_not_in_other.index.values

            # This function adds 1 to all nodes in self.dataframe['_missing_node'] which
            # are in self but not in the other graphframe
            _gfm_cy.insert_one_for_self_nodes(snio_len, self_missing_node, snio_indices)
            self.dataframe["_missing_node"] = np.array(
                [n for n in self_missing_node], dtype=np.short
            )

        # for nodes that only exist in other, set the metric to be nan (since
        # it's a missing node in self)
        # replaces individual metric assignments with np.zeros
        for j in all_metrics:
            other_not_in_self[j] = np.full(onis_len, np.nan)

        # append missing rows (nodes that exist in other, but not in self) to self's
        # dataframe
        self.dataframe = pd.concat(
            [self.dataframe, other_not_in_self], axis=0, sort=True
        )

        return self

    @Logger.loggable
    def groupby_aggregate(self, groupby_column, agg_function):
        """Groupby-aggregate dataframe and reindex the Graph.

        Reindex the graph to match the groupby-aggregated dataframe.

        Update the frame attributes to contain those columns in the dataframe index.

        Arguments:
            self (graphframe): self's graphframe
            groupby_column: column to groupby on dataframe
            agg_function: aggregate function on dataframe

        Return:
            (GraphFrame): new graphframe with reindexed graph and groupby-aggregated dataframe
        """
        # create new nodes for each unique node in the old dataframe
        # length is equal to number of nodes in original graph
        old_to_new = {}

        # list of new roots
        new_roots = []

        # dict of (new) super nodes
        # length is equal to length of dataframe index (after groupby-aggregate)
        node_dicts = []

        def reindex(node, parent, visited):
            """Reindex the graph.

            Connect super nodes to children according to relationships from old graph.
            """
            # grab the super node corresponding to original node
            super_node = old_to_new.get(node)

            if not node.parents and super_node not in new_roots:
                # this is a new root
                new_roots.append(super_node)

            # iterate over parents of old node, adding parents to super node
            for parent in node.parents:
                # convert node to super node
                snode = old_to_new.get(parent)
                # move to next node if parent and super node are to be merged
                if snode == super_node:
                    continue
                # add node to super node's parents if parent does not exist in super
                # node's parents
                if snode not in super_node.parents:
                    super_node.add_parent(snode)

            # iterate over children of old node, adding children to super node
            for child in node.children:
                # convert node to super node
                snode = old_to_new.get(child)
                # move to next node if child and super node are to be merged
                if snode == super_node:
                    continue
                # add node to super node's children if child does not exist in super
                # node's children
                if snode not in super_node.children:
                    super_node.add_child(snode)

            if node not in visited:
                visited.add(node)
                for child in node.children:
                    reindex(child, super_node, visited)

        # groupby-aggregate dataframe based on user-supplied functions
        groupby_obj = self.dataframe.groupby(groupby_column)
        agg_df = groupby_obj.agg(agg_function)

        # traverse groupby_obj, determine old node to super node mapping
        nid = 0
        for k, v in groupby_obj.groups.items():
            node_name = k
            node_type = agg_df.index.name
            super_node = Node(Frame({"name": node_name, "type": node_type}), None, nid)
            n = {"node": super_node, "nid": nid, "name": node_name}
            node_dicts.append(n)
            nid += 1

            # if many old nodes map to the same super node
            for i in v:
                old_to_new[i] = super_node

        # reindex graph by traversing old graph
        visited = set()
        for root in self.graph.roots:
            reindex(root, None, visited)

        # append super nodes to groupby-aggregate dataframe
        df_index = list(agg_df.index.names)
        agg_df.reset_index(inplace=True)
        df_nodes = pd.DataFrame.from_dict(data=node_dicts)
        tmp_df = pd.concat([agg_df, df_nodes], axis=1)
        # add node to dataframe index if it doesn't exist
        if "node" not in df_index:
            df_index.append("node")
        # reset index
        tmp_df.set_index(df_index, inplace=True)

        # update _hatchet_nid in reindexed graph and groupby-aggregate dataframe
        graph = Graph(new_roots)
        graph.enumerate_traverse()

        # put it all together
        new_gf = GraphFrame(
            graph,
            tmp_df,
            list(self.exc_metrics),
            list(self.inc_metrics),
            self.default_metric,
            dict(self.metadata),
            attributes=dict([[x, getattr(self, x)] for x in self.attributes]),
        )
        new_gf.drop_index_levels()
        return new_gf

    @Logger.loggable
    def flat_profile(self, groupby_column=None, as_index=True):
        """Generates flat profile for a given graphframe.
        Returns a new dataframe."""
        return Chopper().flat_profile(self, groupby_column, as_index)

    @Logger.loggable
    def flatten(self, groupby_column=None):
        """
        Flattens the graphframe by changing its graph structure and the dataframe.
        """
        return Chopper().flatten(self, groupby_column)

    @Logger.loggable
    def to_callgraph(self):
        """
        Converts a CCT to a callgraph.
        Returns a new graphframe.
        """
        return Chopper().to_callgraph(self)

    @Logger.loggable
    def load_imbalance(self, metric_column=None, threshold=None):
        """Calculates load imbalance for given metric column(s)
        Takes a graphframe and a list of metric column(s), and
        returns a new graphframe with metric.imbalance column(s).
        """
        return Chopper().load_imbalance(self, metric_column, threshold)

    @Logger.loggable
    def hot_path(self, start_node=None, metric=None, threshold=0.5):
        """Returns the hot_path function.
        Inputs:
         - start_node: Start node of the hot path should be given.
         - metric: A numerical metric on the dataframe
         - threshold: Threshold for parent-child comparison (parent <= child/2).
        Output:
         - hot_path: list of nodes, starting from the start node to the hot node.

        Example:
        root_node = graphframe.graph.roots[0]
        graphframe.hot_path(root_node)
        """
        # call hot_path function on high-level API
        hot_path = Chopper().hot_path(
            self, start_node, metric, threshold, callpath=[start_node]
        )
        return hot_path

    @Logger.loggable
    def add(self, other):
        """Returns the column-wise sum of two graphframes as a new graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        # create a copy of both graphframes
        self_copy = self.copy()
        other_copy = other.copy()

        # unify copies of graphframes
        self_copy.unify(other_copy)

        return self_copy._operator(other_copy, self_copy.dataframe.add)

    @Logger.loggable
    def sub(self, other):
        """Returns the column-wise difference of two graphframes as a new
        graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        # create a copy of both graphframes
        self_copy = self.copy()
        other_copy = other.copy()

        # unify copies of graphframes
        self_copy.unify(other_copy)

        return self_copy._operator(other_copy, self_copy.dataframe.sub)

    @Logger.loggable
    def div(self, other):
        """Returns the column-wise float division of two graphframes as a new graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        # create a copy of both graphframes
        self_copy = self.copy()
        other_copy = other.copy()

        # unify copies of graphframes
        self_copy.unify(other_copy)

        return self_copy._operator(other_copy, self_copy.dataframe.divide)

    @Logger.loggable
    def mul(self, other):
        """Returns the column-wise float multiplication of two graphframes as a new graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        # create a copy of both graphframes
        self_copy = self.copy()
        other_copy = other.copy()

        # unify copies of graphframes
        self_copy.unify(other_copy)

        return self_copy._operator(other_copy, self_copy.dataframe.multiply)

    def __iadd__(self, other):
        """Computes column-wise sum of two graphframes and stores the result in
        self.

        Self's graphframe is the union of self's and other's graphs, and the
        node handles from self will be rewritten with this operation. This
        operation does not modify other.

        Return:
            (GraphFrame): self's graphframe modified
        """
        # create a copy of other's graphframe
        other_copy = other.copy()

        # unify self graphframe and copy of other graphframe
        self.unify(other_copy)

        return self._operator(other_copy, self.dataframe.add)

    def __add__(self, other):
        """Returns the column-wise sum of two graphframes as a new graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        return self.add(other)

    def __mul__(self, other):
        """Returns the column-wise multiplication of two graphframes as a new graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        return self.mul(other)

    def __isub__(self, other):
        """Computes column-wise difference of two graphframes and stores the
        result in self.

        Self's graphframe is the union of self's and other's graphs, and the
        node handles from self will be rewritten with this operation. This
        operation does not modify other.

        Return:
            (GraphFrame): self's graphframe modified
        """
        # create a copy of other's graphframe
        other_copy = other.copy()

        # unify self graphframe and other graphframe
        self.unify(other_copy)

        return self._operator(other_copy, self.dataframe.sub)

    def __sub__(self, other):
        """Returns the column-wise difference of two graphframes as a new
        graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        return self.sub(other)

    def __idiv__(self, other):
        """Computes column-wise float division of two graphframes and stores the
        result in self.

        Self's graphframe is the union of self's and other's graphs, and the
        node handles from self will be rewritten with this operation. This
        operation does not modify other.

        Return:
            (GraphFrame): self's graphframe modified
        """
        # create a copy of other's graphframe
        other_copy = other.copy()

        # unify self graphframe and other graphframe
        self.unify(other_copy)

        return self._operator(other_copy, self.dataframe.div)

    def __truediv__(self, other):
        """Returns the column-wise float division of two graphframes as a new
        graphframe.

        This graphframe is the union of self's and other's graphs, and does not
        modify self or other.

        Return:
            (GraphFrame): new graphframe
        """
        return self.div(other)

    def __imul__(self, other):
        """Computes column-wise float multiplication of two graphframes and stores the
        result in self.

        Self's graphframe is the union of self's and other's graphs, and the
        node handles from self will be rewritten with this operation. This
        operation does not modify other.

        Return:
            (GraphFrame): self's graphframe modified
        """
        # create a copy of other's graphframe
        other_copy = other.copy()

        # unify self graphframe and other graphframe
        self.unify(other_copy)

        return self._operator(other_copy, self.dataframe.mul)


class InvalidFilter(Exception):
    """Raised when an invalid argument is passed to the filter function."""


class EmptyFilter(Exception):
    """Raised when a filter would otherwise return an empty GraphFrame."""
