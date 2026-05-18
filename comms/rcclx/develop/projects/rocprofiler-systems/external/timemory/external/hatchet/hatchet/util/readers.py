# Copyright 2023 Advanced Micro Devices, Inc., and other Hatchet Project Developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd

from ..graph import Graph


def graphframe_indexing_helper(
    root_nodes,
    data,
    extensions=["rank", "thread"],
    columns=[],
    fill_value=0,
    fill_missing_attribute={},
    deepcopy=True,
):
    """Helper for the complicated scenario of reindexing the dataframe
    to a multi-index for dataframes with multiple ranks and/or threads.

    In every readers, the implementation boils down to the creation of
    two principal data structures:

        1. list of root nodes
        2. list of dicts containing table data: node, rank, thread, time, time (inc), etc.

    once these are created, the implementation varies from reader to reader
    but it all boils down to:

        1. create Graph from root nodes + enumerate_traverse()
        2. create dataframe, set index, sort index

    However, when multiple ranks and/or threads are introduced, a MultiIndex
    needs to be used but creating a MultiIndex is problematic because it
    essentially results in pandas giving each rank/thread/rank+thread the
    same call-stack, e.g. if rank 0 called 'foo' and rank 1 called 'bar'
    and A in Node({name: foo}) and B is Node({name: bar}):

        node    rank    name    time
        A       0       foo     1.0
        B       1       bar     0.9

    then a multi-index for ["node", "rank"] requires A+0, A+1, B+0, B+1
    so it fills in B+0 and A+1 since only A+0 and B+1 exist in the
    original data. Thus you end up with a data frame that looks like this:

        node    rank    name    time
        A       0       foo     1.0
        B       0       NaN     NaN
        A       1       NaN     NaN
        B       1       bar     0.9

    Only the columns in the index are copied so we need to "fill" in the
    relevant data to these new entries. We do this by taking the data
    encoded in the dictionary of the node column and copying it over
    into the columns with the same name.
    Since, A+1 and B+0 did not actually exist in the trace/profile,
    we can ignore columns such as "time" so we set it to zero, however,
    the name column is important bc it identifies the name of the function.
    We do that by copying the data encoded into the dict within the node
    column. I.e. since A+1 was created from A+0, it copied over the
    Node({name: foo}) and we can essentially want to do:

    for row in df.rows:
        row["name"] = row["node"].frame["name"]

    (i.e. dict {name: foo} is stored in Node.frame)

    Performing this operation is a bit difficult to perform efficiently
    because the most intuitive was (with dataframe.iterrows) is extremely
    slow. Thus we provide this helper function.

    Args:
        root_nodes (list of hatchet.Node):
            root nodes

        data (list/tuple of dict or pandas.dataframe):
            dict entries are {column_name: value}

        extensions (list of str):
            possible multi-index column names. "node" column is required by
            hatchet, "rank" and "thread" columns are optional. It is safe to
            use the default value even if you do not have these columns:
            this function checks if the columns even exist and also checks to
            make sure there is more than 1 unique value -- in other words, if
            you have both "rank" and "thread" columns but every row has a
            value of 0 and 0, then it is not necessary to create a multi-index
            and this function recognizes that. Similarly, if the data is
            single rank but multiple threads, this function will only create
            a multi-index on "node" and "thread" since factoring in
            "rank" to the multi-index is unnecessary

        columns (list of str):
            This is the dict keys from Node.frame whose values should be copied
            over into the dataframe columns. When set to an empty list, it
            will take the keys from the first Node.frame and use those. Thus,
            EVERY Node.frame SHOULD HAVE THE SAME SET OF KEYS. If, for example,
            sometimes you have file information for a node but not for others,
            set Node.frame["file"] = "<unknown>" or whatever you prefer for the
            value, just make sure the key exists and is neither None nor NaN.

        fill_value (number):
            In the case of multi-index, the NaN's need to be replaced with some
            value, zero is recommended.

        fill_missing_attribute (dict):
            If Node.frame does not have attribute with a specific key, fill with
            the given value instead of using fill_value. E.g.,
            if columns=["name", "file", "line"] and fill_value=0 and the
            Node.frame at a given index only has a "name" attribute, and you
            want the "file" column to be set to "<unknown>" instead of
            0 and the "line" column to be set to -1 then pass the argument:

                fill_missing_attribute={ "file": "<unknown>", "line": -1 }

        deep_copy (bool):
            Perform a full deep-copy of the dataframe (if data argument is
            dataframe). If you are not using the dataframe afterwards or
            assigning the returned dataframe to , set
            to False.
    """

    # make a copy since we cannot update argument inplace
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise RuntimeError("dataframe is empty")
        df_value = data.copy(deep=deepcopy)
    elif isinstance(data, (tuple, list)):
        if not data:
            raise RuntimeError("data is empty")
        elif not isinstance(data[0], dict):
            raise RuntimeError("data in list/tuple is not a dict")
        df_value = pd.DataFrame(data=list(data))
    else:
        raise TypeError("data type must be: pandas.DataFrame, list/tuple of dicts")

    graph_value = Graph(root_nodes)
    graph_value.enumerate_traverse()

    if df_value.empty:
        raise ValueError("dataframe is empty")

    indices = ["node"]
    for ext in extensions:
        if ext in df_value:
            n = len(df_value[ext].unique())
            if n > 1:
                indices.append(ext)

    # if there is no need for a multi-index: set index, sort, and return
    if len(indices) == 1:
        df_value.set_index(indices, inplace=True)
        df_value.sort_index(inplace=True)
        return (graph_value, df_value)

    new_index = pd.MultiIndex.from_product(
        [df_value[itr].unique() for itr in indices],
        names=indices,
    )
    df_value.set_index(indices, inplace=True)
    df_value = df_value.reindex(new_index, fill_value=fill_value)
    df_value.sort_index(inplace=True)

    # set columns to the list of columns to pull from Node.frame and
    # propagate to a column
    #
    # if no additional columns should be copied (aside from name), set columns=None
    # otherwise, if columns is empty list, find first Node().frame and extract the keys
    # not named "node"
    if not columns and columns is not None:
        columns = df_value.first_valid_index()[0].frame.attrs.keys()

    if columns is not None:
        # filter out "node" from columns
        columns = [x for x in columns if x not in ["node"]]
        # make sure name is copy cols
        if "name" not in columns:
            columns.append("name")
    else:
        columns = ["name"]

    # Select rows where name is 0, we need to fix that.
    zeros = df_value.loc[df_value["name"] == fill_value]

    # get a list of the indices
    zeros_idx = zeros.index.to_list()
    # make a list of lists where each inner list (columns) are:
    #   0  -> the indices
    #   1+ -> Node.frame values for the columns we are copying
    # we will zip together the outer list to enable iterating over
    # data
    zeros_data = [zeros_idx] + [
        [
            # extract the value if it exists, otherwise check fill_missing_attribute
            # for a potential override of fill_value
            itr[0].frame.get(col, fill_missing_attribute.get(col, fill_value))
            for itr in zeros_idx
        ]
        for col in columns
    ]

    # put values from Node.frame dictionary into column
    for ditr in zip(*zeros_data):
        # we placed the index at pos 0 and then
        # all the columns we are copying are in the same order
        # as columns but offset by 1 (because 0 is index)
        index = ditr[0]
        for idx, col in enumerate(columns):
            if df_value.at[index, col] == fill_value:
                df_value.at[index, col] = ditr[idx + 1]

    return (graph_value, df_value)
