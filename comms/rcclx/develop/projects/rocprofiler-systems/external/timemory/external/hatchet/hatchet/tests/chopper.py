# Copyright 2021-2023 University of Maryland and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
from hatchet.graphframe import GraphFrame
from hatchet.chopper import Chopper


def test_flat_profile(calc_pi_hpct_db):
    """Validate that the flat profile works correctly."""
    graphframe = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    original_rows = graphframe.dataframe["name"].unique()
    flat_profile = graphframe.flat_profile()

    # Check if the node names are exactly the same
    assert sorted(flat_profile.index.tolist()) == sorted(original_rows)

    # Check if the aggregated time is correct for the root node.
    root = graphframe.graph.roots[0]
    assert flat_profile.loc[root.frame["name"]]["time (inc)"] == np.mean(
        graphframe.dataframe.loc[root]["time (inc)"].to_numpy()
    )

    # Check if the aggregated time is correct for the '__GI_sched_yield' node.
    another = graphframe.dataframe[
        graphframe.dataframe["name"] == "__GI_sched_yield"
    ].index[0][0]
    assert flat_profile.loc[another.frame["name"]]["time (inc)"] == np.mean(
        graphframe.dataframe.loc[another]["time (inc)"].to_numpy()
    )

    # If as_index is False, check if the index is properly numbered
    flat_profile = graphframe.flat_profile(as_index=False)
    assert sorted(flat_profile.index.tolist()) == list(range(len(flat_profile)))


def test_load_imbalance(calc_pi_hpct_db):
    """Validate that the load imbalance is calculated correctly."""

    graphframe = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))
    load_imb_gf = graphframe.load_imbalance(metric_column="time (inc)", threshold=0.01)

    # Check if load imbalance is correct for the root node.
    root = graphframe.graph.roots[0]
    root_metric_max_org = np.max(graphframe.dataframe.loc[root, "time (inc)"])
    root_metric_mean_org = np.mean(graphframe.dataframe.loc[root, "time (inc)"])
    root_imbalance = load_imb_gf.dataframe.loc[root]["time (inc).imbalance"]
    assert root_metric_max_org / root_metric_mean_org == root_imbalance

    # Check if load imbalance is correct for the main node.
    main = root.children[0]
    main_metric_max_org = np.max(graphframe.dataframe.loc[main, "time (inc)"])
    main_metric_mean_org = np.mean(graphframe.dataframe.loc[main, "time (inc)"])
    main_imbalance = load_imb_gf.dataframe.loc[main]["time (inc).imbalance"]
    assert main_metric_max_org / main_metric_mean_org == main_imbalance

    # Check if load imbalance is correct for the the node '__GI_sched_yield'.
    another = graphframe.dataframe[
        graphframe.dataframe["name"] == "__GI_sched_yield"
    ].index[0][0]
    another_lb = load_imb_gf.dataframe[
        load_imb_gf.dataframe["name"] == "__GI_sched_yield"
    ].index[0]
    another_metric_max_org = np.max(graphframe.dataframe.loc[another, "time (inc)"])
    another_metric_mean_org = np.mean(graphframe.dataframe.loc[another, "time (inc)"])
    another_imbalance = load_imb_gf.dataframe.loc[another_lb]["time (inc).imbalance"]
    assert another_metric_max_org / another_metric_mean_org == another_imbalance


def test_hot_path(calc_pi_hpct_db):
    """Validate the hot path with known data from HPCToolkit."""
    graphframe = GraphFrame.from_hpctoolkit(str(calc_pi_hpct_db))

    # Correctness of this list is validated
    # via HPCViewer.
    hpctoolkit_hot_path = [
        "<program root>",
        "main",
        "62:MPI_Finalize",
        "PMPI_Finalize",
        "294:MPID_Finalize",
        "162:MPIDI_CH3_Finalize",
        "230:psm_dofinalize",
        "36:<unknown procedure>",
        "<unknown procedure>",
        "<unknown procedure>",
        "<unknown procedure>",
    ]

    hot_path = graphframe.hot_path(metric="time (inc)")

    for node in hot_path:
        assert node.frame["name"] in hpctoolkit_hot_path

    # test if parameters override.
    # should return the root if threshold equals 2
    hot_path = graphframe.hot_path(metric="time (inc)", threshold=2)
    assert len(hot_path) == 2
    assert hot_path[0].frame["name"] == "<program root>"


def test_multirun_analysis_lulesh(lulesh_caliper_json):
    """Validate that multirun_analysis works correctly with data containing
    non-repeating functions."""
    gf1 = GraphFrame.from_caliper(lulesh_caliper_json)
    gf2, gf4, gf8 = gf1.deepcopy(), gf1.deepcopy(), gf1.deepcopy()

    gf1.update_metadata(1)
    gf2.update_metadata(2)
    gf4.update_metadata(4)
    gf8.update_metadata(8)

    gf1_copy, gf2_copy, gf4_copy, gf8_copy = (
        gf1.deepcopy(),
        gf2.deepcopy(),
        gf4.deepcopy(),
        gf8.deepcopy(),
    )

    # drop index levels, filter values below threshold of 500000, group nodes by name,
    # remove all columns except time, and replace time column with num_processes
    # for each dataframe
    for gf in [gf1, gf2, gf4, gf8]:
        gf.drop_index_levels()
        filtered_rows = gf.dataframe.apply(lambda x: x["time"] > 500000.0, axis=1)
        gf.dataframe = gf.dataframe[filtered_rows]
        gf.dataframe = gf.dataframe.groupby("name").sum()
        gf.dataframe = gf.dataframe[["time"]]
        gf.dataframe = gf.dataframe.rename(
            {"time": gf.metadata["num_processes"]}, axis="columns"
        )

    # join the dataframes column-wise, transpose the dataframe, and set the index name
    # to num_processes
    df_dummy = pd.concat(
        [gf1.dataframe, gf2.dataframe, gf4.dataframe, gf8.dataframe], axis=1
    )
    df_dummy = df_dummy.transpose()
    df_dummy.index.name = "num_processes"

    # run multirun_analyis
    df_test = Chopper.multirun_analysis(
        graphframes=[gf1_copy, gf2_copy, gf4_copy, gf8_copy],
        pivot_index="num_processes",
        columns="name",
        metric="time",
        threshold=500000.0,
    )

    # check if the test and dummy dataframes match
    assert df_test.equals(df_dummy)


def test_multirun_analysis_literal(mock_graph_literal):
    """Validate that multirun_analysis works correctly with data containing
    repeating functions."""
    gf1 = GraphFrame.from_literal(mock_graph_literal)
    gf2, gf4, gf8 = gf1.deepcopy(), gf1.deepcopy(), gf1.deepcopy()

    gf1.update_metadata(1)
    gf2.update_metadata(2)
    gf4.update_metadata(4)
    gf8.update_metadata(8)

    gf1_copy, gf2_copy, gf4_copy, gf8_copy = (
        gf1.deepcopy(),
        gf2.deepcopy(),
        gf4.deepcopy(),
        gf8.deepcopy(),
    )

    # filter values below threshold of 5.0, group nodes by name,
    # remove all columns except time, and replace time column with num_processes
    # for each dataframe
    for gf in [gf1, gf2, gf4, gf8]:
        filtered_rows = gf.dataframe.apply(lambda x: x["time"] > 5.0, axis=1)
        gf.dataframe = gf.dataframe[filtered_rows]
        gf.dataframe = gf.dataframe.drop_duplicates()
        gf.dataframe = gf.dataframe.groupby("name").sum()
        gf.dataframe = gf.dataframe[["time"]]
        gf.dataframe = gf.dataframe.rename(
            {"time": gf.metadata["num_processes"]}, axis="columns"
        )

    # join the dataframes column-wise, transpose the dataframe, and set the index name
    # to num_processes
    df_dummy = pd.concat(
        [gf1.dataframe, gf2.dataframe, gf4.dataframe, gf8.dataframe], axis=1
    )
    df_dummy = df_dummy.transpose()
    df_dummy.index.name = "num_processes"

    # run multirun_analyis
    df_test = Chopper.multirun_analysis(
        graphframes=[gf1_copy, gf2_copy, gf4_copy, gf8_copy],
        pivot_index="num_processes",
        columns="name",
        metric="time",
        threshold=5.0,
    )

    # check if the test and dummy dataframes match
    assert df_test.equals(df_dummy)
