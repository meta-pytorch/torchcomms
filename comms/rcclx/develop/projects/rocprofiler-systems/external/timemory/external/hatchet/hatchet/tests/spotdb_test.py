# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pytest

from hatchet import GraphFrame
from hatchet.readers.spotdb_reader import SpotDatasetReader, SpotDBReader

spotdb_avail = True
try:
    import spotdb  # noqa: F401
except ImportError:
    spotdb_avail = False


def test_spot_dataset_reader():
    """Sanity-check the Spot dataset reader"""

    regionprofile = {
        "a/b/c": {"m": 20, "m#inclusive": 20},
        "a/b": {"m#inclusive": 40},
        "a": {"m#inclusive": 42},
    }
    metadata = {"launchdate": 123456789}
    attr_info = {
        "m": {"type": "double"},
        "m#inclusive": {"type": "int", "alias": "M Alias"},
    }

    reader = SpotDatasetReader(regionprofile, metadata, attr_info)
    gf = reader.read(default_metric="M Alias (inc)")

    assert len(gf.dataframe) == 3
    assert set(gf.dataframe.columns) == {"name", "m", "M Alias (inc)"}

    assert gf.metadata["launchdate"] == metadata["launchdate"]
    assert gf.default_metric == "M Alias (inc)"


@pytest.mark.skipif(not spotdb_avail, reason="spotdb module not available")
def test_spotdb_reader(spotdb_data):
    """Sanity check for the SpotDB reader"""

    db = spotdb_data

    reader = SpotDBReader(db)
    gfs = reader.read()

    assert len(gfs) == 4

    metrics = {"Total time (inc)", "Avg time/rank (inc)"}

    assert len(gfs[0].dataframe) > 2
    assert gfs[0].default_metric == "Total time (inc)"
    assert metrics < set(gfs[0].dataframe.columns)
    assert metrics < set(gfs[3].dataframe.columns)

    assert "launchdate" in gfs[0].metadata.keys()


@pytest.mark.skipif(not spotdb_avail, reason="spotdb module not available")
def test_from_spotdb(spotdb_data):
    """Sanity check for GraphFrame.from_spotdb"""

    db = spotdb_data
    runs = db.get_all_run_ids()
    gfs = GraphFrame.from_spotdb(spotdb_data, runs[0:2])

    assert len(gfs) == 2

    metrics = {"Total time (inc)", "Avg time/rank (inc)"}

    assert len(gfs[0].dataframe) > 2
    assert gfs[0].default_metric == "Total time (inc)"
    assert metrics < set(gfs[0].dataframe.columns)

    assert "launchdate" in gfs[0].metadata.keys()
