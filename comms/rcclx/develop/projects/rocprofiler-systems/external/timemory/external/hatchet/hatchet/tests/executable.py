# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os
import stat

import pytest

from hatchet.util.executable import which


@pytest.fixture
def mock_cali_query(tmpdir):
    """Create a mock cali-query path."""
    tmpdir = tmpdir.mkdir("tmp-bin")
    cali_query = tmpdir.join("cali-query")
    with cali_query.open("w") as file:
        file.write("")

    st = os.stat(str(cali_query))
    os.chmod(str(cali_query), st.st_mode | stat.S_IEXEC)

    # save current PATH variable
    old_path = os.environ.get("PATH")
    # append tmpdir to PATH variable
    os.environ["PATH"] = "%s:%s" % (str(tmpdir), old_path)
    # send it
    yield tmpdir
    # restore original PATH variable
    os.environ["PATH"] = old_path


def test_which(mock_cali_query):
    assert which("cali-query")
