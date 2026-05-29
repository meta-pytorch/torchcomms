# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import warnings
import sys

from .dataframe_writer import DataframeWriter


class HDF5Writer(DataframeWriter):
    def __init__(self, filename):
        if sys.version_info[0] == 2:
            super(HDF5Writer, self).__init__(filename)
        else:
            super().__init__(filename)

    def _write_dataframe_to_file(self, df, **kwargs):
        if "key" not in kwargs:
            raise KeyError("Writing to HDF5 requires a user-supplied key")
        key = kwargs["key"]
        del kwargs["key"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            df.to_hdf(self.filename, key, **kwargs)
