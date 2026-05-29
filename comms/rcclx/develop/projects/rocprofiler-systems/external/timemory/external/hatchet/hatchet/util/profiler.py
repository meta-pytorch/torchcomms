# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import cProfile
import traceback
import sys
import os

from datetime import datetime

try:
    from StringIO import StringIO  # python2
except ImportError:
    from io import StringIO  # python3
import pstats


def print_incomptable_msg(stats_file):
    """
    Function which makes the syntax cleaner in Profiler.write_to_file().
    """
    errmsg = """ Incompatible pstats file: {}\n Please run your code in Python {} to read in this file. """
    if sys.version_info[0] == 2:
        print(errmsg.format(stats_file, 3))
    if sys.version_info[0] == 3:
        print(errmsg.format(stats_file, 2.7))
    traceback.print_exc()


# class for profiling
class Profiler:
    """
    Wrapper class around cProfile.
    Exports a pstats file to be read by the pstats reader.
    """

    def __init__(self):
        self._prf = cProfile.Profile()
        self._output = "hatchet-profile"
        self._active = False

    def start(self):
        """
        Description: Place before the block of code to be profiled.
        """
        if self._active:
            print(
                "Start dissallowed in scope where profiler is running. Please add Profiler.stop() before attempting start."
            )
            raise

        self._active = True
        self._prf.enable()

    def stop(self):
        """
        Description: Place at the end of the block of code being profiled.
        """

        self._active = False
        self._prf.disable()
        self.write_to_file()

    def reset(self):
        """
        Description: Resets the profilier.
        """
        if self._active:
            print(
                "Reset dissallowed in scope where profiler is running. Please add Profiler.stop() before attempting reset."
            )
            raise

        self._prf = cProfile.Profile()

    def __str__(self):
        """
        Description: Writes stats object out as a string.
        """
        s = StringIO()
        pstats.Stats(self._prf, stream=s).print_stats()
        return s.getvalue()

    def write_to_file(self, filename="", add_pstats_files=[]):
        """
        Description: Write the pstats object to a binary
        file to be read in by an appropriate source.
        """
        sts = pstats.Stats(self._prf)

        if len(add_pstats_files) > 0:
            for stats_file in add_pstats_files:
                try:
                    sts.add(stats_file)
                except ValueError:
                    print_incomptable_msg(stats_file)
                    raise

        if filename == "":
            if os.path.exists(self._output + ".pstats"):
                now = datetime.now().strftime("%H%M%S")
                self.write_to_file(
                    "{}_{}.pstats".format(self._output, now), add_pstats_files
                )
            else:
                sts.dump_stats(self._output + ".pstats")
        else:
            if os.path.exists(filename):
                now = datetime.now().strftime("%H%M%S")
                filename = "{}_{}.pstats".format(filename, now)
            sts.dump_stats(filename)
