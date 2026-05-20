# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os


def which(executable):
    """Finds an `executable` in the user's PATH like command-line which.

    Args:
        executable (str): executable to search for
    """
    path = os.environ.get("PATH", "/usr/sbin:/usr/bin:/sbin:/bin")
    path = path.split(os.pathsep)

    for directory in path:
        exe = os.path.join(directory, executable)
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return exe

    return None
