#!/usr/bin/env python

import hatchet as ht
from sys import version_info as python_version


if __name__ == "__main__":
    # Path to pstats file.
    # A pstats file produced with a particualar version of python (2 or 3) must be read in with that version
    if python_version[0] == 2:
        pstats_file = "../../../hatchet/tests/data/cprofile-hatchet-pstats/cprofile-cycle-py2.pstats"
    elif python_version[0] == 3:
        pstats_file = (
            "../../../hatchet/tests/data/cprofile-hatchet-pstats/cprofile-cycle.pstats"
        )

    # Use hatchet's ``from_cprofile`` API to read in the pstats file. The result
    # is stored into Hatchet's GraphFrame.
    gf = ht.GraphFrame.from_cprofile(pstats_file)

    # Printout the DataFrame component of the GraphFrame.
    print(gf.dataframe)

    # Printout the graph component of the GraphFrame.
    # Because no metric parameter is specified, ``time`` is used by default.
    print(gf.tree())
