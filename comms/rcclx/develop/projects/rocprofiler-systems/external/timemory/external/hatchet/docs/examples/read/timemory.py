#!/usr/bin/env python

import argparse
import hatchet as ht

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--metric", default="sum", type=str, help="Metric to display"
    )
    args, argv = parser.parse_known_args()
    files = argv[:]
    if not files:
        files.append("../../../hatchet/tests/data/timemory/wall.tree.json")

    for json_file in files:
        # Use hatchet's ``from_timemory`` API with the hierarchical json output.
        # The result is stored into Hatchet's GraphFrame.
        gf = ht.GraphFrame.from_timemory(json_file)

        # Printout the DataFrame component of the GraphFrame.
        print(gf.dataframe)

        # Printout the graph component of the GraphFrame.
        # one dimensional components use standardize metric labels,
        # e.g., sum, sum.inc, mean, mean.inc, etc.
        # but multi-dimensional data, such as the current-peak-rss
        # (which reports the peak RSS at the start of the marker
        # and at the end) and hardware-counter data, create
        # unique suffixes, e.g. sum.start-peak-rss,
        # sum.start-peak-rss, et.c
        print(gf.tree(args.metric))
