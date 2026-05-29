#!/usr/bin/env python

import caliperreader as cr

import hatchet as ht


if __name__ == "__main__":
    # Path to caliper cali file.
    cali_file = "../../../hatchet/tests/data/caliper-example-cali/example-profile.cali"

    r = cr.CaliperReader()
    r.read(cali_file)

    gf = ht.GraphFrame.from_caliperreader(r)

    # Printout the DataFrame component of the GraphFrame.
    print(gf.dataframe)

    # Printout the graph component of the GraphFrame with the specified metric.
    print(gf.tree(metric_column="avg#inclusive#sum#time.duration"))
