#!/usr/bin/env python

with open("./hatchet/version.py", "r") as fp:
    version = {}
    exec(fp.read(), version)
    version = version["__version__"]
    print("{}".format(version))
