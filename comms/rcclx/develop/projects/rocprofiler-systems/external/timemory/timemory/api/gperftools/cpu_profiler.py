#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2020, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from . import utils


__all__ = ["execute"]


def execute(
    cmd,
    prefix="cpu.prof",
    freq=250,
    malloc_stats=0,
    realtime=1,
    preload=True,
    selected=0,
    image_type="jpeg",
    echo_dart=False,
    libs=[],
    args=["--no_strip_temp", "--functions"],
    generate=["text", "cum", "dot"],
    dot_args=[],
):
    os.environ["CPUPROFILE_FREQUENCY"] = "{}".format(freq)
    os.environ["MALLOCSTATS"] = "{}".format(malloc_stats)
    os.environ["CPUPROFILE_REALTIME"] = "{}".format(realtime)
    os.environ["PROFILESELECTED"] = "{}".format(selected)

    if preload:
        _libpath = None
        for libname in ("profiler", "tcmalloc_and_profiler"):
            if _libpath is None:
                _libpath = utils.find_library_path(libname)
        if _libpath is None:
            raise RuntimeError("Preload failed. Cannot find libprofiler")
        utils.add_preload(_libpath)

    exe = os.path.basename(cmd[0])
    prefix += ".{}".format(exe)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    N = 0
    fprefix = os.path.join(prefix, "gperf.")
    fname = fprefix + "{}".format(N)
    while os.path.exists(fname):
        N += 1
        fname = fprefix + "{}".format(N)

    os.environ["CPUPROFILE"] = "{}".format(fname)

    print("exeuting: {}, exe: {}".format(cmd, exe))
    utils.execute(cmd, "{}.log".format(fname))

    utils.post_process(
        cmd[0], fname, image_type, echo_dart, libs, args, generate, dot_args
    )
