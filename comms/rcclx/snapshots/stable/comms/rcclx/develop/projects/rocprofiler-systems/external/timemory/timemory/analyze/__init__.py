#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2018, The Regents of the University of California,
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

from __future__ import absolute_import
from __future__ import division

import os
import sys
import argparse

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

from .analyze import (
    load,
    match,
    search,
    expression,
    sort,
    group,
    add,
    subtract,
    unify,
    dump,
)


def embedded_analyze(
    args=None,
    data=[],
    call_exit=False,
    verbose=os.environ.get("TIMEMORY_VERBOSE", 0),
    ranks=[],
):
    """This is intended to be called from the embedded python interpreter"""
    if len(ranks) > 0:
        try:
            import mpi4py  # noqa: F401
            from mpi4py import MPI  # noqa: F401

            rank = MPI.COMM_WORLD.Get_rank()
            if rank not in ranks:
                return
        except (ImportError, RuntimeError):
            pass

    call_exit = False
    cmd_line = False
    if args is None:
        args = sys.argv[:]
        call_exit = True
        cmd_line = True

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "files",
            metavar="file",
            type=str,
            nargs="*" if not cmd_line else "+",
            help="Files to analyze",
            default=[],
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            help="Data output format.",
            choices=(
                "dot",
                "flamegraph",
                "tree",
                "table",
                "markdown",
                "html",
                "markdown_grid",
            ),
            nargs="*",
            default=None,
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            help="Output to file(s)",
            nargs="*",
            default=None,
        )
        parser.add_argument(
            "-M",
            "--mode",
            help="Analysis mode",
            nargs="*",
            choices=("add", "subtract", "unify", "group"),
            default=[],
        )
        parser.add_argument(
            "-m",
            "--metric",
            "--column",
            type=str,
            help="Metric(s) to extract",
            default="sum.*(.inc)$",
        )
        parser.add_argument(
            "-s",
            "--sort",
            type=str,
            help="Sort the metric",
            choices=("ascending", "descending"),
            default=None,
        )
        parser.add_argument(
            "-g",
            "--group",
            type=str,
            help="Group by a column name in the dataframe",
            default=None,
        )
        parser.add_argument(
            "--field",
            type=str,
            help="Dataframe column to search/match against",
            default="name",
        )
        parser.add_argument(
            "--search",
            type=str,
            default=None,
            help="Regular expression for re.search(...), i.e. a substring match",
        )
        parser.add_argument(
            "--match",
            type=str,
            default=None,
            help="Regular expression for re.match(...), i.e. a full string match",
        )
        parser.add_argument(
            "--expression",
            type=str,
            default=None,
            help=(
                "A space-delimited comparison operation expression using 'x' "
                + "for the variable, numerical values, and: < <= > >= && ||. "
                + "E.g. 'x > 1.0e3 && x < 100000'. 'x' will be -m/--metric"
            ),
        )
        parser.add_argument(
            "-e",
            "--echo-dart",
            help="echo Dart measurement for CDash",
            action="store_true",
        )
        parser.add_argument(
            "--per-thread",
            help=(
                "Encode the thread ID in node hash to ensure squashing doesn't "
                + "combine thread-data"
            ),
            action="store_true",
        )
        parser.add_argument(
            "--per-rank",
            help=(
                "Encode the rank ID in node hash to ensure squashing doesn't "
                + "combine rank-data"
            ),
            action="store_true",
        )
        parser.add_argument(
            "--select",
            help=(
                "Select the component type if the JSON input contains output "
                + "from multiple components"
            ),
            type=str,
            default=None,
        )
        parser.add_argument(
            "--exit-on-failure",
            help="Abort with non-zero exit code if errors arise",
            action="store_true",
        )

        _args = parser.parse_args(args)

        if _args.exit_on_failure:
            call_exit = True

        if _args.group is None and _args.format is None:
            _args.format = ["tree"]
        if _args.group is not None and _args.format not in ("table", None):
            raise RuntimeError("Invalid data format for group")

        if isinstance(_args.select, str):
            _args.select = _args.select.split()

        gfs = []
        for itr in data:
            if not isinstance(itr, dict):
                print("data: {}\ndata-type: {}".format(itr, type(itr).__name__))
            gfs.append(
                load(
                    itr,
                    select=_args.select,
                    per_thread=_args.per_thread,
                    per_rank=_args.per_rank,
                )
            )
        for itr in _args.files:
            if not os.path.exists(itr):
                print("file: {}\nfile-type: {}".format(itr, type(itr).__name__))
            gfs.append(
                load(
                    itr,
                    select=_args.select,
                    per_thread=_args.per_thread,
                    per_rank=_args.per_rank,
                )
            )

        def apply(_input, _mode, *_args, **_kwargs):
            if _mode == "search":
                return search(_input, *_args, **_kwargs)
            elif _mode == "match":
                return match(_input, *_args, **_kwargs)
            elif _mode == "expression":
                return expression(_input, *_args, **_kwargs)
            elif _mode == "sort":
                return sort(_input, *_args, **_kwargs)
            elif _mode == "add":
                return add(_input)
            elif _mode == "subtract":
                return subtract(_input)
            elif _mode == "unify":
                return unify(_input)
            elif _mode == "group":
                return group(_input, *_args, **_kwargs)

        # apply search before match since this is less restrictive
        if _args.search is not None:
            gfs = apply(gfs, "search", pattern=_args.search, field=_args.field)

        # apply match after search since this is more restrictive
        if _args.match is not None:
            gfs = apply(gfs, "match", pattern=_args.match, field=_args.field)

        # apply numerical expression last
        if _args.expression is not None:
            gfs = apply(
                gfs,
                "expression",
                math_expr=_args.expression,
                metric=_args.metric,
            )

        # apply the mutating operations
        if "add" in _args.mode:
            gfs = apply(gfs, "add")
        elif "subtract" in _args.mode:
            gfs = apply(gfs, "subtract")
        elif "unify" in _args.mode:
            gfs = apply(gfs, "unify")

        if _args.sort is not None:
            gfs = apply(
                gfs,
                "sort",
                metric=_args.metric,
                ascending=(_args.sort == "ascending"),
            )

        files = _args.output
        if files is not None and len(files) == 1:
            files = files[0]

        if _args.group is not None:
            gfs = apply(
                gfs,
                "group",
                metric=_args.metric,
                field=_args.group,
                ascending=(_args.sort == "ascending"),
            )
            _args.format = [None]

        for fmt in _args.format:
            dump(gfs, _args.metric, fmt, files, _args.echo_dart)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        import traceback

        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
        print("Exception - {}".format(e))
        if call_exit or _args.exit_on_failure:
            sys.exit(1)
        elif not cmd_line:
            raise
