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

""" @file plotting/__init__.py
Plotting routines for timemory module
"""

from __future__ import absolute_import
from __future__ import division

import os
import sys
import json
import argparse
import traceback

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

from .plotting import (
    plot_maximums,
    plot,
    plot_all,
    plot_generic,
    read,
    plot_data,
    timemory_data,
    echo_dart_tag,
    add_plotted_files,
    make_output_directory,
    plot_parameters,
    plotted_files,
)

from .table import timem

__all__ = [
    "plotting",
    "plot",
    "plot_maximums",
    "plot_all",
    "plot_generic",
    "read",
    "plot_data",
    "timemory_data",
    "echo_dart_tag",
    "add_plotted_files",
    "make_output_directory",
    "plot_parameters",
    "plotted_files",
    "embedded_plot",
    "table",
    "timem",
]


def embedded_plot(
    _argv, _call_exit=False, _verbose=os.environ.get("TIMEMORY_VERBOSE", 0)
):
    """This is intended to be called from the embedded python interpreter"""
    _call_exit = False
    if _argv is None:
        _argv = sys.argv
        _call_exit = True

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-f", "--files", nargs="*", help="File input", type=str
        )
        parser.add_argument(
            "-d",
            "--display",
            required=False,
            action="store_true",
            help="Display plot",
            dest="display_plot",
        )
        parser.add_argument(
            "-t", "--titles", nargs="*", help="Plot titles", type=str
        )
        parser.add_argument(
            "-c",
            "--combine",
            required=False,
            action="store_true",
            help="Combined data into a single plot",
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            help="Output directory",
            type=str,
            required=False,
        )
        parser.add_argument(
            "-e",
            "--echo-dart",
            help="echo Dart measurement for CDash",
            required=False,
            action="store_true",
        )
        parser.add_argument(
            "--min-percent",
            required=False,
            type=float,
            help="Exclude plotting below this percentage of maximum",
        )
        parser.add_argument(
            "--img-dpi",
            help="Image dots per sq inch",
            required=False,
            type=int,
        )
        parser.add_argument(
            "--img-size",
            help="Image dimensions",
            nargs=2,
            required=False,
            type=int,
        )
        parser.add_argument(
            "--img-type", help="Image type", required=False, type=str
        )
        parser.add_argument(
            "--plot-max",
            help="Plot the maximums from a set of inputs to <filename>",
            required=False,
            type=str,
            dest="plot_max",
        )
        parser.add_argument(
            "--log-x", help="Plot X-axis in a log scale", action="store_true"
        )
        parser.add_argument(
            "--font-size", help="Font size of y-axis labels", type=int
        )
        parser.add_argument(
            "--exit-on-failure",
            help="Call exit if failed",
            action="store_true",
        )
        parser.add_argument(
            "-T",
            "--table",
            help="Generate a table",
            action="store_true",
        )

        parser.set_defaults(display_plot=False)
        parser.set_defaults(combine=False)
        parser.set_defaults(output_dir=".")
        parser.set_defaults(echo_dart=False)
        parser.set_defaults(min_percent=plot_parameters.min_percent)
        parser.set_defaults(img_dpi=plot_parameters.img_dpi)
        parser.set_defaults(
            img_size=[
                plot_parameters.img_size["w"],
                plot_parameters.img_size["h"],
            ]
        )
        parser.set_defaults(img_type=plot_parameters.img_type)
        parser.set_defaults(plot_max="")
        parser.set_defaults(font_size=plot_parameters.font_size)

        args = parser.parse_args(_argv)

        do_plot_max = True if len(args.plot_max) > 0 else False

        params = plot_parameters(
            min_percent=args.min_percent,
            img_dpi=args.img_dpi,
            img_size={"w": args.img_size[0], "h": args.img_size[1]},
            img_type=args.img_type,
            log_xaxis=args.log_x,
            font_size=args.font_size,
        )

        if not args.table:
            if do_plot_max:
                if len(args.titles) != 1:
                    raise Exception("Error must provide one title")
            else:
                if len(args.titles) != 1 and len(args.titles) != len(
                    args.files
                ):
                    raise Exception(
                        "Error must provide one title or a title for each file"
                    )

        data = {}
        for i in range(len(args.files)):
            f = open(args.files[i], "r")
            _jdata = json.load(f)
            _cdata = _jdata["timemory"]

            if "timem" in _cdata:
                if args.table:
                    timem(_cdata, args.files[i], args)
                    continue
                _tdata = _cdata["timem"]
                nranks = len(_tdata)
                concurrency = 1  # always unknown
                _groups = {
                    "time": [
                        "wall_clock",
                        "user_clock",
                        "system_clock",
                        "cpu_clock",
                    ],
                    "utilization": [
                        "cpu_util",
                    ],
                    "memory": [
                        "peak_rss",
                        "page_rss",
                        "virtual_memory",
                    ],
                    "page_faults": [
                        "num_major_page_faults",
                        "num_minor_page_faults",
                    ],
                    "context_switches": [
                        "priority_context_switch",
                        "voluntary_context_switch",
                    ],
                    "io": [
                        "read_char",
                        "read_bytes",
                        "written_char",
                        "written_bytes",
                    ],
                    "hw": [
                        "papi_array",
                    ],
                }

                for cid, itr in _groups.items():
                    data = {}
                    unitr = ""
                    for k in itr:
                        if k in _tdata and "unit_repr" in _tdata[k]:
                            unitr = _tdata[k]["unit_repr"]
                            break

                    _dict = {
                        "cid": cid,
                        "mpi_size": nranks,
                        "concurrency": concurrency,
                        "units": unitr,
                        "ctype": cid,
                        "description": "",
                        "plot_params": params,
                    }

                    for j in range(0, nranks):
                        rdata = _tdata[j]

                        _idata = {}
                        for k in itr:
                            if k in rdata.keys():
                                _idata[k] = rdata[k]
                                if not unitr and "unit_repr" in rdata[k]:
                                    _dict["units"] = rdata[k]["unit_repr"]

                        if len(_idata) == 0:
                            continue
                        _data = read(_idata, **_dict)
                        _rtag = "" if nranks == 1 else "_{}".format(j)
                        _rtitle = (
                            "" if nranks == 1 else " (MPI rank: {})".format(j)
                        )

                        _data.filename = args.files[i].replace(".json", _rtag)
                        _data.filename = "{}_{}".format(_data.filename, cid)
                        if len(args.titles) == 1:
                            _data.title = args.titles[0] + _rtitle
                        else:
                            _data.title = args.titles[i] + _rtitle
                        if j not in data.keys():
                            data[j] = [_data]
                        else:
                            data[j] += [_data]

                    _pargs = {
                        "plot_params": params,
                        "display": args.display_plot,
                        "output_dir": args.output_dir,
                        "echo_dart": args.echo_dart,
                    }

                    if do_plot_max:
                        _pargs["combine"] = args.combine

                    for _rank, _data in data.items():
                        if do_plot_max:
                            plot_maximums(
                                args.plot_max, args.titles[0], _data, **_pargs
                            )
                        else:
                            plot(data=_data, **_pargs)

            else:
                for key, _json in _cdata.items():
                    nranks = (
                        int(_json["num_ranks"]) if "num_ranks" in _json else 1
                    )
                    concurrency = (
                        int(_json["concurrency"])
                        if "concurrency" in _json
                        else 1
                    )
                    ctype = _json["type"]  # collection type
                    cdesc = _json["description"]  # collection description
                    unitr = _json["unit_repr"]  # collection unit display repr
                    cid = " ".join(key.split("_")).title()
                    # roofl = _json["roofline"] if "roofline" in _json else None

                    _dict = {
                        "cid": cid,
                        "mpi_size": nranks,
                        "concurrency": concurrency,
                        "units": unitr,
                        "ctype": ctype,
                        "description": cdesc,
                        "plot_params": params,
                    }

                    rdata = _json["ranks"]
                    for j in range(0, len(rdata)):
                        _data = read(rdata[j], **_dict)
                        _rtag = "" if len(rdata) == 1 else "_{}".format(j)
                        _rtitle = (
                            ""
                            if len(rdata) == 1
                            else " (MPI rank: {})".format(j)
                        )

                        _data.filename = args.files[i].replace(".json", _rtag)
                        if len(args.titles) == 1:
                            _data.title = args.titles[0] + _rtitle
                        else:
                            _data.title = args.titles[i] + _rtitle
                        if j not in data.keys():
                            data[j] = [_data]
                        else:
                            data[j] += [_data]

            _pargs = {
                "plot_params": params,
                "display": args.display_plot,
                "output_dir": args.output_dir,
                "echo_dart": args.echo_dart,
            }

            if do_plot_max:
                _pargs["combine"] = args.combine

            for _rank, _data in data.items():
                if do_plot_max:
                    plot_maximums(
                        args.plot_max, args.titles[0], _data, **_pargs
                    )
                else:
                    plot(data=_data, **_pargs)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print("Exception - {}".format(e))
        if _call_exit or args.exit_on_failure:
            sys.exit(1)
