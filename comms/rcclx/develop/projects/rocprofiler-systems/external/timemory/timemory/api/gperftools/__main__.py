#!@PYTHON_EXECUTABLE@
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
#

""" @file __main__.py
Command line execution for gperftools
"""

import sys
import argparse
import warnings
import traceback

from . import cpu_profiler as _cpu
from . import utils as _utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--echo-dart",
        action="store_true",
        help="Echo dart measurement files",
    )
    parser.add_argument(
        "-E", "--exe", action="store_true", help="Echo dart measurement files"
    )
    parser.add_argument(
        "-t",
        "--type",
        help="gperftools type",
        choices=("cpu_profiler", "heap_profiler"),
        default="cpu_profiler",
        type=str,
    )
    parser.add_argument(
        "--img-type",
        help="Image format",
        default="jpeg",
        type=str,
        choices=("png", "jpeg", "tif"),
    )
    parser.add_argument(
        "-f", "--frequency", help="sampling frequency", default=200, type=int
    )
    parser.add_argument(
        "-m", "--malloc-stats", help="malloc stats", default=0, type=int
    )
    parser.add_argument(
        "-r", "--realtime", help="use realtime", default=1, type=int
    )
    parser.add_argument(
        "-s", "--selected", help="profile selected", default=0, type=int
    )
    parser.add_argument(
        "-o", "--output-prefix", type=str, help="Output prefix", default=None
    )
    parser.add_argument(
        "-l", "--libs", type=str, help="--add_libs arguments", default=[]
    )
    parser.add_argument(
        "-g",
        "--generate",
        type=str,
        help="Generated output types",
        nargs="+",
        default=["text", "cum", "dot"],
        choices=("text", "cum", "dot", "callgrind"),
    )
    parser.add_argument(
        "-a",
        "--args",
        type=str,
        help="Arguments to google-pprof",
        nargs="*",
        default=["--no_strip_temp", "--functions"],
    )
    parser.add_argument(
        "-d",
        "--dot-args",
        type=str,
        help="Arguments to dot",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "-p",
        "--preload",
        help="Enable preloading gperftools library",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input file for existing tool data",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # if args.help:
    #    parser.print_help()
    #    sys.exit(1)

    if args.output_prefix is None:
        args.output_prefix = "{}.prof".format(args.type.split("_")[0])

    for lib in args.libs:
        _libpath = _utils.find_library_path(lib)
        if _libpath is not None:
            lib = "--add_lib={}".format(_libpath)
        else:
            lib = "--add_lib={}".format(lib)

    return args


def post_process(args):
    try:
        if args.type == "cpu_profiler":
            if args.input is None:
                raise RuntimeError("No input file provided for post-processing")

            _utils.post_process(
                args.exe,
                args.input,
                args.img_type,
                args.echo_data,
                args.libs,
                args.args,
                args.generate,
                args.dot_args,
            )

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print("Exception - {}".format(e))
        sys.exit(1)


def run(args, cmd):
    try:
        if args.type == "cpu_profiler":
            _cpu.execute(
                cmd,
                args.output_prefix,
                args.frequency,
                args.malloc_stats,
                args.realtime,
                args.preload,
                args.selected,
                args.img_type,
                args.echo_dart,
                args.libs,
                args.args,
                args.generate,
                args.dot_args,
            )

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print("Exception - {}".format(e))
        sys.exit(1)


if __name__ == "__main__":
    try:
        # look for "--" and interpret anything after that
        # to be a command to execute
        _argv = []
        _cmd = []

        _argsets = [_argv, _cmd]
        _i = 0
        _separator = "--"

        for _arg in sys.argv[1:]:
            if _arg == _separator and _i < len(_argsets):
                _i += 1
            else:
                _argsets[_i].append(_arg)

        print("CMD: {}, ARGS: {}".format(_cmd, _argv))
        sys.argv[1:] = _argv
        args = parse_args()
        if len(_cmd) == 0:
            post_process(args)
        else:
            run(args, _cmd)

    except Exception as e:
        print(f"\nException :: command line argument error\n\t{e}\n")
        # warnings.warn(msg)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=20)
        warnings.warn("timemory.gperftools is disabled")
        sys.exit(1)
