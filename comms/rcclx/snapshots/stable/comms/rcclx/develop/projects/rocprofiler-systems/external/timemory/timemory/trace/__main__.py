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

from __future__ import absolute_import

""" @file __main__.py
Command line execution for tracer
"""

import os
import sys
import argparse
import traceback

PY3 = sys.version_info[0] == 3


# Python 3.x compatibility utils: execfile
try:
    execfile
except NameError:
    # Python 3.x doesn't have 'execfile' builtin
    import builtins

    exec_ = getattr(builtins, "exec")

    def execfile(filename, globals=None, locals=None):
        with open(filename, "rb") as f:
            exec_(compile(f.read(), filename, "exec"), globals, locals)


def find_script(script_name):
    """Find the script.

    If the input is not a file, then $PATH will be searched.
    """
    if os.path.isfile(script_name):
        return script_name
    path = os.getenv("PATH", os.defpath).split(os.pathsep)
    for dir in path:
        if dir == "":
            continue
        fn = os.path.join(dir, script_name)
        if os.path.isfile(fn):
            return fn

    sys.stderr.write("Could not find script %s\n" % script_name)
    raise SystemExit(1)


def parse_args(args=None):
    """Parse the arguments"""

    if args is None:
        args = sys.argv

    from ..libs.libpytimemory.trace import config as _tracer_config

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-c",
        "--components",
        nargs="+",
        default=["wall_clock"],
        help="List of components",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        default=None,
        type=str,
        help="Output path of results",
    )
    parser.add_argument(
        "-b",
        "--builtin",
        action="store_true",
        help="Put 'trace' in the builtins. Use 'trace.enable()' and "
        "'trace.disable()' in your code to turn it on and off, or "
        "'@trace' to decorate a single function, or 'with trace:' "
        "to trace a single section of code.",
    )
    parser.add_argument(
        "-s",
        "--setup",
        default=None,
        help="Code to execute before the code to trace",
    )
    parser.add_argument(
        "-a",
        "--include-args",
        type=str2bool,
        nargs="?",
        const=True,
        default=_tracer_config.include_args,
        help="Encode the argument values",
    )
    parser.add_argument(
        "-l",
        "--include-line",
        type=str2bool,
        nargs="?",
        const=True,
        default=_tracer_config.include_line,
        help="Encode the function line number",
    )
    parser.add_argument(
        "-f",
        "--include-file",
        type=str2bool,
        nargs="?",
        const=True,
        default=_tracer_config.include_filename,
        help="Encode the function filename",
    )
    parser.add_argument(
        "-F",
        "--full-filepath",
        type=str2bool,
        nargs="?",
        const=True,
        default=_tracer_config.full_filepath,
        help="Encode the full function filename (instead of basename)",
    )
    parser.add_argument(
        "--skip-funcs",
        type=str,
        nargs="+",
        default=_tracer_config.skip_functions,
        help="Filter out any entries with these function names",
    )
    parser.add_argument(
        "--skip-files",
        type=str,
        nargs="+",
        default=_tracer_config.skip_filenames,
        help="Filter out any entries from these files",
    )
    parser.add_argument(
        "--only-funcs",
        type=str,
        nargs="+",
        default=_tracer_config.only_functions,
        help="Select only entries with these function names",
    )
    parser.add_argument(
        "--only-files",
        type=str,
        nargs="+",
        default=_tracer_config.only_filenames,
        help="Select only entries from these files",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=_tracer_config.verbosity,
        help="Logging verbosity",
    )

    return parser.parse_known_args(args)


def get_value(env_var, default_value, dtype, arg=None):
    if arg is not None:
        return dtype(arg)
    else:
        val = os.environ.get(env_var)
        if val is None:
            os.environ[env_var] = "{}".format(default_value)
            return dtype(default_value)
        else:
            return dtype(val)


def run(tracer, cmd):
    if len(cmd) == 0:
        return

    progname = cmd[0]
    sys.path.insert(0, os.path.dirname(progname))
    with open(progname, "rb") as fp:
        code = compile(fp.read(), progname, "exec")

    import __main__

    dict = __main__.__dict__
    print("code: {} {}".format(type(code).__name__, code))
    globs = {
        "__file__": progname,
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
        **dict,
    }

    tracer.runctx(code, globs, None)


def main():
    """Main function"""

    opts = None
    argv = None
    if "--" in sys.argv:
        _idx = sys.argv.index("--")
        _argv = sys.argv[(_idx + 1) :]  # noqa: E203
        opts, argv = parse_args(sys.argv[:_idx])
        argv = _argv
    else:
        if "-h" in sys.argv or "--help" in sys.argv:
            opts, argv = parse_args()
        else:
            argv = sys.argv[1:]
            opts, discard = parse_args()
            if len(argv) == 0 or not os.path.isfile(argv[0]):
                raise RuntimeError(
                    "Could not determine input script. Use '--' before "
                    "the script and its arguments to ensure correct parsing. \nE.g. "
                    "python -m timemory.trace -- ./script.py"
                )

    from ..libs.libpytimemory import initialize
    from ..libs.libpytimemory import settings

    if os.path.isfile(argv[0]):
        argv[0] = os.path.realpath(argv[0])

    settings.output_path = "timemory-{}-trace-output".format(
        os.path.basename(argv[0]).strip().replace("_", "-").replace("--", "-")
    )
    initialize(argv)

    from ..libs.libpytimemory.trace import config as _tracer_config

    _tracer_config.include_args = opts.include_args
    _tracer_config.include_line = opts.include_line
    _tracer_config.include_filename = opts.include_file
    _tracer_config.full_filepath = opts.full_filepath
    _tracer_config.skip_functions = opts.skip_funcs
    _tracer_config.skip_filenames = opts.skip_files
    _tracer_config.only_functions = opts.only_funcs
    _tracer_config.only_filenames = opts.only_files
    _tracer_config.verbosity = opts.verbosity

    print("[timemory]> tracing: {}".format(argv))

    sys.argv[:] = argv
    if opts.setup is not None:
        # Run some setup code outside of the tracer. This is good for large
        # imports.
        setup_file = find_script(opts.setup)
        __file__ = setup_file
        __name__ = "__main__"
        # Make sure the script's directory is on sys.path
        sys.path.insert(0, os.path.dirname(setup_file))
        ns = locals()
        execfile(setup_file, ns, ns)

    from . import Tracer, FakeTracer

    output_path = get_value(
        "TIMEMORY_OUTPUT_PATH", settings.output_path, str, opts.output_dir
    )
    settings.output_path = output_path

    # if len(argv) > 1 and argv[0] != "--"):
    #    raise RuntimeError("Specify command to run with -- <command-to-run>")
    script_file = find_script(sys.argv[0])
    __file__ = script_file
    __name__ = "__main__"
    # Make sure the script's directory is on sys.path
    sys.path.insert(0, os.path.dirname(script_file))

    tracer = Tracer(opts.components)
    fake = FakeTracer()

    if PY3:
        import builtins
    else:
        import __builtin__ as builtins

    builtins.__dict__["trace"] = tracer
    builtins.__dict__["notrace"] = fake
    builtins.__dict__["profile"] = tracer
    builtins.__dict__["noprofile"] = fake

    try:
        try:
            if not opts.builtin:
                tracer.start()
            tracer._unset += 1
            execfile_ = execfile
            ns = locals()
            if opts.builtin:
                execfile(script_file, ns, ns)
            else:
                tracer.runctx(
                    "execfile_(%r, globals())" % (script_file,), ns, ns
                )
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            if not opts.builtin:
                tracer.stop()
            tracer._unset -= 1
            del tracer
            del fake
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
        print("Exception - {}".format(e))


if __name__ == "__main__":
    main()
    from ..libs.libpytimemory import finalize

    finalize()
