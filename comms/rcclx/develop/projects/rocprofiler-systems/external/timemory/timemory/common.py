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

"""
Imports common utilities
"""

from __future__ import absolute_import

__author__ = "Jonathan Madsen"
__copyright__ = "Copyright 2020, The Regents of the University of California"
__credits__ = ["Jonathan Madsen"]
__license__ = "MIT"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

import sys
import os
from os.path import dirname
from os.path import basename
from os.path import join

__all__ = [
    "file",
    "func",
    "line",
    "frame",
    "is_generator",
    "is_coroutine",
    "execute_",
    "PY3",
    "PY35",
    "FILE",
    "FUNC",
    "LINE",
    "FRAME",
    "popen",
    "dart_measurement",
    "dart_measurement_file",
    "write_ctest_notes",
    "get_bin_script",
]

CO_GENERATOR = 0x0020
# Python 2/3 compatibility utils
PY3 = sys.version_info[0] == 3
PY35 = PY3 and sys.version_info[1] >= 5
PY37 = PY3 and sys.version_info[1] >= 7


def frame(back=2):
    """Returns a frame"""
    return sys._getframe(back)


def file(back=2, only_basename=True, use_dirname=False, noquotes=True):
    """
    Returns the file name
    """

    def get_fcode(back):
        fname = "<module>"
        try:
            fname = sys._getframe(back).f_code.co_filename
        except Exception as e:
            print(e)
            fname = "<module>"
        return fname

    result = None
    if only_basename is True:
        if use_dirname is True:
            result = "{}".format(
                join(
                    basename(dirname(get_fcode(back))),
                    basename(get_fcode(back)),
                )
            )
        else:
            result = "{}".format(basename(get_fcode(back)))
    else:
        result = "{}".format(get_fcode(back))

    if noquotes is False:
        result = "'{}'".format(result)

    return result


def func(back=2):
    """
    Returns the function name
    """
    return "{}".format(sys._getframe(back).f_code.co_name)


def line(back=1):
    """
    Returns the line number
    """
    return int(sys._getframe(back).f_lineno)


FRAME = frame
FILE = file
FUNC = func
LINE = line


def is_generator(f):
    """Return True if a function is a generator."""
    isgen = (f.__code__.co_flags & CO_GENERATOR) != 0
    return isgen


# exec (from https://bitbucket.org/gutworth/six/):
if PY3:
    import builtins

    execute_ = getattr(builtins, "exec")
    del builtins
else:

    def execute_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")


if PY35:
    import inspect

    def is_coroutine(f):
        return inspect.iscoroutinefunction(f)

else:

    def is_coroutine(f):
        return False


def popen(cmd, outf=None, keep_going=True, timeout=None, shell=False):
    """Execute a command and write to a file or get the output"""

    def handle_error(ret, cmd, keep_going):
        """Handle error messaging for executed command"""
        err_msg = "Error executing: '{}'".format(" ".join(cmd))
        if ret != 0 and not keep_going:
            raise RuntimeError(err_msg)
        elif ret != 0 and keep_going:
            barrier = "=" * 80
            err_msg = (
                "\n\n"
                + barrier
                + "\n\n    ERROR: "
                + err_msg
                + "\n\n"
                + barrier
                + "\n\n"
            )
            sys.stderr.write(err_msg)
            sys.stderr.flush()

    import subprocess as sp

    p = None
    outs = None
    errs = None
    retc = 0
    if shell:
        if PY37:
            p = sp.run(
                " ".join(cmd),
                shell=True,
                check=False,
                capture_output=True,
            )
        else:
            p = sp.run(
                " ".join(cmd),
                shell=True,
                check=False,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )
    else:
        p = sp.Popen(cmd)
        try:
            if timeout is not None:
                outs, errs = p.communicate(timeout=timeout)
            else:
                outs, errs = p.communicate()
        except sp.TimeoutExpired:
            p.kill()

    retc = p.returncode
    if retc != 0:
        if errs is not None:
            print("{}".format(errs.decode("utf-8")))
        elif p.stderr is not None:
            print("{}".format(p.stderr.decode("utf-8")))

    handle_error(retc, cmd, keep_going)

    if outf is not None:
        with open(outf, "w") as f:
            if outs is not None:
                f.write(outs)
            elif p.stdout is not None:
                f.write(p.stdout)

    if outs is None and p.stdout is not None:
        outs = "{}".format(p.stdout.decode("utf-8"))

    if errs is None and p.stderr is not None:
        errs = "{}".format(p.stderr.decode("utf-8"))

    return (retc, outs, errs)


def which(exe):
    """Portable implementation of which"""
    try:
        import shutil

        return shutil.which(exe)
    except (ImportError, AttributeError):
        try:
            from distutils.spawn import find_executable

            return find_executable(exe)
        except ImportError:
            import os

            _path = os.getenv("PATH")
            for p in _path.split(os.path.pathsep):
                p = os.path.join(p, exe)
                if os.path.exists(p) and os.access(p, os.X_OK):
                    return p

    return None


def dart_measurement(name, value):
    """Prints out an XML tag which gets detected by CTest and recorded by CDash"""
    print(
        f'<DartMeasurement name="{name}" type="numeric/double">{value}</DartMeasurement>'
    )


def dart_measurement_file(name, path, format="png", type="image"):
    """Prints out an XML tag which gets detected by CTest and uploaded to CDash"""
    _path = os.path.abspath(path)
    tag = "DartMeasurementFile"
    print(f'<{tag} name="{name}" type="{type}/{format}">{_path}</{tag}>')


def write_ctest_notes(fname, path=None, mode="a"):
    """Writes or appends to a CTestNotes.txt file"""

    if path is None:
        path = os.path.abspath(os.path.dirname(fname))
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.abspath(fname)
    with open(os.path.join(path, "CTestNotes.txt"), "a") as ofs:
        ofs.write(f'\nlist(APPEND CTEST_NOTES_FILES "{fname}")\n')


def get_bin_script(fname):
    """Returns the path to a script in the internal bin folder"""
    _this_dir = os.path.dirname(__file__)
    _fpath = os.path.join(_this_dir, "bin", fname)
    if os.path.exists(_fpath):
        return _fpath
    import glob

    for itr in glob.glob(f"{_fpath}*"):
        if os.path.exists(itr):
            return itr
    return None
