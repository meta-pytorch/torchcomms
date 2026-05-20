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
import platform
import subprocess as sp

__all__ = [
    "get_shared_lib_ext",
    "get_library_prefixes",
    "find_library_path",
    "get_linked_libraries",
    "echo_dart_measurement",
    "execute",
    "add_preload",
    "post_process",
]

is_darwin = False
is_linux = False
is_windows = False


if platform.system() == "Darwin":
    is_darwin = True
elif platform.system() == "Linux":
    is_linux = True
elif platform.system() == "Windows":
    is_windows = True


def get_shared_lib_ext():
    """
    Get shared library extension
    """
    if is_darwin:
        return ".dylib"
    elif is_linux:
        return ".so"
    elif is_windows:
        return ".dll"
    else:
        return ".so"


def get_library_prefixes():
    """
    Get library search paths
    """
    envvars = []
    if is_darwin:
        envvars += ["DYLD_LIBRARY_PATH"]
    if is_linux:
        envvars += ["LD_LIBRARY_PATH", "LIBRARY_PATH"]

    _prefixes = []
    for envvar in envvars:
        _var = os.environ.get(envvar)
        if _var is not None:
            _prefixes += _var.split(":")
    return _prefixes


def find_library_path(fname):
    _prefixes = get_library_prefixes()
    _lib_ext = get_shared_lib_ext()

    if _lib_ext not in fname:
        return None

    fname = fname.strip("@rpath/")
    if os.path.exists(fname):
        return os.path.realpath(fname)
    else:
        for _prefix in _prefixes:
            _outp = os.path.join(_prefix, fname)
            if os.path.exists(_outp):
                return os.path.realpath(_outp)
            _outp = os.path.join(_prefix, "lib" + fname)
            if os.path.exists(_outp):
                return os.path.realpath(_outp)

    print("Unable to find path to library '{}'".format(fname))
    return None


def get_linked_libraries(exes, libs=[]):
    """
    Get a list of libraries linked to the list of executables
    and (optional) find realpath to libraries
    """
    _linked_libs = []

    for exe in exes:
        output = None
        exe = os.path.realpath(exe)
        if not os.path.exists(exe):
            print("Executable '{}' does not exist".format(exe))
            continue

        if is_darwin:
            result = sp.run(["otool", "-L", "{}".format(exe)], stdout=sp.PIPE)
            output = result.stdout.decode("utf-8").split()
        elif is_linux:
            result = sp.run(["ldd", "{}".format(exe)], stdout=sp.PIPE)
            output = result.stdout.decode("utf-8").split()

        if output is None:
            print("No linked libraries found for '{}'!".format(exe))
            continue

        for out in output:
            _libpath = find_library_path(out)
            if _libpath is not None:
                _linked_libs += [_libpath]

    for _lib in libs:
        _libpath = find_library_path(_lib)
        if _libpath is not None:
            _linked_libs += [_libpath]

    return _linked_libs


def echo_dart_measurement(_name, _path, _type="jpeg"):
    from timemory.common import dart_measurement_file

    dart_measurement_file(_name, _path, _type)


def execute(cmd, outf, keep_going=True, timeout=5 * 60):
    """Execute a command"""
    from timemory.common import popen

    popen(cmd, outf, keep_going, timeout)


def add_preload(libs):
    preload_env = None
    if is_linux:
        preload_env = "LD_PRELOAD"
    elif is_darwin:
        preload_env = "DYLD_INSERT_LIBRARIES"
        os.environ["DYLD_FORCE_FLAT_NAMESPACE"] = "1"
    else:
        return

    if isinstance(libs, list) or isinstance(libs, tuple):
        preload = ":".join(libs)
    elif libs is not None:
        preload = libs
    else:
        return

    if preload_env is not None:
        current_preload = os.environ.get(preload_env)
        if current_preload is not None:
            os.environ[preload_env] = "{}:{}".format(current_preload, preload)
        else:
            os.environ[preload_env] = "{}".format(preload)


def post_process(
    exe,
    fname,
    image_type="jpeg",
    echo_dart=False,
    liblist=[],
    args=["--no_strip_temp", "--functions"],
    generate=["text", "cum", "dot"],
    dot_args=[],
):
    linked = get_linked_libraries(exe)
    libs = []
    if linked is not None:
        for _lib in linked:
            signature = "--add_lib={}".format(_lib)
            if signature not in libs:
                libs += [signature]

    if "text" in generate:
        execute(
            ["google-pprof", "--text"] + libs + args + [exe, fname],
            "{}.txt".format(fname),
        )

    if "cum" in generate:
        execute(
            ["google-pprof", "--text", "--cum"] + libs + args + [exe, fname],
            "{}.cum.txt".format(fname),
        )

    if "dot" in generate:
        oname = "{}.dot".format(fname)
        iname = "{}.{}".format(fname, image_type)
        execute(["google-pprof", "--dot"] + libs + args + [exe, fname], oname)
        execute(
            ["dot"]
            + dot_args
            + ["-T{}".format(image_type), oname, "-o", iname],
            None,
        )
        if echo_dart:
            echo_dart_measurement(
                os.path.basename(iname), os.path.realpath(iname), image_type
            )

    if "callgrind" in generate:
        execute(
            ["google-pprof", "--callgrind"] + libs + args + [exe, fname],
            "{}.callgrind".format(fname),
        )
