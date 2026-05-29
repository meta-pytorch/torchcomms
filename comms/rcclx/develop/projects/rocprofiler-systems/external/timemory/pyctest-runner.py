#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for timemory
"""

import os
import re
import sys
import platform
import traceback
import warnings
import multiprocessing as mp
import pyctest.pyctest as pyct
import pyctest.pycmake as pycm
import pyctest.helpers as helpers
from collections import OrderedDict

clobber_notes = True
available_tools = {
    "avail": "TIMEMORY_BUILD_AVAIL",
    "timem": "TIMEMORY_BUILD_TIMEM",
    "kokkos": "TIMEMORY_BUILD_KOKKOS_TOOLS",
    "kokkos-config": "TIMEMORY_BUILD_KOKKOS_CONFIG",
    "dyninst": "TIMEMORY_BUILD_DYNINST_TOOLS",
    "mpip": "TIMEMORY_BUILD_MPIP_LIBRARY",
    "ompt": "TIMEMORY_BUILD_OMPT_LIBRARY",
    "ncclp": ["TIMEMORY_BUILD_NCCLP_LIBRARY", "TIMEMORY_USE_NCCL"],
    "mallocp": "TIMEMORY_BUILD_MALLOCP_LIBRARY",
    "compiler": "TIMEMORY_BUILD_COMPILER_INSTRUMENTATION",
    "ert": "TIMEMORY_BUILD_ERT",
}
argparse_defaults = {}
build_name = ""


def get_branch(wd=pyct.SOURCE_DIRECTORY):
    # handle pull-request
    prname = None
    if os.environ.get("CIRCLE_PULL_REQUEST", None) is not None:
        prname = "pr"
    prname = os.environ.get("CIRCLE_PR_REPONAME", prname)

    if prname is None:
        if os.environ.get("TRAVIS_EVENT_TYPE", "").lower() == "pull_request":
            prname = os.environ.get("TRAVIS_PULL_REQUEST_SLUG", "pr").replace(
                "/", "-"
            )

    # handle env specified
    for env_var in ["CIRCLE_BRANCH", "TRAVIS_BRANCH"]:
        env_branch = os.environ.get(env_var, None)
        if env_branch is not None:
            if prname is not None:
                return "{}-{}".format(prname, env_branch)
            return env_branch

    cmd = pyct.command(["git", "show", "-s", "--pretty=%d", "HEAD"])
    cmd.SetOutputStripTrailingWhitespace(True)
    cmd.SetWorkingDirectory(wd)
    cmd.Execute()
    branch = cmd.Output()
    branch = branch.split(" ")
    if branch:
        branch = branch[len(branch) - 1]
        branch = branch.strip(")")
    if not branch:
        branch = pyct.GetGitBranch(wd)

    return branch


def install_compile_time_perf(_dir):
    import tempfile

    if os.path.exists(os.path.join(_dir, "bin", "timem")):
        return

    source_dir = tempfile.mkdtemp()

    def run_cmd(_cmd):
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
        cmd = pyct.command(_cmd)
        cmd.SetWorkingDirectory(source_dir)
        cmd.SetOutputQuiet(False)
        cmd.SetErrorQuiet(False)
        cmd.Execute()
        if int(cmd.Result()) > 0:
            print("output message : {}".format(cmd.Output()))
            print("error message  : {}".format(cmd.Error()))
            print(
                "command failed with errc {}: {}".format(
                    cmd.Result(), " ".join(_cmd)
                )
            )
            raise RuntimeError("command error")

    git_cmd = helpers.FindExePath("git")
    cmake_cmd = helpers.FindExePath("cmake")

    if (git_cmd, cmake_cmd) is None:
        return

    run_cmd(
        [
            git_cmd,
            "clone",
            "https://github.com/jrmadsen/compile-time-perf.git",
        ]
    )
    run_cmd(
        [
            cmake_cmd,
            "-B",
            "build-ctp",
            "-D",
            f"CMAKE_INSTALL_PREFIX={_dir}",
            "-D",
            "TIMEMORY_USE_PAPI=OFF",
            "compile-time-perf",
        ]
    )
    run_cmd([cmake_cmd, "--build", "build-ctp", "--target", "all"])
    run_cmd([cmake_cmd, "--build", "build-ctp", "--target", "install"])

    os.environ["CMAKE_PREFIX_PATH"] = ":".join(
        [_dir, os.environ.get("CMAKE_PREFIX_PATH", "")]
    )


def configure():
    # Get pyctest argument parser that include PyCTest arguments
    parser = helpers.ArgumentParser(
        project_name="timemory",
        source_dir=os.getcwd(),
        binary_dir=os.path.join(
            os.getcwd(), "build-timemory", platform.system()
        ),
        build_type="Release",
        vcs_type="git",
        use_launchers=False,
    )

    parser.add_argument(
        "--quiet",
        help="Disable reporting memory usage",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--arch",
        help="TIMEMORY_USE_ARCH=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--profile",
        help="Run gperf profiler",
        default=None,
        type=str,
        choices=("cpu", "heap"),
    )
    parser.add_argument(
        "--sanitizer",
        help="Type of sanitizer",
        default=None,
        type=str,
        choices=("leak", "memory", "address", "thread"),
    )
    parser.add_argument(
        "--coverage",
        help="TIMEMORY_USE_COVERAGE=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--static-analysis",
        help="TIMEMORY_USE_CLANG_TIDY=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--tools",
        help="TIMEMORY_BUILD_TOOLS=ON",
        default=[],
        nargs="*",
        choices=available_tools.keys(),
    )
    parser.add_argument(
        "--tau", help="TIMEMORY_USE_TAU=ON", default=False, action="store_true"
    )
    parser.add_argument(
        "--cuda",
        help="TIMEMORY_USE_CUDA=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--nvtx",
        help="TIMEMORY_USE_NVTX=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--cupti",
        help="TIMEMORY_USE_CUPTI=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--hip",
        help="TIMEMORY_USE_HIP=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--upcxx",
        help="TIMEMORY_USE_UPCXX=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gotcha",
        help="TIMEMORY_USE_GOTCHA=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gperftools",
        help="TIMEMORY_USE_GPERFTOOLS=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--caliper",
        help="TIMEMORY_USE_CALIPER=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--likwid",
        help="TIMEMORY_USE_LIKWID=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--papi",
        help="TIMEMORY_USE_PAPI=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--mpi", help="TIMEMORY_USE_MPI=ON", default=False, action="store_true"
    )
    parser.add_argument(
        "--mpi-init",
        help="TIMEMORY_USE_MPI_INIT=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--python",
        help="TIMEMORY_BUILD_PYTHON=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--build-ompt",
        help="TIMEMORY_BUILD_OMPT=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--extra-optimizations",
        help="TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--lto",
        help="TIMEMORY_BUILD_LTO=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ipo",
        help="CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--developer",
        help="TIMEMORY_BUILD_DEVELOPER=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--xray",
        help="TIMEMORY_BUILD_XRAY=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--stats",
        help="TIMEMORY_USE_STATISTICS=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--timing",
        help="TIMEMORY_USE_COMPILE_TIMING=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--unwind",
        help="TIMEMORY_USE_LIBUNWIND=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--plot",
        help="TIMEMORY_PLOT_OUTPUT=ON during tests",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dart",
        help="TIMEMORY_DART_OUTPUT=ON during tests",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--build-libs",
        help="Build library type(s)",
        default=["shared"],
        nargs="*",
        type=str,
        choices=("static", "shared"),
    )
    parser.add_argument(
        "--tls-model",
        help="Thread-local static model",
        default=("global-dynamic"),
        type=str,
        choices=(
            "global-dynamic",
            "local-dynamic",
            "initial-exec",
            "local-exec",
        ),
    )
    parser.add_argument(
        "--cxx-standard",
        help="C++ standard",
        type=str,
        default="17",
        choices=("17", "20", "23"),
    )
    parser.add_argument(
        "--generate", help="Generate the tests only", action="store_true"
    )
    parser.add_argument(
        "-j",
        "--cpu-count",
        type=int,
        default=mp.cpu_count(),
        help="Parallel build jobs to run",
    )
    parser.add_argument(
        "--quick",
        help="Only build the library",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--minimal",
        help="Only build unit tests (not examples)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--compile-time-perf",
        help="Build and install compile-time-perf in the given directory",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    if "kokkos-config" in args.tools and "kokkos" not in args.tools:
        args.tools.append("kokkos")

    if "shared" not in args.build_libs and args.python:
        raise RuntimeError("Python cannot be built with static libraries")

    if os.environ.get("CTEST_SITE") is not None:
        pyct.set("CTEST_SITE", "{}".format(os.environ.get("CTEST_SITE")))

    if os.path.exists(os.path.join(pyct.BINARY_DIRECTORY, "CMakeCache.txt")):
        from pyctest import cmake_executable as cm
        from pyctest import version_info as _pyctest_version

        if (
            _pyctest_version[0] == 0
            and _pyctest_version[1] == 0
            and _pyctest_version[2] < 11
        ):
            cmd = pyct.command(
                [cm, "--build", pyct.BINARY_DIRECTORY, "--target", "clean"]
            )
            cmd.SetWorkingDirectory(pyct.BINARY_DIRECTORY)
            cmd.SetOutputQuiet(True)
            cmd.SetErrorQuiet(True)
            cmd.Execute()
        else:
            from pyctest.cmake import CMake

            CMake("--build", pyct.BINARY_DIRECTORY, "--target", "clean")
        helpers.RemovePath(
            os.path.join(pyct.BINARY_DIRECTORY, "CMakeCache.txt")
        )

    if platform.system() != "Linux":
        args.papi = False

    os.environ["PYCTEST_TESTING"] = "ON"
    os.environ["TIMEMORY_BANNER"] = "OFF"
    os.environ["TIMEMORY_CTEST_NOTES"] = "ON"
    os.environ["TIMEMORY_ENABLE_SIGNAL_HANDLER"] = "ON"
    # os.environ["TIMEMORY_PLOT_OUTPUT"] = "OFF"

    # update PYTHONPATH for the unit tests
    pypath = os.environ.get("PYTHONPATH", "").split(":")
    pypath = [pyct.BINARY_DIRECTORY] + pypath
    os.environ["PYTHONPATH"] = ":".join(pypath)

    global build_name
    global argparse_defaults
    # generate the defaults
    argparse_defaults = {key: parser.get_default(key) for key in vars(args)}
    # sort them for consistency
    argparse_defaults = OrderedDict(sorted(argparse_defaults.items()))
    # construct a build name from the arguments that were changed
    for key, itr in argparse_defaults.items():
        # ignore all pyctest args except the build type
        if "pyctest_" in key and key != "pyctest_build_type":
            continue
        if "quiet" in key:
            continue
        if "compile_time_perf" in key:
            continue
        # get the current value
        curr = args.__getattribute__(key)
        # if the value is true
        if curr != itr:
            if isinstance(curr, bool) and curr:
                # just add name of boolean options
                build_name = "-".join([build_name, key[:4]])
            elif isinstance(curr, list):
                # if list, join the args
                build_name = "-".join(
                    [
                        build_name,
                        "-".join(["{}".format(val[:6]) for val in curr]),
                    ]
                )
            elif isinstance(curr, str):
                # if string, just abbreviated name
                build_name = "-".join([build_name, curr[:6]])
            elif key == "cxx_standard":
                # ignore all else except for the C++ standard
                build_name = "-".join([build_name, f"cxx{itr}"])

    build_name = "-".join(sorted(build_name.strip("-").split("-"))).replace(
        "kokkos-kokkos", "kokkos-config"
    )

    if args.compile_time_perf is not None:
        try:
            install_compile_time_perf(args.compile_time_perf)
        except RuntimeError:
            pass

    return args


def run_pyctest():
    # run argparse, checkout source, copy over files
    #
    args = configure()

    google_pprof = helpers.FindExePath("google-pprof")
    if google_pprof is None:
        google_pprof = helpers.FindExePath("pprof")

    # find srun and mpirun
    #
    dmprun = None
    dmpargs = ["-n", "2"]
    for dmpexe in ("srun", "jsrun", "mpirun"):
        try:
            dmprun = helpers.FindExePath(dmpexe)
            if dmprun is not None and os.path.isabs(dmprun):
                if dmpexe == "srun":
                    dmpargs += ["-c", "1"]
                elif dmpexe == "jsrun":
                    dmpargs += ["-c", "1"]
                break
        except Exception as e:
            print("Exception: {}".format(e))

    # Compiler version
    #
    if os.environ.get("CXX") is None:
        os.environ["CXX"] = helpers.FindExePath("c++")
    cmd = pyct.command([os.environ["CXX"], "--version"])
    cmd.SetOutputStripTrailingWhitespace(True)
    cmd.Execute()
    compiler_version = cmd.Output().replace("Ubuntu", "").lower()
    cn = os.environ["CXX"]
    try:
        cn = compiler_version.split()[0]
        cv = re.search(r"(\b)\d+.\d", compiler_version)
        compiler_version = "{}-{}".format(cn, cv.group()).replace("++", "xx")
    except Exception as e:
        print("Exception! {}".format(e))
        cmd = pyct.command([os.environ["CXX"], "-dumpversion"])
        cmd.SetOutputStripTrailingWhitespace(True)
        cmd.Execute()
        compiler_version = "{}{}".format(cn, cmd.Output()).replace("++", "xx")

    # Set the build name
    #
    pyct.BUILD_NAME = (
        "{}-{}-{}".format(
            get_branch(pyct.SOURCE_DIRECTORY),
            platform.uname()[0],
            compiler_version,
        )
        .replace("/", "-")
        .replace(" ", "-")
        .replace("--", "-")
    )

    #   build specifications
    #
    build_opts = {
        "BUILD_SHARED_LIBS": "ON" if "shared" in args.build_libs else "OFF",
        "BUILD_STATIC_LIBS": "ON" if "static" in args.build_libs else "OFF",
        "CMAKE_INTERPROCEDURAL_OPTIMIZATION": "ON" if args.ipo else "OFF",
        "CMAKE_CXX_STANDARD": "{}".format(args.cxx_standard),
        "TIMEMORY_CI": "ON",
        "TIMEMORY_TLS_MODEL": "{}".format(args.tls_model),
        "TIMEMORY_CCACHE_BUILD": "OFF",
        "TIMEMORY_BUILD_C": "ON",
        "TIMEMORY_BUILD_LTO": "ON" if args.lto else "OFF",
        "TIMEMORY_BUILD_OMPT": "ON" if args.build_ompt else "OFF",
        "TIMEMORY_BUILD_TOOLS": "ON" if len(args.tools) > 0 else "OFF",
        "TIMEMORY_BUILD_GOTCHA": "ON" if args.gotcha else "OFF",
        "TIMEMORY_BUILD_PYTHON": "ON" if args.python else "OFF",
        "TIMEMORY_BUILD_CALIPER": "ON" if args.caliper else "OFF",
        "TIMEMORY_BUILD_LIBUNWIND": "ON" if args.unwind else "OFF",
        "TIMEMORY_BUILD_DEVELOPER": "ON" if args.developer else "OFF",
        "TIMEMORY_BUILD_TESTING": "ON" if not args.quick else "OFF",
        "TIMEMORY_BUILD_EXAMPLES": (
            "OFF" if args.quick or args.minimal else "ON"
        ),
        "TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS": (
            "ON" if args.extra_optimizations else "OFF"
        ),
        "TIMEMORY_USE_CTP": (
            "ON" if args.compile_time_perf is not None else "OFF"
        ),
        "TIMEMORY_USE_HIP": "ON" if args.hip else "OFF",
        "TIMEMORY_USE_MPI": "ON" if args.mpi else "OFF",
        "TIMEMORY_USE_TAU": "ON" if args.tau else "OFF",
        "TIMEMORY_USE_ARCH": "ON" if args.arch else "OFF",
        "TIMEMORY_USE_PAPI": "ON" if args.papi else "OFF",
        "TIMEMORY_USE_CUDA": "ON" if args.cuda else "OFF",
        "TIMEMORY_USE_NVTX": "ON" if args.nvtx else "OFF",
        "TIMEMORY_USE_OMPT": "ON" if "ompt" in args.tools else "OFF",
        "TIMEMORY_USE_XRAY": "ON" if args.xray else "OFF",
        "TIMEMORY_USE_CUPTI": "ON" if args.cupti else "OFF",
        "TIMEMORY_USE_UPCXX": "ON" if args.upcxx else "OFF",
        "TIMEMORY_USE_LIKWID": "ON" if args.likwid else "OFF",
        "TIMEMORY_USE_GOTCHA": "ON" if args.gotcha else "OFF",
        "TIMEMORY_USE_PYTHON": "ON" if args.python else "OFF",
        "TIMEMORY_USE_CALIPER": "ON" if args.caliper else "OFF",
        "TIMEMORY_USE_COVERAGE": "ON" if args.coverage else "OFF",
        "TIMEMORY_USE_GPERFTOOLS": "ON" if args.gperftools else "OFF",
        "TIMEMORY_USE_STATISTICS": "ON" if args.stats else "OFF",
        "TIMEMORY_USE_LIBUNWIND": "ON" if args.unwind else "OFF",
        "TIMEMORY_USE_COMPILE_TIMING": "ON" if args.timing else "OFF",
        "TIMEMORY_USE_SANITIZER": "OFF",
        "TIMEMORY_USE_CLANG_TIDY": "ON" if args.static_analysis else "OFF",
    }

    if args.minimal:
        build_opts["TIMEMORY_BUILD_MINIMAL_TESTING"] = "ON"
        build_opts["TIMEMORY_BUILD_EXAMPLES"] = "OFF"

    if args.papi:
        build_opts["USE_PAPI"] = "ON"

    if args.caliper:
        build_opts["USE_CALIPER"] = "ON"

    if args.mpi:
        build_opts["USE_MPI"] = "ON"

    if args.mpi and args.mpi_init:
        build_opts["TIMEMORY_USE_MPI_INIT"] = "ON"

    if args.build_ompt:
        build_opts["OPENMP_ENABLE_LIBOMPTARGET"] = "OFF"

    if "avail" not in args.tools:
        args.tools.append("avail")

    for key, opt in available_tools.items():
        if isinstance(opt, list):
            for itr in opt:
                build_opts[itr] = "ON" if (key in args.tools) else "OFF"
        else:
            build_opts[opt] = "ON" if (key in args.tools) else "OFF"

    if "dyninst" in args.tools:
        build_opts["TIMEMORY_USE_DYNINST"] = "ON"

    if args.python:
        pyver = "{}{}".format(
            sys.version_info[0],
            sys.version_info[1],
        )
        pyct.BUILD_NAME = "{}-py{}".format(pyct.BUILD_NAME, pyver)

    if args.profile is not None:
        build_opts["TIMEMORY_USE_GPERFTOOLS"] = "ON"
        components = "profiler" if args.profile == "cpu" else "tcmalloc"
        build_opts["TIMEMORY_gperftools_COMPONENTS"] = components

    if args.sanitizer is not None:
        build_opts["SANITIZER_TYPE"] = args.sanitizer
        build_opts["TIMEMORY_USE_SANITIZER"] = "ON"

    if args.coverage:
        gcov_exe = helpers.FindExePath("gcov")
        if gcov_exe is not None:
            pyct.COVERAGE_COMMAND = "{}".format(gcov_exe)
            build_opts["TIMEMORY_USE_COVERAGE"] = "ON"
            if pyct.BUILD_TYPE != "Debug":
                warnings.warn(
                    "Forcing build type to 'Debug' when coverage is enabled"
                )
                pyct.BUILD_TYPE = "Debug"
        else:
            build_opts["TIMEMORY_USE_COVERAGE"] = "OFF"

    pyct.set(
        "CTEST_CUSTOM_COVERAGE_EXCLUDE",
        ";".join(
            [
                "/usr/.*",
                ".*external/.*",
                ".*examples/.*",
                ".*source/tests/.*",
                ".*source/tools/.*",
                ".*source/python/.*",
                ".*source/timemory/tpls/.*",
                ".*/signals.hpp",
                ".*/popen.cpp",
            ]
        ),
    )
    pyct.set("CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS", "100")
    pyct.set("CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS", "100")
    pyct.set("CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE", "10240")

    # Use the options to create a build name with configuration
    pyct.BUILD_NAME = (
        (
            "{}-{}".format(pyct.BUILD_NAME, build_name)
            .replace("/", "-")
            .replace(" ", "-")
        )
        .strip("-")
        .replace("origin-", "")
    )

    # default options
    cmake_args = "-DCMAKE_BUILD_TYPE={}".format(pyct.BUILD_TYPE)

    # customized from args
    for key, val in build_opts.items():
        cmake_args = "{} -D{}={}".format(cmake_args, key, val)

    cmake_args = "-DPYTHON_EXECUTABLE={} {} {}".format(
        sys.executable, cmake_args, " ".join(pycm.ARGUMENTS)
    )

    # how to build the code
    #
    ctest_cmake_cmd = "${CTEST_CMAKE_COMMAND}"
    pyct.CONFIGURE_COMMAND = "{} {} {}".format(
        ctest_cmake_cmd, cmake_args, pyct.SOURCE_DIRECTORY
    )

    # how to build the code
    #
    pyct.BUILD_COMMAND = (
        "{} --build {} --target all --parallel {} --verbose".format(
            ctest_cmake_cmd,
            pyct.BINARY_DIRECTORY,
            args.cpu_count,
        )
    )

    # Windows build command extras
    #
    if platform.system() == "Windows":
        pyct.BUILD_COMMAND = "{} -- /MP -A x64".format(pyct.BUILD_COMMAND)

    # how to update the code
    #
    git_exe = helpers.FindExePath("git")
    pyct.UPDATE_COMMAND = "{}".format(git_exe)
    pyct.set("CTEST_UPDATE_TYPE", "git")
    pyct.set("CTEST_GIT_COMMAND", "{}".format(git_exe))

    # find the CTEST_TOKEN_FILE
    #
    if args.pyctest_token_file is None and args.pyctest_token is None:
        home = helpers.GetHomePath()
        if home is not None:
            token_path = os.path.join(
                home, os.path.join(".tokens", "nersc-cdash")
            )
            if os.path.exists(token_path):
                pyct.set("CTEST_TOKEN_FILE", token_path)

    # construct a command
    #
    def construct_name(test_name):
        return test_name.replace("_", "-")

    # construct a command
    #
    def construct_command(cmd, args):
        global clobber_notes
        _cmd = []
        if args.profile is not None and google_pprof is not None:
            _exe = os.path.basename(cmd[0])
            if args.profile == "cpu":
                _cmd.append(
                    os.path.join(pyct.BINARY_DIRECTORY, "gperf-cpu-profile.sh")
                )
                pyct.add_note(
                    pyct.BINARY_DIRECTORY,
                    "cpu.prof.{}/gperf.0.txt".format(_exe),
                    clobber=clobber_notes,
                )
                pyct.add_note(
                    pyct.BINARY_DIRECTORY,
                    "cpu.prof.{}/gperf.0.cum.txt".format(_exe),
                    clobber=False,
                )
                clobber_notes = False
            elif args.profile == "heap":
                _cmd.append(
                    os.path.join(pyct.BINARY_DIRECTORY, "gperf-heap-profile.sh")
                )
                for itr in [
                    "alloc_objects",
                    "alloc_space",
                    "inuse_objects",
                    "inuse_space",
                ]:
                    pyct.add_note(
                        pyct.BINARY_DIRECTORY,
                        "heap.prof.{}/gperf.0.0001.heap.{}.txt".format(
                            _exe, itr
                        ),
                        clobber=clobber_notes,
                    )
                    # make sure all subsequent iterations don't clobber
                    clobber_notes = False
        _cmd.extend(cmd)
        return _cmd

    # construct a command
    #
    def construct_roofline_command(cmd, dir, extra_opts=[], use_mpi=True):
        _cmd = [
            sys.executable,
            "-m",
            "timemory.roofline",
            "-e",
            "-D",
            dir,
            "--format",
            "png",
        ]
        _cmd.extend(extra_opts)
        _cmd.extend(["--"])
        if use_mpi and args.mpi and dmprun is not None:
            _cmd += [dmprun] + dmpargs
        _cmd.extend(cmd)
        return _cmd

    # testing environ
    #
    pypath = ":".join(
        ["{}".format(pyct.BINARY_DIRECTORY), os.environ.get("PYTHONPATH", "")]
    )
    base_env = ";".join(
        [
            "CPUPROFILE_FREQUENCY=200",
            "CPUPROFILE_REALTIME=1",
            "CALI_CONFIG_PROFILE=runtime-report",
            "TIMEMORY_ENABLE_SIGNAL_HANDLER=ON",
            "PYTHONPATH={}".format(pypath),
            "TIMEMORY_PLOT_OUTPUT={}".format("ON" if args.plot else "OFF"),
        ]
    )
    if args.coverage or pyct.BUILD_TYPE == "Debug":
        base_env = ";".join(
            [base_env, "TIMEMORY_DEBUG=ON", "TIMEMORY_VERBOSE=6"]
        )
    test_env = (
        ";".join([base_env, "TIMEMORY_DART_OUTPUT=ON", "TIMEMORY_DART_COUNT=1"])
        if args.dart
        else base_env[:]
    )

    # create tests
    #
    if "avail" in args.tools:
        pyct.test(
            "timemory-avail",
            ["./timemory-avail", "-a"],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "30",
                "ENVIRONMENT": test_env,
            },
        )

    if "timem" in args.tools:

        def add_timem_test(name, cmd):
            if len(cmd) > 1:
                cmd.append("--")
            cmd.append("sleep")

            pyct.test(
                "{}-zero".format(name),
                cmd + ["0"],
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "10",
                    "ENVIRONMENT": base_env,
                },
            )

            pyct.test(
                name,
                cmd + ["2"],
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "10",
                    "ENVIRONMENT": base_env,
                },
            )

        add_timem_test("timemory-timem", ["./timem"])

        add_timem_test(
            "timemory-timem-shell", ["./timem", "-s", "-v", "2", "--debug"]
        )

        add_timem_test(
            "timemory-timem-no-sample",
            ["./timem", "--disable-sample"],
        )

        add_timem_test(
            "timemory-timem-json",
            ["./timem", "-o", "timem-output"],
        )

        if args.mpi and dmprun is not None:
            add_timem_test(
                "timemory-timem-mpi",
                [dmprun] + dmpargs + ["./timem-mpi"],
            )

            add_timem_test(
                "timemory-timem-mpi-shell",
                [dmprun] + dmpargs + ["./timem-mpi", "-s"],
            )

            add_timem_test(
                "timemory-timem-mpi-individual",
                [dmprun] + dmpargs + ["./timem-mpi", "-i"],
            )

            add_timem_test(
                "timemory-timem-mpi-individual-json",
                [dmprun]
                + dmpargs
                + [
                    "./timem-mpi",
                    "-i",
                    "-o",
                    "timem-mpi-output",
                ],
            )

    if args.python:
        pyct.test(
            "timemory-python",
            [
                sys.executable,
                "-c",
                "import timemory; print(timemory.__file__)",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-profiler",
            [
                sys.executable,
                "./ex_python_profiler",
                "10",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-profiler-main",
            [
                sys.executable,
                "-m",
                "timemory.profiler",
                "--max-stack-depth=10",
                "-l",
                "-f",
                "-F",
                "-c",
                "wall_clock",
                "peak_rss",
                "--",
                "./ex_python_external",
                "12",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-profiler-main-no-opts",
            [
                sys.executable,
                "-m",
                "timemory.profiler",
                "./ex_python_external",
                "5",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-profiler-main-script",
            [
                os.path.join(
                    pyct.BINARY_DIRECTORY, "bin", "timemory-python-profiler"
                ),
                "--max-stack-depth=10",
                "-l",
                "-f",
                "-F",
                "-c",
                "wall_clock",
                "peak_rss",
                "--",
                "./ex_python_external",
                "12",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-profiler-builtin",
            [
                sys.executable,
                "-m",
                "timemory.profiler",
                "--max-stack-depth=10",
                "-b",
                "-l",
                "-f",
                "-F",
                "-c",
                "wall_clock",
                "peak_rss",
                "--",
                "./ex_python_builtin",
                "10",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-trace",
            [
                sys.executable,
                "./ex_python_tracer",
                "10",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-trace-main",
            [
                sys.executable,
                "-m",
                "timemory.trace",
                "-l",
                "-f",
                "-F",
                "-c",
                "wall_clock",
                "peak_rss",
                "--",
                "./ex_python_external",
                "12",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-trace-main-no-opts",
            [
                sys.executable,
                "-m",
                "timemory.trace",
                "./ex_python_external",
                "5",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-trace-main-script",
            [
                os.path.join(
                    pyct.BINARY_DIRECTORY, "bin", "timemory-python-trace"
                ),
                "-l",
                "-f",
                "-F",
                "-c",
                "wall_clock",
                "peak_rss",
                "--",
                "./ex_python_external",
                "12",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-trace-builtin",
            [
                sys.executable,
                "-m",
                "timemory.trace",
                "-b",
                "-l",
                "-f",
                "-F",
                "-c",
                "wall_clock",
                "peak_rss",
                "--",
                "./ex_python_builtin",
                "10",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-sample",
            [
                sys.executable,
                "ex_python_sample",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        pyct.test(
            "timemory-python-general",
            [
                sys.executable,
                "ex_python_general",
            ],
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": base_env,
            },
        )

        if len(args.tools) > 0 and not (
            len(args.tools) == 1 and args.tools[0] == "avail"
        ):
            pyct.test(
                "timemory-python-timem",
                [
                    "./pytimem",
                    "sleep",
                    "2",
                ],
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "15",
                    "ENVIRONMENT": base_env,
                },
            )

        pyunittests = [
            "flat",
            "general",
            "hatchet",
            "rusage",
            "storage",
            "throttle",
            "timeline",
            "timing",
            "tools",
        ]

        pyusempi = {"hatchet": True}
        for t in pyunittests:
            pyunitcmd = [
                sys.executable,
                "-m",
                "timemory.test.test_{}".format(t),
            ]
            if (
                t in pyusempi
                and pyusempi[t]
                and args.mpi
                and dmprun is not None
            ):
                pyunitcmd = [dmprun] + dmpargs + pyunitcmd
            pyct.test(
                "python-unittest-{}".format(t),
                pyunitcmd,
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

    if not args.quick and not args.coverage and not args.minimal:
        if "compiler" in args.tools:
            pyct.test(
                construct_name("ex-compiler-instrument"),
                construct_command(["./ex_compiler_instrument"], args),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": ";".join(
                        [test_env, "TIMEMORY_COMPILER_COUT_OUTPUT=ON"]
                    ),
                },
            )

        pyct.test(
            construct_name("ex-derived"),
            construct_command(["./ex_derived"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-optional-off"),
            construct_command(["./ex_optional_off"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        overhead_cmd = ["./ex_cxx_overhead"]
        if args.coverage:
            overhead_cmd += ["40", "30"]

        pyct.test(
            construct_name("ex-cxx-overhead"),
            construct_command(overhead_cmd, args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "600",
                "ENVIRONMENT": test_env,
            },
        )

        if args.cuda or args.hip:
            pyct.test(
                construct_name("ex-gpu-event"),
                ["./ex_gpu_event"],
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

            pyct.test(
                construct_name("ex-gpu-kernel-instrument-v1"),
                ["./ex_kernel_instrument"],
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

            pyct.test(
                construct_name("ex-gpu-kernel-instrument-v2"),
                ["./ex_kernel_instrument_v2"],
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

            pyct.test(
                construct_name("ex-gpu-kernel-instrument-v3"),
                ["./ex_kernel_instrument_v3"],
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

        pyct.test(
            construct_name("ex-cxx-minimal"),
            construct_command(["./ex_cxx_minimal"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-c-minimal-library-overload"),
            construct_command(["./ex_c_minimal_library_overload"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-c-timing"),
            construct_command(["./ex_c_timing"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-cxx-minimal-library"),
            construct_command(["./ex_cxx_minimal_library"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-optional-on"),
            construct_command(["./ex_optional_on"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-c-minimal-library"),
            construct_command(["./ex_c_minimal_library"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        if pyct.BUILD_TYPE.upper() not in ["DEBUG", "MINSIZEREL"]:
            ert_cmd = ["./ex_ert"]
            if args.coverage:
                ert_cmd += ["512", "1081344", "2"]

                pyct.test(
                    construct_name("ex-ert"),
                    construct_command(ert_cmd, args),
                    {
                        "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                        "LABELS": pyct.PROJECT_NAME,
                        "TIMEOUT": "120",
                        "ENVIRONMENT": test_env,
                    },
                )

        pyct.test(
            construct_name("ex-cxx-tuple"),
            construct_command(["./ex_cxx_tuple"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        if args.gotcha:
            pyct.test(
                construct_name("ex-gotcha"),
                construct_command(["./ex_gotcha"], args),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

            pyct.test(
                construct_name("ex-gotcha-replacement"),
                construct_command(["./ex_gotcha_replacement"], args),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

            if args.mpi and dmprun is not None:
                ex_gotcha_cmd = [dmprun] + dmpargs + ["./ex_gotcha_mpi"]
                pyct.test(
                    construct_name("ex-gotcha-mpi"),
                    construct_command(ex_gotcha_cmd, args),
                    {
                        "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                        "LABELS": pyct.PROJECT_NAME,
                        "TIMEOUT": "120",
                        "ENVIRONMENT": test_env,
                    },
                )

        if args.python:
            _bindings_cmd = [sys.executable, "./ex_python_bindings"]
            if dmprun is not None:
                _bindings_cmd = [dmprun] + dmpargs + _bindings_cmd

            pyct.test(
                construct_name("ex-python-bindings-ref"),
                construct_command(_bindings_cmd, args),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": ";".join(
                        [
                            base_env,
                            "TIMEMORY_OUTPUT_PATH=timemory-ex-python-bindings-output-ref",
                        ]
                    ),
                },
            )

            pyct.test(
                construct_name("ex-python-bindings"),
                construct_command(_bindings_cmd, args),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "DEPENDS": construct_name("ex-python-bindings-ref"),
                    "ENVIRONMENT": ";".join(
                        [
                            base_env,
                            "TIMEMORY_INPUT_PATH=timemory-ex-python-bindings-output-ref",
                            "TIMEMORY_DIFF_OUTPUT=ON",
                        ]
                    ),
                },
            )

            if args.caliper:
                pyct.test(
                    construct_name("ex-python-caliper"),
                    construct_command(
                        [sys.executable, "./ex_python_caliper", "-n", "10"],
                        args,
                    ),
                    {
                        "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                        "LABELS": pyct.PROJECT_NAME,
                        "TIMEOUT": "120",
                        "ENVIRONMENT": base_env,
                    },
                )

            pyct.test(
                construct_name("ex-python-general"),
                construct_command(
                    [sys.executable, "./ex_python_general"], args
                ),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": base_env,
                },
            )

            pyct.test(
                construct_name("ex-python-profiler"),
                construct_command(
                    [sys.executable, "./ex_python_profiler"], args
                ),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": base_env,
                },
            )

            pyct.test(
                construct_name("ex-python-sample"),
                construct_command([sys.executable, "./ex_python_sample"], args),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": base_env,
                },
            )

            pyct.test(
                construct_name("ex-python-minimal"),
                construct_command(
                    [sys.executable, "./ex_python_minimal"], args
                ),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "480",
                    "ENVIRONMENT": test_env,
                },
            )

        if args.caliper:
            pyct.test(
                construct_name("ex-caliper"),
                construct_command(["./ex_caliper"], args),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

        pyct.test(
            construct_name("ex-c-minimal"),
            construct_command(["./ex_c_minimal"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-cxx-minimal-library-overload"),
            construct_command(["./ex_cxx_minimal_library_overload"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-cxx-basic"),
            construct_command(["./ex_cxx_basic"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        pyct.test(
            construct_name("ex-statistics"),
            construct_command(["./ex_cxx_statistics"], args),
            {
                "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                "LABELS": pyct.PROJECT_NAME,
                "TIMEOUT": "120",
                "ENVIRONMENT": test_env,
            },
        )

        if args.likwid:
            pyct.test(
                construct_name("ex-likwid"),
                construct_command(["./ex_likwid"], args),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "120",
                    "ENVIRONMENT": test_env,
                },
            )

            if args.python:
                pyct.test(
                    construct_name("ex-python-likwid"),
                    construct_command(
                        [sys.executable, "./ex_python_likwid"], args
                    ),
                    {
                        "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                        "LABELS": pyct.PROJECT_NAME,
                        "TIMEOUT": "120",
                        "ENVIRONMENT": test_env,
                    },
                )

        paranoid = 3
        try:
            if os.path.exists("/proc/sys/kernel/perf_event_paranoid"):
                with open("/proc/sys/kernel/perf_event_paranoid", "r") as f:
                    data = [int(line) for line in f.readlines()]
                    if len(data) > 0:
                        paranoid = data[0]
        except IOError:
            pass

        if args.papi and paranoid < 2:
            pyct.test(
                construct_name("ex-cpu-roofline"),
                construct_roofline_command(
                    ["./ex_cpu_roofline"],
                    "cpu-roofline",
                    ["-t", "cpu_roofline"],
                ),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "900",
                    "ENVIRONMENT": test_env,
                },
            )

            pyct.test(
                construct_name("ex-cpu-roofline.sp"),
                construct_roofline_command(
                    ["./ex_cpu_roofline.sp"],
                    "cpu-roofline.sp",
                    ["-t", "cpu_roofline"],
                    use_mpi=False,
                ),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "900",
                    "ENVIRONMENT": test_env,
                },
            )

        if args.cupti:
            pyct.test(
                construct_name("ex-gpu-roofline"),
                construct_roofline_command(
                    ["./ex_gpu_roofline"],
                    "gpu-roofline",
                    ["-t", "gpu_roofline"],
                    use_mpi=False,
                ),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "900",
                    "ENVIRONMENT": ";".join([test_env, "TIMEMORY_VERBOSE=3"]),
                },
            )

            pyct.test(
                construct_name("ex-gpu-roofline.sp"),
                construct_roofline_command(
                    ["./ex_gpu_roofline.sp"],
                    "gpu-roofline.sp",
                    ["-t", "gpu_roofline"],
                    use_mpi=False,
                ),
                {
                    "WORKING_DIRECTORY": pyct.BINARY_DIRECTORY,
                    "LABELS": pyct.PROJECT_NAME,
                    "TIMEOUT": "900",
                    "ENVIRONMENT": ";".join([test_env, "TIMEMORY_VERBOSE=3"]),
                },
            )

    pyct.generate_config(pyct.BINARY_DIRECTORY)
    pyct.generate_test_file(os.path.join(pyct.BINARY_DIRECTORY, "tests"))
    pyct.copy_files()
    # copy over the custom scripts
    for itr in [
        "CustomInit.cmake",
        "CustomStages.cmake",
        "PyCTestPreInit.cmake",
    ]:
        _src = os.path.join(pyct.SOURCE_DIRECTORY, "cmake", "Templates", itr)
        _dst = os.path.join(pyct.BINARY_DIRECTORY, itr)
        if os.path.exists(_src):
            import shutil

            shutil.copy2(_src, _dst)

    if not args.generate:
        mem = None
        fname = "._report-memory.tmp"
        if not args.quiet and platform.system() in ["Darwin", "Linux"]:
            import subprocess as sp

            f = open(fname, "w")
            f.write("\n")
            f.close()
            script = os.path.join(
                pyct.SOURCE_DIRECTORY, "scripts", "report-memory.sh"
            )
            mem = sp.Popen([script, fname])

        ret = pyct.run(pyct.ARGUMENTS, pyct.BINARY_DIRECTORY)

        if os.path.exists(fname):
            os.remove(fname)

        if mem is not None:
            mem.terminate()

        if ret is not None and ret is False:
            sys.exit(1)
        if args.coverage:
            script = os.path.join(
                pyct.SOURCE_DIRECTORY, "scripts", "submit-coverage.sh"
            )
            cov = pyct.command([script, pyct.BINARY_DIRECTORY])
            cov.SetWorkingDirectory(pyct.SOURCE_DIRECTORY)
            cov.Execute()
            print("{}".format(cov.Output()))
    else:
        print("BUILD_NAME: {}".format(pyct.BUILD_NAME))


#
if __name__ == "__main__":
    try:
        run_pyctest()

    except Exception as e:
        print("Error running pyctest - {}".format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)

    sys.exit(0)
