#!/usr/bin/env python

import os
import sys
import argparse
import warnings
import platform


# some Cray systems default to static libraries and the build
# will fail because BUILD_SHARED_LIBS will get set to off
if os.environ.get("CRAYPE_VERSION") is not None:
    os.environ["CRAYPE_LINK_TYPE"] = "dynamic"

cmake_args = [
    "-DPYTHON_EXECUTABLE:FILEPATH={}".format(sys.executable),
    "-DPython3_EXECUTABLE:FILEPATH={}".format(sys.executable),
]
cmake_options = []  # array of functors
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", help="Print help", action="store_true")

# support CMAKE_ARGS environment variables because
#  --install-option for pip is a pain to use
env_cmake_args = os.environ.get("CMAKE_ARGS", None)
if env_cmake_args is not None:
    for _arg in env_cmake_args.split(" "):
        if "CMAKE_INSTALL_PREFIX" not in _arg:
            cmake_args += [_arg]

gotcha_opt = False
if platform.system() == "Linux":
    gotcha_opt = True


def parse_requirements(fname="requirements.txt"):
    _req = []
    requirements = []
    # read in the initial set of requirements
    with open(fname, "r") as fp:
        _req = list(filter(bool, (line.strip() for line in fp)))
    # look for entries which read other files
    for itr in _req:
        if itr.startswith("-r "):
            # read another file
            for fitr in itr.split(" "):
                if os.path.exists(fitr):
                    requirements.extend(parse_requirements(fitr))
        else:
            # append package
            requirements.append(itr)
    # return the requirements
    return requirements


def get_project_version():
    # open "VERSION"
    with open(os.path.join(os.getcwd(), "VERSION"), "r") as f:
        data = f.read().replace("\n", "")
    # make sure is string
    if isinstance(data, list) or isinstance(data, tuple):
        return data[0]
    else:
        return data


def get_long_description():
    long_descript = ""
    try:
        long_descript = open("README.md").read()
    except Exception:
        long_descript = ""
    return long_descript


def __str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1", "on")


def get_bool_option(_args, _name, default=False):
    _name = _name.replace("-", "_")
    _enable = getattr(_args, f"enable_{_name}")
    _disable = getattr(_args, f"disable_{_name}")
    _ret = default
    if _enable:
        _ret = True
    if _disable:
        _ret = False
    return _ret


def set_cmake_bool_option(opt, enable_opt, disable_opt):
    global cmake_args
    try:
        if enable_opt:
            cmake_args.append("-D{}:BOOL={}".format(opt, "ON"))
        if disable_opt:
            cmake_args.append("-D{}:BOOL={}".format(opt, "OFF"))
    except Exception as e:
        print("Exception: {}".format(e))


def add_arg_bool_option(
    lc_name, disp_name, default=None, doc="", disp_aliases=[]
):
    global parser
    global cmake_options

    # enable option
    parser.add_argument(
        "--enable-{}".format(lc_name),
        action="store_true",
        default=(None if default is False else default),
        help="Explicitly enable {} build. {}".format(disp_name, doc),
    )
    # disable option
    parser.add_argument(
        "--disable-{}".format(lc_name),
        action="store_true",
        default=None,
        help="Explicitly disable {} build. {}".format(disp_name, doc),
    )

    def _add_cmake_bool_option(_args):
        _name = lc_name.replace("-", "_")
        _enable = getattr(_args, f"enable_{_name}")
        _disable = getattr(_args, f"disable_{_name}")
        # if default=False is passed in, set _disable to false
        # only when neither --enable_{} and --disable_{} were not specified
        if default is False and _enable is None and _disable is None:
            _disable = True
        for itr in [disp_name] + disp_aliases:
            set_cmake_bool_option(itr, _enable, _disable)

    cmake_options.append(_add_cmake_bool_option)


# build variants
add_arg_bool_option(
    "require-packages",
    "TIMEMORY_REQUIRE_PACKAGES",
    doc=(
        "Enables auto-detection of third-party packages"
        + " and suppresses configuration failure"
        + " when packages are not found"
    ),
)
add_arg_bool_option("shared-libs", "BUILD_SHARED_LIBS")
add_arg_bool_option("static-libs", "BUILD_STATIC_LIBS")
add_arg_bool_option(
    "install-rpath-use-link-path",
    "CMAKE_INSTALL_RPATH_USE_LINK_PATH",
    default=True,
)
add_arg_bool_option("c", "TIMEMORY_BUILD_C")
add_arg_bool_option(
    "python",
    "TIMEMORY_USE_PYTHON",
    default=True,
    doc="Build python bindings",
)
add_arg_bool_option("fortran", "TIMEMORY_BUILD_FORTRAN")
add_arg_bool_option(
    "arch", "TIMEMORY_USE_ARCH", doc="Compile everything for CPU arch"
)
add_arg_bool_option(
    "portable",
    "TIMEMORY_BUILD_PORTABLE",
    doc="Disable CPU arch flags likely to cause portability issue",
)
add_arg_bool_option("ert", "TIMEMORY_BUILD_ERT")
add_arg_bool_option(
    "skip-build",
    "TIMEMORY_SKIP_BUILD",
    doc="Disable building any libraries",
)
add_arg_bool_option(
    "unity-build",
    "TIMEMORY_UNITY_BUILD",
    doc="timemory-localized CMAKE_UNITY_BUILD",
)
add_arg_bool_option(
    "install-headers",
    "TIMEMORY_INSTALL_HEADERS",
    doc="Install timemory headers",
)
add_arg_bool_option(
    "install-config",
    "TIMEMORY_INSTALL_CONFIG",
    doc="Install cmake configuration",
)
add_arg_bool_option(
    "install-all",
    "TIMEMORY_INSTALL_ALL",
    doc="install target depends on all target",
)
# distributed memory parallelism
add_arg_bool_option("mpi", "TIMEMORY_USE_MPI")
add_arg_bool_option("nccl", "TIMEMORY_USE_NCCL")
add_arg_bool_option("upcxx", "TIMEMORY_USE_UPCXX")
# components
add_arg_bool_option("cuda", "TIMEMORY_USE_CUDA")
add_arg_bool_option("cupti", "TIMEMORY_USE_CUPTI")
add_arg_bool_option("hip", "TIMEMORY_USE_HIP")
add_arg_bool_option("papi", "TIMEMORY_USE_PAPI")
add_arg_bool_option("ompt", "TIMEMORY_USE_OMPT")
add_arg_bool_option("gotcha", "TIMEMORY_USE_GOTCHA", default=gotcha_opt)
add_arg_bool_option("caliper", "TIMEMORY_USE_CALIPER")
add_arg_bool_option("likwid", "TIMEMORY_USE_LIKWID")
add_arg_bool_option("perfetto", "TIMEMORY_USE_PERFETTO")
add_arg_bool_option("gperftools", "TIMEMORY_USE_GPERFTOOLS")
add_arg_bool_option("tau", "TIMEMORY_USE_TAU")
add_arg_bool_option("vtune", "TIMEMORY_USE_VTUNE")
# submodules
add_arg_bool_option("build-caliper", "TIMEMORY_BUILD_CALIPER")
add_arg_bool_option("build-gotcha", "TIMEMORY_BUILD_GOTCHA")
add_arg_bool_option("build-dyninst", "TIMEMORY_BUILD_DYNINST")
add_arg_bool_option(
    "build-dyninst-tpls",
    "TIMEMORY_BUILD_DYNINST_TPLS",
    doc="Build third-party library dependencies for Dyninst",
)
add_arg_bool_option(
    "build-ert",
    "TIMEMORY_BUILD_ERT",
    doc="Build/install empirical roofline toolkit tool",
)
add_arg_bool_option(
    "build-ompt",
    "TIMEMORY_BUILD_OMPT",
    doc="Build/install OpenMP with OMPT support",
)
add_arg_bool_option(
    "build-python",
    "TIMEMORY_BUILD_PYTHON",
    doc="Build bindings with internal pybind11",
)
# tools
add_arg_bool_option("tools", "TIMEMORY_BUILD_TOOLS", default=True)
add_arg_bool_option("timem", "TIMEMORY_BUILD_TIMEM", doc="Build timem tool")
add_arg_bool_option("jump", "TIMEMORY_BUILD_JUMP", default=False)
add_arg_bool_option("stubs", "TIMEMORY_BUILD_STUBS", default=False)
add_arg_bool_option(
    "avail", "TIMEMORY_BUILD_AVAIL", doc="Build timemory-avail tool"
)
add_arg_bool_option(
    "dyninst",
    "TIMEMORY_USE_DYNINST",
    disp_aliases=["TIMEMORY_BUILD_DYNINST_TOOLS", "TIMEMORY_BUILD_RUN"],
)
add_arg_bool_option(
    "compiler-instrumentation",
    "TIMEMORY_BUILD_COMPILER_INSTRUMENTATION",
    default=False,
)
add_arg_bool_option(
    "kokkos",
    "TIMEMORY_BUILD_KOKKOS_TOOLS",
    doc="Build separate KokkosP libraries",
)
add_arg_bool_option(
    "kokkos-config",
    "TIMEMORY_BUILD_KOKKOS_CONFIG",
    doc="Build array of dedicated KokkosP libraries",
)
add_arg_bool_option("mpip-library", "TIMEMORY_BUILD_MPIP_LIBRARY")
add_arg_bool_option("ompt-library", "TIMEMORY_BUILD_OMPT_LIBRARY")
add_arg_bool_option("ncclp-library", "TIMEMORY_BUILD_NCCLP_LIBRARY")
add_arg_bool_option("mallocp-library", "TIMEMORY_BUILD_MALLOCP_LIBRARY")
add_arg_bool_option("python-hatchet", "TIMEMORY_BUILD_PYTHON_HATCHET")
# miscellaneous
add_arg_bool_option("pybind-install", "PYBIND11_INSTALL", default=False)
add_arg_bool_option("build-testing", "TIMEMORY_BUILD_TESTING")
parser.add_argument(
    "--cxx-standard",
    default=17,
    type=int,
    choices=[17, 20, 23],
    help="Set C++ language standard",
)
parser.add_argument(
    "--cmake-args",
    default=[],
    type=str,
    nargs="*",
    help="{}{}".format(
        "Pass arguments to cmake. Use w/ pip installations and --install-option, e.g. ",
        '--install-option=--cmake-args="-DTIMEMORY_BUILD_LTO=ON -DCMAKE_UNITY_BUILD=OFF"',
    ),
)
parser.add_argument(
    "--c-flags",
    default=None,
    type=str,
    nargs="*",
    help="Explicitly set CMAKE_C_FLAGS",
)
parser.add_argument(
    "--cxx-flags",
    default=None,
    type=str,
    nargs="*",
    help="Explicitly set CMAKE_CXX_FLAGS",
)
parser.add_argument(
    "--enable-develop",
    action="store_true",
    default=None,
    help="Enable a development install (timemory headers, cmake config, etc.)",
)
parser.add_argument(
    "--disable-develop",
    action="store_true",
    default=None,
    help="Disable a development install (timemory headers, cmake config, etc.)",
)
parser.add_argument(
    "--dist-info",
    action="store_true",
    default=None,
    help="Generate dist-info (internal usage)",
)

args, left = parser.parse_known_args()
# if help was requested, print these options and then add '--help' back
# into arguments so that the skbuild/setuptools argparse catches it
if args.help:
    parser.print_help()
    left.append("--help")
sys.argv = sys.argv[:1] + left

if args.c_flags is not None:
    cmake_args.append("-DCMAKE_C_FLAGS={}".format(" ".join(args.c_flags)))

if args.cxx_flags is not None:
    cmake_args.append("-DCMAKE_CXX_FLAGS='{}'".format(" ".join(args.cxx_flags)))

if args.enable_develop:
    for itr in ["install_headers", "install_config"]:
        setattr(args, f"enable_{itr}", True)

if args.disable_develop:
    for itr in ["install_headers", "install_config"]:
        setattr(args, f"disable_{itr}", True)

# loop over the functors
for itr in cmake_options:
    itr(args)

runtime_req_file = (
    ".requirements/mpi_runtime.txt"
    if args.enable_mpi and not args.disable_mpi
    else ".requirements/runtime.txt"
)

cmake_args.append("-DCMAKE_CXX_STANDARD={}".format(args.cxx_standard))

for itr in args.cmake_args:
    cmake_args += itr.split()

if platform.system() == "Darwin":
    # scikit-build will set this to 10.6 and C++ compiler check will fail
    version = platform.mac_ver()[0].split(".")
    version = ".".join([version[0], version[1]])
    cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(version)]

# DO THIS LAST!
# support TIMEMORY_SETUP_ARGS environment variables because
#  --install-option for pip is a pain to use
# Env variables should be space-delimited set of cmake arguments, e.g.:
#   export TIMEMORY_SETUP_ARGS="-DTIMEMORY_USE_MPI=ON -DTIMEMORY_USE_OMPT=OFF"
env_cmake_args = os.environ.get("TIMEMORY_SETUP_ARGS", None)
if env_cmake_args is not None:
    cmake_args += env_cmake_args.split(" ")


def _generate_dist_info():
    from setuptools import setup

    # suppress:
    #  "setuptools_scm/git.py:68: UserWarning: "/.../<PACKAGE>"
    #       is shallow and may cause errors"
    # since 'error' in output causes CDash to interpret warning as error
    with warnings.catch_warnings():
        setup(
            name="timemory",
            packages=["timemory"],
            version=get_project_version(),
            long_description=get_long_description(),
            long_description_content_type="text/markdown",
            install_requires=parse_requirements(runtime_req_file),
            extras_require={
                "all": parse_requirements("requirements.txt")
                + parse_requirements(".requirements/mpi_runtime.txt"),
                "mpi": parse_requirements(".requirements/mpi_runtime.txt"),
                "build": parse_requirements(".requirements/build.txt"),
            },
            python_requires=">=3.6",
            entry_points={
                "console_scripts": [
                    "timemory-plotter=timemory.plotting.__main__:try_plot",
                    "timemory-roofline=timemory.roofline.__main__:try_plot",
                    "timemory-analyze=timemory.analyze.__main__:try_analyze",
                    "timemory-python-profiler=timemory.profiler.__main__:main",
                    "timemory-python-trace=timemory.trace.__main__:main",
                ],
            },
        )


_use_skbuild = True
_dist_info_only = os.environ.get("TIMEMORY_DIST_INFO", None)
if args.dist_info or (
    _dist_info_only is not None and __str2bool(_dist_info_only)
):
    _use_skbuild = False

if _use_skbuild:
    try:
        from skbuild import setup
        from skbuild.command.install import install as skinstall
    except (ImportError, ModuleNotFoundError) as e:
        sys.stderr.write(
            "{} {}".format(
                "Warning! timemory requires scikit-build for full installation.",
                "Generating dist-info only...\n",
            )
        )
        sys.stderr.write(f"{e}\n")
        sys.stderr.flush()
        _use_skbuild = False

if _use_skbuild:

    class custom_install(skinstall):
        """Custom installation"""

        def __init__(self, *args, **kwargs):
            skinstall.__init__(self, *args, **kwargs)

        def run(self):
            print("\n\n\t[timemory] Running install...")
            print("\t\t {:10} : {}".format("base", self.install_base))
            print("\t\t {:10} : {}".format("purelib", self.install_purelib))
            print("\t\t {:10} : {}".format("platlib", self.install_platlib))
            print("\t\t {:10} : {}".format("headers", self.install_headers))
            print("\t\t {:10} : {}".format("lib", self.install_lib))
            print("\t\t {:10} : {}".format("scripts", self.install_scripts))
            print("\t\t {:10} : {}".format("data", self.install_data))
            print("\t\t {:10} : {}".format("userbase", self.install_userbase))
            print("\t\t {:10} : {}".format("usersite", self.install_usersite))
            print("\n\n")
            skinstall.run(self)
            for itr in self.get_outputs():
                print('[timemory] installed file : "{}"'.format(itr))

    def exclude_install_hook(cmake_manifest):
        def _filter_manifest(_manifest, *args):
            for itr in args:
                _manifest = list(
                    filter(
                        lambda name: itr not in name,
                        cmake_manifest,
                    )
                )
            return _manifest

        cmake_manifest = _filter_manifest(
            cmake_manifest,
            "pytest.ini",
            os.path.join("hatchet", "tests", "timemory_test.py"),
        )
        if not get_bool_option(args, "develop"):
            cmake_manifest = list(
                filter(lambda name: not (name.endswith(".a")), cmake_manifest)
            )
            if not get_bool_option(args, "install-config"):
                cmake_manifest = _filter_manifest(
                    cmake_manifest,
                    os.path.join("share", "cmake"),
                    os.path.join("lib", "cmake"),
                )
            if not get_bool_option(args, "install-headers"):
                cmake_manifest = _filter_manifest(cmake_manifest, "include")

        return cmake_manifest

    # suppress:
    #  "setuptools_scm/git.py:68: UserWarning: "/.../<PACKAGE>"
    #       is shallow and may cause errors"
    # since 'error' in output causes CDash to interpret warning as error
    with warnings.catch_warnings():
        print("CMake arguments: {}".format(" ".join(cmake_args)))
        setup(
            name="timemory",
            packages=["timemory"],
            version=get_project_version(),
            cmake_args=cmake_args,
            cmake_languages=("C", "CXX"),
            long_description=get_long_description(),
            long_description_content_type="text/markdown",
            install_requires=parse_requirements(runtime_req_file),
            extras_require={
                "all": parse_requirements("requirements.txt")
                + parse_requirements(".requirements/mpi_runtime.txt"),
                "mpi": parse_requirements(".requirements/mpi_runtime.txt"),
                "build": parse_requirements(".requirements/build.txt"),
            },
            python_requires=">=3.6",
            cmdclass=dict(install=custom_install),
            cmake_process_manifest_hook=exclude_install_hook,
            entry_points={
                "console_scripts": [
                    "timemory-plotter=timemory.plotting.__main__:try_plot",
                    "timemory-roofline=timemory.roofline.__main__:try_plot",
                    "timemory-analyze=timemory.analyze.__main__:try_analyze",
                    "timemory-python-profiler=timemory.profiler.__main__:main",
                    "timemory-python-trace=timemory.trace.__main__:main",
                ],
            },
        )

else:
    _generate_dist_info()
