#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Standalone setup.py for the `torchcomms-ncclx` wheel.
#
# This produces a separate wheel that ships the NCCLX backend extension
# (`torchcomms_ncclx._comms_ncclx`) and registers it under the
# `torchcomms.backends` entry-point group. It depends on `torchcomms`
# (which provides `libtorchcomms.so` and the core Python API).
#
# Build with:
#   pip install --no-build-isolation -v ./comms/torchcomms/ncclx
# from the torchcomms repo root.

import os.path
import pathlib
import shlex
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as build_ext_orig

try:
    import torch
    from torch.utils.cpp_extension import _get_pybind11_abi_build_flags
except ModuleNotFoundError:
    print(
        "\n"
        "ERROR: PyTorch is required to build torchcomms-ncclx but was not found.\n"
        "\n"
        "Install PyTorch first, e.g.:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n",
        file=sys.stderr,
    )
    raise


HERE = os.path.abspath(os.path.dirname(__file__))
# Repo root is three levels up from comms/torchcomms/ncclx/
TORCHCOMMS_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TORCH_ROOT = os.path.dirname(torch.__file__)


def get_version() -> str:
    if "BUILD_VERSION" in os.environ:
        return os.environ["BUILD_VERSION"]
    with open(os.path.join(TORCHCOMMS_ROOT, "version.txt")) as f:
        return f.readline().strip()


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
            break  # all extensions share one cmake invocation

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.parent.mkdir(parents=True, exist_ok=True)

        build_flags = list(_get_pybind11_abi_build_flags())

        cfg = os.environ.get("CMAKE_BUILD_TYPE", "RelWithDebInfo")

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_PREFIX={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_DIR={extdir.parent.absolute()}",
            f"-DCMAKE_PREFIX_PATH={TORCH_ROOT}",
            f"-DCMAKE_CXX_FLAGS={shlex.quote(' '.join(build_flags))}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DLIB_SUFFIX={os.environ.get('LIB_SUFFIX', 'lib')}",
            f"-DTORCHCOMMS_ROOT={TORCHCOMMS_ROOT}",
        ]
        build_args = ["--", "-j"]

        os.chdir(str(build_temp))
        self.spawn(["cmake", HERE] + cmake_args)
        if not self.dry_run:
            # Build only (no install) — CMAKE_LIBRARY_OUTPUT_DIRECTORY puts the
            # .so exactly where setuptools expects to find it.
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(str(cwd))


extras_require = {
    "dev": [
        "pytest",
        "numpy",
        "psutil",
        "parameterized",
        "expecttest",
    ],
}

install_requires = [
    f"torch=={torch.__version__.partition('+')[0]}",
    "torchcomms",
]

setup(
    name="torchcomms-ncclx",
    version=get_version(),
    description="NCCLX backend for torchcomms (Meta's extended NCCL fork)",
    long_description=(
        open(os.path.join(HERE, "README.md")).read()
        if os.path.exists(os.path.join(HERE, "README.md"))
        else ""
    ),
    long_description_content_type="text/markdown",
    packages=find_packages(where=HERE, include=["torchcomms_ncclx*"]),
    package_dir={"": "."},
    package_data={"torchcomms_ncclx": ["*.pyi", "py.typed"]},
    ext_modules=[CMakeExtension("torchcomms_ncclx._comms_ncclx")],
    cmdclass={"build_ext": build_ext},
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "torchcomms.backends": [
            "ncclx = torchcomms_ncclx._comms_ncclx",
        ],
    },
    python_requires=">=3.10",
)
