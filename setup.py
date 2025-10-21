#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-3 license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import pathlib
import shlex
import sys

import torch

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as build_ext_orig
from torch.utils.cpp_extension import _get_pybind11_abi_build_flags


def flag_enabled(flag: str, default: bool):
    enabled = os.environ.get(flag)
    if enabled is None:
        enabled = default
    else:
        enabled = enabled in ("1", "ON")

    print(f"- {flag}={flag_str(enabled)}")
    return enabled


def flag_str(val: bool):
    return "ON" if val else "OFF"


ROOT = os.path.abspath(os.path.dirname(__file__))
TORCH_ROOT = os.path.dirname(torch.__file__)

print("Configuration:")
USE_NCCL = flag_enabled("USE_NCCL", True)
USE_NCCLX = flag_enabled("USE_NCCLX", True)
USE_GLOO = flag_enabled("USE_GLOO", True)
USE_RCCL = flag_enabled("USE_RCCL", False)

requirement_path = os.path.join(ROOT, "requirements.txt")
try:
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
except FileNotFoundError:
    install_requires = []


class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
            # All extensions are built from the same directory so we can
            # just use the first one
            break

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))

        pybind11_build_flags = _get_pybind11_abi_build_flags()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_PREFIX={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_DIR={extdir.parent.absolute()}",
            f"-DCMAKE_PREFIX_PATH={TORCH_ROOT}",
            f"-DCMAKE_CXX_FLAGS={shlex.quote(' '.join(pybind11_build_flags))}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DLIB_SUFFIX={os.environ.get('LIB_SUFFIX', 'lib')}",
            f"-DUSE_NCCL={flag_str(USE_NCCL)}",
            f"-DUSE_NCCLX={flag_str(USE_NCCLX)}",
            f"-DUSE_GLOO={flag_str(USE_GLOO)}",
            f"-DUSE_RCCL={flag_str(USE_RCCL)}",
        ]
        build_args = ["--", "-j"]

        os.chdir(str(build_temp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "--target", "install"] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


extras_require = {
    "dev": [
        "pytest",
        "numpy",
        "psutil",
    ],
}

ext_modules = [
    CMakeExtension("torchcomms._comms"),
]

if USE_NCCL:
    ext_modules += [
        CMakeExtension("torchcomms._comms_nccl"),
    ]
if USE_NCCLX:
    ext_modules += [
        CMakeExtension("torchcomms._comms_ncclx"),
    ]
if USE_GLOO:
    ext_modules += [
        CMakeExtension("torchcomms._comms_gloo"),
    ]
if USE_RCCL:
    ext_modules += [
        CMakeExtension("torchcomms._comms_rccl"),
    ]

setup(
    name="torchcomms",
    version="0.0.1",
    packages=find_packages("comms"),
    package_dir={"": "comms"},
    entry_points={
        "torchcomms.backends": [
            "nccl = torchcomms._comms_nccl",
            "ncclx = torchcomms._comms_ncclx",
            "gloo = torchcomms._comms_gloo",
            "rccl = torchcomms._comms_rccl",
        ]
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=install_requires,
    extras_require=extras_require,
)
