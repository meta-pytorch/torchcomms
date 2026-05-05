#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-3 license found in the
# LICENSE file in the root directory of this source tree.

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
    # Fail with a helpful message — torch is required for all torchcomms builds.
    print(
        "\n"
        "ERROR: PyTorch is required to build torchcomms but was not found.\n"
        "\n"
        "If PyTorch is already installed (e.g. in a conda env), use:\n"
        "  pip install --no-build-isolation -e .\n"
        "\n"
        "Otherwise, install PyTorch first. For CUDA builds:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
        "\n"
        "  Adjust the CUDA suffix (cu118, cu121, cu124, cu126, cu128) to match your\n"
        "  installed CUDA toolkit version (check with: nvcc --version).\n"
        "\n"
        "If using the oss conda env, install PyTorch with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
        "  (adjust cu128 to match your CUDA version: cu118, cu121, cu124, cu126, cu128)\n"
        "  (check your CUDA version with: nvcc --version)\n",
        file=sys.stderr,
    )
    raise


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
USE_RCCLX = flag_enabled("USE_RCCLX", False)
USE_XCCL = flag_enabled("USE_XCCL", False)
IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
# Transport is CUDA-only; disable by default on ROCm but allow explicit opt-in.
USE_TRANSPORT = flag_enabled("USE_TRANSPORT", not IS_ROCM)
USE_TRITON = flag_enabled("USE_TRITON", False)

requirement_path = os.path.join(ROOT, "requirements.txt")
try:
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
except FileNotFoundError:
    install_requires = []

for i, req in enumerate(install_requires):
    if req.startswith("torch"):
        install_requires[i] = f"torch=={torch.__version__.partition('+')[0]}"


def get_version() -> str:
    with open(os.path.join(ROOT, "version.txt")) as f:
        version = f.readline().strip()

    # Overridden for nightly builds.
    if "BUILD_VERSION" in os.environ:
        version = os.environ["BUILD_VERSION"]

    return version


def detect_hipify_v2():
    try:
        from packaging.version import Version
        from torch.utils.hipify import __version__

        if Version(__version__) >= Version("2.0.0"):
            return True
    except Exception as e:
        print(
            "failed to detect pytorch hipify version, defaulting to version 1.0.0 behavior"
        )
        print(e)
    return False


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

        build_flags = []
        build_flags += _get_pybind11_abi_build_flags()
        if detect_hipify_v2():
            build_flags += ["-DHIPIFY_V2"]

        cfg = os.environ.get("CMAKE_BUILD_TYPE", "RelWithDebInfo")
        print(f"- Building with {cfg} configuration")

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
            f"-DUSE_NCCL={flag_str(USE_NCCL)}",
            f"-DUSE_NCCLX={flag_str(USE_NCCLX)}",
            f"-DUSE_GLOO={flag_str(USE_GLOO)}",
            f"-DUSE_RCCL={flag_str(USE_RCCL)}",
            f"-DUSE_RCCLX={flag_str(USE_RCCLX)}",
            f"-DUSE_XCCL={flag_str(USE_XCCL)}",
            f"-DUSE_TRANSPORT={flag_str(USE_TRANSPORT)}",
            f"-DUSE_TRITON={flag_str(USE_TRITON)}",
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
        "lintrunner",
        "parameterized",
        "pydot",
    ],
}

BACKEND_FLAGS = [
    ("nccl", USE_NCCL),
    ("ncclx", USE_NCCLX),
    ("gloo", USE_GLOO),
    ("rccl", USE_RCCL),
    ("rcclx", USE_RCCLX),
    ("xccl", USE_XCCL),
]

ext_modules = [CMakeExtension("torchcomms._comms")]
ext_modules += [
    CMakeExtension(f"torchcomms._comms_{name}")
    for name, enabled in BACKEND_FLAGS
    if enabled
]
if USE_TRANSPORT:
    ext_modules.append(CMakeExtension("torchcomms._transport"))

backend_entry_points = ["fake = torchcomms._comms"] + [
    f"{name} = torchcomms._comms_{name}" for name, enabled in BACKEND_FLAGS if enabled
]

setup(
    name="torchcomms",
    version=get_version(),
    packages=find_packages("comms"),
    package_dir={"": "comms"},
    package_data={
        "torchcomms.triton.fb": ["*.bc"],
    },
    entry_points={
        "torchcomms.backends": backend_entry_points,
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=install_requires,
    extras_require=extras_require,
)
