# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import platform

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

# The colltrace v2 Python bindings (the `colltrace` module), linked against a
# prebuilt libcommsutils.so. Two distributions build from here: the conda
# `commsutils` feedstock (fbcode/conda/feedstock/commsutils) runs pip in this
# directory against the `nccl` package's libcommsutils; the `commsutils` wheel
# (genai/msl_infra/dev/wheels/commsutils) copies this file and colltrace.cpp
# next to its own pyproject.toml and builds against the ncclx build tree.
HERE = os.path.dirname(os.path.abspath(__file__))
FBCODE_DIR = os.environ.get(
    "FBCODE_DIR", os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
)
COMMSUTILS_LIB_DIR = os.environ.get("COMMSUTILS_LIB_DIR") or (
    os.path.join(os.environ["CONDA_PREFIX"], "lib")
    if "CONDA_PREFIX" in os.environ
    else None
)
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
if not os.path.isfile(os.path.join(CUDA_HOME, "include", "cuda_runtime.h")):
    conda_cuda = os.path.join(
        os.environ.get("CONDA_PREFIX", ""), "targets", f"{platform.machine()}-linux"
    )
    if os.path.isfile(os.path.join(conda_cuda, "include", "cuda_runtime.h")):
        CUDA_HOME = conda_cuda

library_dirs = [COMMSUTILS_LIB_DIR] if COMMSUTILS_LIB_DIR else []

setup(
    ext_modules=[
        Pybind11Extension(
            "colltrace",
            ["colltrace.cpp"],
            include_dirs=[FBCODE_DIR, os.path.join(CUDA_HOME, "include")],
            library_dirs=library_dirs,
            runtime_library_dirs=library_dirs,
            libraries=["commsutils"],
            cxx_std=20,
        ),
    ],
)
