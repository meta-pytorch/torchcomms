# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

BUILDDIR = os.environ.get("BUILDDIR", None)
LIBDIR = None
if BUILDDIR is not None:
    LIBDIR = os.path.join(BUILDDIR, "lib")

TORCH_DIR = None
try:
    import torch

    TORCH_DIR = os.path.dirname(torch.__file__)
except ImportError:
    pass

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

include_dirs = []
library_dirs = []
libraries = ["nccl"]

if os.path.isdir(os.path.join(CUDA_HOME, "include")):
    include_dirs.append(os.path.join(CUDA_HOME, "include"))
    library_dirs.append(os.path.join(CUDA_HOME, "lib64"))

if LIBDIR is not None:
    library_dirs.append(LIBDIR)

if TORCH_DIR is not None:
    include_dirs.append(os.path.join(TORCH_DIR, "include"))
    include_dirs.append(
        os.path.join(TORCH_DIR, "include", "torch", "csrc", "api", "include")
    )
    library_dirs.append(os.path.join(TORCH_DIR, "lib"))
    libraries.extend(["c10", "c10_cuda", "torch", "torch_cpu", "torch_cuda"])

runtime_library_dirs = list(library_dirs)

ext_modules = [
    Pybind11Extension(
        "cca4py",
        ["cca4py.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
    ),
]

setup(
    name="cca4py",
    version="0.1.0",
    description="CCA4Py: PyTorch CUDACachingAllocator <-> NCCL memory registration",
    ext_modules=ext_modules,
)
