# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import PlatformError

PACKAGE = "nccl.bindings"
LIBNAMES = ["nccl"]

_prefix = os.environ.get("PREFIX", "")
NCCL_INC = os.environ.get(
    "NCCL_INC", os.path.join(_prefix, "include") if _prefix else ""
)
NCCL_LIB = os.environ.get("NCCL_LIB", os.path.join(_prefix, "lib") if _prefix else "")


def _cuda_include_dir() -> str:
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        raise PlatformError("CUDA_HOME is not set")

    cuda_include = Path(cuda_home) / "include"
    if not cuda_include.is_dir():
        raise PlatformError(f"CUDA include directory does not exist: {cuda_include}")

    return str(cuda_include)


class BuildExt(build_ext):
    """Add CUDA headers only when extension compilation starts."""

    def build_extensions(self) -> None:
        cuda_include = _cuda_include_dir()
        for extension in self.extensions:
            extension.include_dirs.append(cuda_include)

        super().build_extensions()


def _ext(module: str, source: str) -> Extension:
    return Extension(
        module,
        sources=[source],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )


def libname_extensions(libname: str) -> list[Extension]:
    """Three per-library extensions: lowpp, cy variant, _internal loader.

    For libname="nccl":
        nccl.bindings.nccl              <- nccl/bindings/nccl.pyx
        nccl.bindings.cynccl            <- nccl/bindings/cynccl.pyx
        nccl.bindings._internal.nccl    <- nccl/bindings/_internal/nccl_linux.pyx
    """
    return [
        _ext(f"{PACKAGE}.{libname}", os.path.join(*PACKAGE.split("."), f"{libname}.pyx")),
        _ext(f"{PACKAGE}.cy{libname}", os.path.join(*PACKAGE.split("."), f"cy{libname}.pyx")),
        _ext(
            f"{PACKAGE}._internal.{libname}",
            os.path.join(*PACKAGE.split("."), "_internal", f"{libname}_linux.pyx"),
        ),
    ]


ext_modules = [
    _ext(f"{PACKAGE}._internal.utils", os.path.join(*PACKAGE.split("."), "_internal", "utils.pyx"))
]
for libname in LIBNAMES:
    ext_modules.extend(libname_extensions(libname))

ncclx_include_dirs = [NCCL_INC] if NCCL_INC else []
ncclx_library_dirs = [NCCL_LIB] if NCCL_LIB else []
ext_modules.append(
    Extension(
        f"{PACKAGE}.ncclx_internal",
        sources=[os.path.join(*PACKAGE.split("."), "ncclx_internal.pyx")],
        include_dirs=ncclx_include_dirs,
        library_dirs=ncclx_library_dirs,
        language="c++",
        extra_compile_args=["-std=c++17"],
        libraries=["nccl"],
    )
)


compiler_directives = {
    "embedsignature": True,
    "show_performance_hints": True,
    "freethreading_compatible": True,
}


setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives=compiler_directives,
    ),
    zip_safe=False,
)
