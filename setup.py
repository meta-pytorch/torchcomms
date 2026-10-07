#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-3 license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os.path
import pathlib
import shlex
import sys
from typing import Any, cast

from setuptools import Command, Extension, find_packages, setup
from setuptools.command.build import build as build_orig
from setuptools.command.build_ext import build_ext as build_ext_orig
from setuptools.command.egg_info import egg_info as egg_info_orig


def load_bdist_wheel() -> Any:
    try:
        module = importlib.import_module("setuptools.command.bdist_wheel")
    except ImportError:
        module = importlib.import_module("wheel.bdist_wheel")
    return module.bdist_wheel


bdist_wheel_orig = load_bdist_wheel()

# setup.py imports a source-tree helper below. Keep that import from creating
# __pycache__ in a revision-addressed source materialization.
sys.dont_write_bytecode = True
from torchcomms_build_info import (
    build_information,
    source_identity,
    write_build_information,
)

try:
    import torch
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
TORCHCOMMS_REVISION, TORCHCOMMS_SOURCE_DIRTY = source_identity(pathlib.Path(ROOT))


def get_torch_pybind11_include_root(build_temp: pathlib.Path) -> pathlib.Path:
    torch_include = pathlib.Path(torch.__file__).resolve().parent / "include"
    torch_pybind11 = torch_include / "pybind11"
    if not (torch_pybind11 / "pybind11.h").exists():
        raise RuntimeError(
            f"PyTorch pybind11 headers were not found under {torch_pybind11}."
        )

    include_root = build_temp / "torch_pybind11_include"
    include_root.mkdir(parents=True, exist_ok=True)
    link = include_root / "pybind11"
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            raise RuntimeError(f"Expected {link} to be a symlink.")
        link.unlink()
    link.symlink_to(torch_pybind11, target_is_directory=True)
    return include_root


print("Configuration:")
IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
# Backend defaults flip automatically based on whether the installed torch is ROCm.
USE_NCCL = flag_enabled("USE_NCCL", not IS_ROCM)
USE_NCCLX = flag_enabled("USE_NCCLX", not IS_ROCM)
USE_GLOO = flag_enabled("USE_GLOO", True)
USE_RCCL = flag_enabled("USE_RCCL", IS_ROCM)
USE_RCCLX = flag_enabled("USE_RCCLX", False)
# Select the real rcclx-dev sharded-relay implementation (RcclxApiShardedRelay.cpp)
# over the stub (RcclxApiShardedRelayStub.cpp). Both TUs define the same
# DefaultRcclxApi::shardedRelay* symbols with no #ifdef guards, so exactly one
# must be compiled. In buck this is a select() on the rccl constraint (see
# comms/torchcomms/rcclx/BUCK); the OSS/wheel CMake path keys off this flag.
# Only meaningful when USE_RCCLX is set; requires linking an rcclx-dev librccl
# that actually exports the ncclShardedRelayMultiGroup* symbols.
USE_RCCLX_DEV = flag_enabled("USE_RCCLX_DEV", False)
USE_XCCL = flag_enabled("USE_XCCL", False)
# Transport is CUDA-only; disable by default on ROCm but allow explicit opt-in.
USE_TRANSPORT = flag_enabled("USE_TRANSPORT", not IS_ROCM)
# Minimal RDMA CCA-hook extension. CUDA-only and requires the NCCLX static lib;
# default ON when NCCLX is built (and not ROCm).
USE_TRANSPORT_CCA_HOOK = flag_enabled(
    "USE_TRANSPORT_CCA_HOOK", USE_NCCLX and not IS_ROCM
)
USE_TRITON = flag_enabled("USE_TRITON", False)
# The NCCLX make build produces libobservatory.so but no package ships it, so
# the wheel has to. Turn this off wherever the environment already supplies the
# process's one copy. A second copy gives each backend its own registry.
TORCHCOMMS_BUNDLE_OBSERVATORY = flag_enabled("TORCHCOMMS_BUNDLE_OBSERVATORY", True)


def parse_requirements(path: str) -> list[str]:
    """Parse a pip requirements file, skipping blank lines and comments."""
    requirements = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


requirement_path = os.path.join(ROOT, "requirements.txt")
install_requires = parse_requirements(requirement_path)

for i, req in enumerate(install_requires):
    if req.startswith("torch"):
        install_requires[i] = f"torch=={torch.__version__.partition('+')[0]}"

dev_requirement_path = os.path.join(ROOT, "dev-requirements.txt")
dev_requires = parse_requirements(dev_requirement_path)


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


def configured_directory(name: str) -> pathlib.Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError(f"{name} is not a directory: {resolved}")
    return resolved


def required_executable(name: str) -> pathlib.Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"{name} is not executable: {resolved}")
    return resolved


def python_build_directory(name: str) -> pathlib.Path:
    root = configured_directory("TORCHCOMMS_PYTHON_BUILD_DIR")
    if root is None:
        raise RuntimeError(
            "TORCHCOMMS_PYTHON_BUILD_DIR is required for isolated builds"
        )
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class IsolatedBuild(build_orig):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(python_build_directory("build"))


class IsolatedEggInfo(egg_info_orig):
    def initialize_options(self):
        super().initialize_options()
        self.egg_base = str(python_build_directory("metadata"))


class IsolatedBdistWheel(bdist_wheel_orig):
    def initialize_options(self):
        super().initialize_options()
        self.bdist_dir = str(python_build_directory("wheel"))


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
            # All extensions are built from the same directory so we can
            # just use the first one
            break
        if not self.dry_run:
            package_root = pathlib.Path(self.build_lib).absolute() / "torchcomms"
            strip_value = os.environ.get("STRIP_EXECUTABLE", "").strip()
            if strip_value:
                strip = required_executable("STRIP_EXECUTABLE")
                for artifact in sorted(package_root.glob("*.so*")):
                    if artifact.is_file() and not artifact.is_symlink():
                        self.spawn([str(strip), "--strip-debug", str(artifact)])
            enabled_backends = [name for name, enabled in BACKEND_FLAGS if enabled]
            information = build_information(
                root=pathlib.Path(ROOT),
                package_root=package_root,
                package_version=PACKAGE_VERSION,
                pytorch_version=torch.__version__,
                pytorch_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                torchcomms_revision=TORCHCOMMS_REVISION,
                source_dirty=TORCHCOMMS_SOURCE_DIRTY,
                use_ncclx=USE_NCCLX,
                bundle_observatory=TORCHCOMMS_BUNDLE_OBSERVATORY,
                enabled_backends=enabled_backends,
            )
            write_build_information(package_root / "_build_info.json", information)

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))

        compile_flags = shlex.split(os.environ.get("TORCHCOMMS_COMPILE_FLAGS", ""))
        build_flags = list(compile_flags)
        cuda_flags = shlex.split(os.environ.get("TORCHCOMMS_CUDA_FLAGS", ""))
        linker_flags = shlex.split(os.environ.get("TORCHCOMMS_LINKER_FLAGS", ""))
        if detect_hipify_v2():
            build_flags += ["-DHIPIFY_V2"]
        pybind11_include_root = get_torch_pybind11_include_root(build_temp)
        cmake_prefixes = []
        dependency_prefix = os.environ.get("CONDA_PREFIX", "").strip()
        if dependency_prefix:
            cmake_prefixes.append(dependency_prefix)
        configured_prefixes = os.environ.get("CMAKE_PREFIX_PATH", "").strip()
        if configured_prefixes:
            cmake_prefixes.extend(configured_prefixes.split(os.pathsep))
        cmake_prefixes.append(TORCH_ROOT)
        cmake_prefix_path = ";".join(dict.fromkeys(cmake_prefixes))
        cuda_home = os.environ.get("CUDA_HOME", "").strip()

        cfg = os.environ.get("CMAKE_BUILD_TYPE", "RelWithDebInfo")
        print(f"- Building with {cfg} configuration")

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_PREFIX={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_DIR={extdir.parent.absolute()}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-DTORCHCOMMS_PYBIND11_INCLUDE_DIR={pybind11_include_root}",
            f"-DCMAKE_C_FLAGS={shlex.join(compile_flags)}",
            f"-DCMAKE_CXX_FLAGS={shlex.join(build_flags)}",
            f"-DCMAKE_CUDA_FLAGS={shlex.join(cuda_flags)}",
            f"-DCMAKE_SHARED_LINKER_FLAGS={shlex.join(linker_flags)}",
            f"-DCMAKE_MODULE_LINKER_FLAGS={shlex.join(linker_flags)}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DLIB_SUFFIX={os.environ.get('LIB_SUFFIX', 'lib')}",
            f"-DUSE_NCCL={flag_str(USE_NCCL)}",
            f"-DUSE_NCCLX={flag_str(USE_NCCLX)}",
            f"-DUSE_GLOO={flag_str(USE_GLOO)}",
            f"-DUSE_RCCL={flag_str(USE_RCCL)}",
            f"-DUSE_RCCLX={flag_str(USE_RCCLX)}",
            f"-DUSE_RCCLX_DEV={flag_str(USE_RCCLX_DEV)}",
            f"-DUSE_XCCL={flag_str(USE_XCCL)}",
            f"-DUSE_TRANSPORT={flag_str(USE_TRANSPORT)}",
            f"-DUSE_TRANSPORT_CCA_HOOK={flag_str(USE_TRANSPORT_CCA_HOOK)}",
            f"-DUSE_TRITON={flag_str(USE_TRITON)}",
            f"-DTORCHCOMMS_BUNDLE_OBSERVATORY={flag_str(TORCHCOMMS_BUNDLE_OBSERVATORY)}",
        ]
        if cuda_home:
            cmake_args.extend(
                [
                    f"-DCUDAToolkit_ROOT={cuda_home}",
                    f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_home}",
                ]
            )
        parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "").strip()
        if parallel_level:
            try:
                parallel_jobs = int(parallel_level)
            except ValueError as error:
                raise ValueError(
                    "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer"
                ) from error
            if parallel_jobs <= 0:
                raise ValueError(
                    "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer"
                )
            build_args = ["--parallel", str(parallel_jobs)]
        else:
            build_args = ["--", "-j"]

        os.chdir(str(build_temp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "--target", "install"] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


extras_require = {
    "dev": dev_requires,
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
if USE_TRANSPORT_CCA_HOOK:
    ext_modules.append(CMakeExtension("torchcomms._transport_cca_hook"))

backend_entry_points = ["fake = torchcomms._comms"] + [
    f"{name} = torchcomms._comms_{name}" for name, enabled in BACKEND_FLAGS if enabled
]
# nccl-lazy is implemented inside the _comms_nccl extension via the
# LazyBackend<TorchCommNCCL> template; expose it as an additional entry
# point alias so `register_backend` discovery picks it up.
if USE_NCCL:
    backend_entry_points.append("nccl-lazy = torchcomms._comms_nccl")

PACKAGE_VERSION = get_version()

cmdclass: dict[str, type[Command]] = {"build_ext": build_ext}
if configured_directory("TORCHCOMMS_PYTHON_BUILD_DIR") is not None:
    cmdclass.update(
        {
            "bdist_wheel": IsolatedBdistWheel,
            "build": cast(type[Command], IsolatedBuild),
            "egg_info": IsolatedEggInfo,
        }
    )


setup(
    name="torchcomms",
    version=PACKAGE_VERSION,
    packages=find_packages("comms"),
    package_dir={"": "comms"},
    package_data={
        "torchcomms": ["_build_info.json"],
        "torchcomms.triton.fb": ["*.bc"],
    },
    entry_points={
        "torchcomms.backends": backend_entry_points,
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    extras_require=extras_require,
)
