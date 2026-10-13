# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import torch
from packaging.version import InvalidVersion, Version
from setuptools import Extension, setup
from setuptools.command.build import build as build_orig
from setuptools.command.build_ext import build_ext as build_ext_orig
from setuptools.command.build_py import build_py as build_py_orig
from setuptools.command.egg_info import egg_info as egg_info_orig
from wheel.bdist_wheel import bdist_wheel as bdist_wheel_orig


PACKAGING_ROOT = pathlib.Path(__file__).resolve().parent
BACKEND_ROOT = PACKAGING_ROOT.parent
TORCHCOMMS_REPO_ROOT = BACKEND_ROOT.parents[2]
sys.path.insert(0, str(BACKEND_ROOT))
if TYPE_CHECKING:
    from torchcomms.mccl import _identity as identity
else:
    import _identity as identity


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def required_directory(name: str) -> pathlib.Path:
    path = pathlib.Path(required_environment(name))
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError(f"{name} is not a directory: {resolved}")
    return resolved


def python_build_directory(name: str) -> pathlib.Path:
    root = required_directory("TORCHCOMMS_MCCL_PYTHON_BUILD_DIR")
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_version() -> str:
    raw_version = os.environ.get("TORCHCOMMS_MCCL_VERSION", "0.1.0.dev0")
    try:
        return str(Version(raw_version))
    except InvalidVersion as error:
        raise RuntimeError(
            f"TORCHCOMMS_MCCL_VERSION is not a valid PEP 440 version: {raw_version}"
        ) from error


def installed_file(distribution: str, relative_path: str) -> pathlib.Path:
    return identity.distribution_file(distribution, relative_path)


def get_torch_pybind11_include_root(build_temp: pathlib.Path) -> pathlib.Path:
    torch_include = pathlib.Path(torch.__file__).resolve().parent / "include"
    torch_pybind11 = torch_include / "pybind11"
    if not (torch_pybind11 / "pybind11.h").exists():
        raise RuntimeError(
            f"PyTorch pybind11 headers were not found under {torch_pybind11}"
        )
    include_root = build_temp / "torch_pybind11_include"
    include_root.mkdir(parents=True, exist_ok=True)
    link = include_root / "pybind11"
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            raise RuntimeError(f"Expected {link} to be a symlink")
        link.unlink()
    link.symlink_to(torch_pybind11, target_is_directory=True)
    return include_root


def checked_source_root(name: str, expected: pathlib.Path) -> pathlib.Path:
    configured = pathlib.Path(required_environment(name)).resolve()
    if configured != expected.resolve():
        raise RuntimeError(
            f"{name} must identify the repository containing this packaging source"
        )
    return configured


PACKAGE_VERSION = package_version()
TORCHCOMMS_SOURCE_ROOT = checked_source_root(
    "TORCHCOMMS_SOURCE_ROOT", TORCHCOMMS_REPO_ROOT
)
MCCL_SOURCE_ROOT = pathlib.Path(required_environment("MCCL_SOURCE_ROOT")).resolve()
TORCHCOMMS_DEPS_ROOT = pathlib.Path(
    required_environment("TORCHCOMMS_DEPS_ROOT")
).resolve()
TORCHCOMMS_LIBRARY = installed_file(
    "torchcomms", "torchcomms/libtorchcomms.so"
).resolve()
MCCL_LIBRARY = installed_file("mccl", "mccl/lib/libmccl.so").resolve()
BUILD_INFO = identity.create_companion_build_info(
    package_version=PACKAGE_VERSION,
    torchcomms_source_revision=required_environment("TORCHCOMMS_SOURCE_REVISION"),
    torchcomms_source_tree_sha256=required_environment("TORCHCOMMS_SOURCE_TREE_SHA256"),
    mccl_source_revision=required_environment("MCCL_SOURCE_REVISION"),
    torchcomms_core_wheel_sha256=required_environment("TORCHCOMMS_CORE_WHEEL_SHA256"),
    mccl_wheel_sha256=required_environment("MCCL_WHEEL_SHA256"),
    runtime_closure_decision_sha256=os.environ.get(
        "MCCL_RUNTIME_CLOSURE_DECISION_SHA256"
    ),
)


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class IsolatedBuild(build_orig):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.build_base = str(python_build_directory("build"))


class IsolatedEggInfo(egg_info_orig):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.egg_base = str(python_build_directory("metadata"))


class IsolatedBdistWheel(bdist_wheel_orig):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.bdist_dir = str(python_build_directory("wheel"))


class BuildPy(build_py_orig):
    def run(self) -> None:
        super().run()
        build_root = pathlib.Path(self.build_lib)
        package_root = build_root / "torchcomms" / "mccl"
        package_root.mkdir(parents=True, exist_ok=True)
        # The native extension belongs to the parent package, so its stub does too.
        (package_root / "_comms_mccl.pyi").unlink(missing_ok=True)
        (package_root / "_build_info.json").write_text(
            json.dumps(BUILD_INFO, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(
            BACKEND_ROOT / "_comms_mccl.pyi",
            build_root / "torchcomms" / "_comms_mccl.pyi",
        )


class CMakeBuild(build_ext_orig):
    def run(self) -> None:
        identity.require_c10d_torchcomms_factory()
        self.build_cmake()

    def build_cmake(self) -> None:
        build_temp = pathlib.Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)
        extension_path = pathlib.Path(
            self.get_ext_fullpath("torchcomms._comms_mccl")
        ).absolute()
        extension_path.parent.mkdir(parents=True, exist_ok=True)
        pybind11_include_root = get_torch_pybind11_include_root(build_temp)

        cmake_args = [
            "-G",
            "Ninja",
            f"-DCMAKE_BUILD_TYPE={os.environ.get('CMAKE_BUILD_TYPE', 'Release')}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path};{TORCHCOMMS_DEPS_ROOT}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DTORCHCOMMS_SOURCE_ROOT={TORCHCOMMS_SOURCE_ROOT}",
            f"-DMCCL_SOURCE_ROOT={MCCL_SOURCE_ROOT}",
            f"-DTORCHCOMMS_DEPS_ROOT={TORCHCOMMS_DEPS_ROOT}",
            f"-DTORCHCOMMS_LIBRARY={TORCHCOMMS_LIBRARY}",
            f"-DMCCL_LIBRARY={MCCL_LIBRARY}",
            f"-DTORCHCOMMS_MCCL_OUTPUT_DIRECTORY={extension_path.parent}",
            f"-DTORCHCOMMS_PYBIND11_INCLUDE_DIR={pybind11_include_root}",
        ]
        cuda_math_include = os.environ.get("CUDA_MATH_INCLUDE", "").strip()
        if cuda_math_include:
            cmake_args.append(f"-DCUDA_MATH_INCLUDE={cuda_math_include}")
        cmake_args.extend(shlex.split(os.environ.get("CMAKE_ARGS", "")))
        self.spawn(
            ["cmake", "-S", str(PACKAGING_ROOT), "-B", str(build_temp)] + cmake_args
        )
        if self.dry_run:
            return
        parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "").strip()
        build_command = [
            "cmake",
            "--build",
            str(build_temp),
            "--target",
            "torchcomms_comms_mccl",
        ]
        if parallel:
            if not parallel.isdigit() or int(parallel) < 1:
                raise RuntimeError(
                    "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer"
                )
            build_command.extend(["--parallel", parallel])
        self.spawn(build_command)
        self.prepare_extension(extension_path, build_temp)

    def prepare_extension(
        self, extension_path: pathlib.Path, build_temp: pathlib.Path
    ) -> None:
        for executable in ("nm", "patchelf", "strip"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"{executable} is required to build this wheel")
        if not extension_path.is_file():
            raise RuntimeError(f"Expected extension was not produced: {extension_path}")

        self.spawn(
            [
                "patchelf",
                "--set-rpath",
                "$ORIGIN:$ORIGIN/../mccl/lib:$ORIGIN/../torch/lib",
                str(extension_path),
            ]
        )
        self.spawn(["strip", "--strip-debug", str(extension_path)])
        undefined_symbols = subprocess.check_output(
            ["nm", "-D", "--undefined-only", str(extension_path)], text=True
        ).splitlines()
        unresolved_fmt_symbols = sorted(
            fields[-1]
            for line in undefined_symbols
            if (fields := line.split()) and fields[-1].startswith("_ZN3fmt")
        )
        if unresolved_fmt_symbols:
            raise RuntimeError(
                "The companion extension has unresolved fmt symbols: "
                + ", ".join(unresolved_fmt_symbols)
            )
        needed = set(
            subprocess.check_output(
                ["patchelf", "--print-needed", str(extension_path)], text=True
            ).splitlines()
        )
        required = {"libmccl.so", "libtorchcomms.so"}
        if not required.issubset(needed):
            raise RuntimeError(
                "The companion extension must link installed libmccl and libtorchcomms"
            )
        forbidden = sorted(
            library
            for library in needed
            if library.startswith("libctran") or library.startswith("libobservatory")
        )
        if forbidden:
            raise RuntimeError(
                "The companion extension owns an unexpected runtime library: "
                + ", ".join(forbidden)
            )

        binary = extension_path.read_bytes()
        leaked_roots = [
            path
            for path in (
                TORCHCOMMS_SOURCE_ROOT,
                MCCL_SOURCE_ROOT,
                TORCHCOMMS_DEPS_ROOT,
                build_temp,
            )
            if os.fsencode(path) in binary
        ]
        if leaked_roots:
            raise RuntimeError(
                "The companion extension contains an absolute build path"
            )


setup(
    name="torchcomms-mccl",
    version=PACKAGE_VERSION,
    packages=["torchcomms.mccl"],
    package_dir={"torchcomms.mccl": ".."},
    package_data={"torchcomms.mccl": ["_build_info.json"]},
    entry_points={
        "torchcomms.backends": ["mccl = torchcomms.mccl"],
        "torch.distributed.backends": ["mccl = torchcomms.mccl:_register_c10d_backend"],
    },
    ext_modules=[CMakeExtension("torchcomms._comms_mccl")],
    cmdclass={
        "bdist_wheel": IsolatedBdistWheel,
        "build": IsolatedBuild,
        "build_ext": CMakeBuild,
        "build_py": BuildPy,
        "egg_info": IsolatedEggInfo,
    },
    install_requires=[
        f"torch=={BUILD_INFO['torch_version']}",
        f"torchcomms=={BUILD_INFO['torchcomms_version']}",
        f"mccl=={BUILD_INFO['mccl_version']}",
    ],
    zip_safe=False,
)
