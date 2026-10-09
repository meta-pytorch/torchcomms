#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-3 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import sysconfig
from typing import Any


REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _command_output(command: list[str]) -> list[str]:
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _tool_identity(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    lines = _command_output([executable, "--version"])
    return lines[0] if lines else None


def _cuda_identity() -> str | None:
    cuda_home = os.environ.get("CUDA_HOME", "").strip()
    nvcc = pathlib.Path(cuda_home) / "bin" / "nvcc" if cuda_home else None
    if nvcc is None or not nvcc.is_file():
        resolved = shutil.which("nvcc")
        nvcc = pathlib.Path(resolved) if resolved is not None else None
    if nvcc is None:
        return None
    lines = _command_output([str(nvcc), "--version"])
    release = next((line for line in lines if "release " in line), None)
    return release or (lines[0] if lines else None)


def _git_output(root: pathlib.Path, *arguments: str) -> str | None:
    lines = _command_output(["git", "-C", str(root), *arguments])
    return lines[0] if lines else None


def _git_dirty(root: pathlib.Path) -> bool | None:
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(output.strip())


def source_identity(root: pathlib.Path) -> tuple[str | None, bool | None]:
    requested = os.environ.get("TORCHCOMMS_SOURCE_REVISION", "").strip().lower()
    if requested and REVISION_PATTERN.fullmatch(requested) is None:
        raise RuntimeError(
            "TORCHCOMMS_SOURCE_REVISION must be a full lowercase revision"
        )

    if not (root / ".git").exists():
        return requested or None, None

    actual = _git_output(root, "rev-parse", "HEAD")
    if actual is None or REVISION_PATTERN.fullmatch(actual) is None:
        if not requested:
            return None, None
        raise RuntimeError("Could not determine the TorchComms source revision")
    dirty = _git_dirty(root)
    if requested and requested != actual:
        raise RuntimeError(
            "TORCHCOMMS_SOURCE_REVISION does not match the source checkout"
        )
    if requested and dirty is None:
        raise RuntimeError("Could not determine whether the source checkout is clean")
    if requested and dirty:
        raise RuntimeError(
            "TORCHCOMMS_SOURCE_REVISION requires a clean tracked checkout"
        )
    return requested or actual, dirty


def dependency_prefix_digest() -> str | None:
    digest = os.environ.get("TORCHCOMMS_DEPS_PREFIX_DIGEST", "").strip().lower()
    if not digest:
        return None
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise RuntimeError("TORCHCOMMS_DEPS_PREFIX_DIGEST must be a lowercase SHA-256")
    return digest


def source_tree_sha256() -> str | None:
    digest = os.environ.get("TORCHCOMMS_SOURCE_TREE_SHA256", "").strip().lower()
    if not digest:
        return None
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise RuntimeError("TORCHCOMMS_SOURCE_TREE_SHA256 must be a lowercase SHA-256")
    return digest


def ncclx_identity(root: pathlib.Path, enabled: bool) -> str | None:
    if not enabled:
        return None
    stable = root / "comms" / "ncclx" / "stable"
    if not stable.is_symlink():
        raise RuntimeError(f"NCCLX selector is not a symlink: {stable}")
    identity = os.readlink(stable)
    if pathlib.PurePosixPath(identity).name != identity:
        raise RuntimeError(f"Unexpected NCCLX selector: {identity}")
    return identity


def parse_soname(dynamic_section: str) -> str | None:
    match = re.search(r"\(SONAME\).*\[([^]]+)]", dynamic_section)
    return match.group(1) if match is not None else None


def observatory_soname(package_root: pathlib.Path, required: bool) -> str | None:
    libraries = sorted(package_root.glob("libobservatory.so*"))
    if not libraries:
        if required:
            raise RuntimeError("The core wheel did not install libobservatory.so")
        return None
    readelf = shutil.which("readelf")
    if readelf is None:
        if required:
            raise RuntimeError("readelf is required to record the Observatory SONAME")
        return None
    for library in libraries:
        if library.is_symlink() or not library.is_file():
            continue
        try:
            dynamic = subprocess.check_output(
                [readelf, "--dynamic", "--wide", str(library)],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        soname = parse_soname(dynamic)
        if soname is not None:
            return soname
    if required:
        raise RuntimeError("Could not determine the Observatory SONAME")
    return None


def build_information(
    *,
    root: pathlib.Path,
    package_root: pathlib.Path,
    package_version: str,
    pytorch_version: str,
    pytorch_cxx11_abi: bool,
    torchcomms_revision: str | None,
    source_dirty: bool | None,
    use_ncclx: bool,
    bundle_observatory: bool,
    enabled_backends: list[str],
) -> dict[str, Any]:
    use_system_libs = os.environ.get("USE_SYSTEM_LIBS", "") in ("1", "ON")
    expects_observatory = use_ncclx and bundle_observatory and not use_system_libs
    return {
        "schema_version": 1,
        "package_version": package_version,
        "torchcomms_revision": torchcomms_revision,
        "torchcomms_source_tree_sha256": source_tree_sha256(),
        "source_dirty": source_dirty,
        "pytorch_version": pytorch_version,
        "python_version": sys.version.split()[0],
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "pytorch_cxx11_abi": pytorch_cxx11_abi,
        "c_compiler": _tool_identity(os.environ.get("CC", "cc")),
        "compiler": _tool_identity(os.environ.get("CXX", "c++")),
        "cmake": _tool_identity(os.environ.get("CMAKE_COMMAND", "cmake")),
        "ninja": _tool_identity(os.environ.get("NINJA_COMMAND", "ninja")),
        "strip": _tool_identity(os.environ.get("STRIP_EXECUTABLE", "strip")),
        "cuda_toolkit": _cuda_identity(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST") or None,
        "build_type": os.environ.get("CMAKE_BUILD_TYPE") or None,
        "enabled_backends": sorted(enabled_backends),
        "use_ncclx": use_ncclx,
        "use_system_libs": use_system_libs,
        "torchcomms_bundle_observatory": bundle_observatory,
        "ncclx_identity": ncclx_identity(root, use_ncclx),
        "observatory_soname": observatory_soname(package_root, expects_observatory),
        "torchcomms_deps_prefix_digest": dependency_prefix_digest(),
    }


def serialized_build_information(information: dict[str, Any]) -> str:
    return json.dumps(information, indent=2, sort_keys=True) + "\n"


def write_build_information(
    destination: pathlib.Path, information: dict[str, Any]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized_build_information(information))
