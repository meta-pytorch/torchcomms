# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


_CUDA_RUNTIME_REQUIREMENTS = {
    12: "nvidia-cuda-runtime-cu12>=12.8,<13",
    13: "nvidia-cuda-runtime>=13,<14",
}
_build_state = "metadata_wheel"


def build_state(state: str) -> None:
    global _build_state
    _build_state = state


def dynamic_wheel(_settings: Mapping[str, Any]) -> dict[str, bool]:
    return {"dependencies": True}


def _cmake_arguments() -> list[str]:
    try:
        return shlex.split(os.environ.get("CMAKE_ARGS", ""))
    except ValueError:
        return os.environ.get("CMAKE_ARGS", "").split()


def _cmake_value(name: str) -> str | None:
    prefixes = (f"-D{name}=", f"-D{name}:BOOL=", f"-D{name}:PATH=")
    for argument in reversed(_cmake_arguments()):
        for prefix in prefixes:
            if argument.startswith(prefix):
                return argument.removeprefix(prefix)
    return None


def _program_exists(name: str, root: str | None = None) -> bool:
    if root and Path(root, "bin", name).is_file():
        return True
    return shutil.which(name) is not None


def _cuda_enabled() -> bool:
    platform = (_cmake_value("UNIFLOW_GPU_PLATFORM") or "AUTO").upper()
    if platform in {"NONE", "HIP"}:
        return False
    if platform == "CUDA":
        return True
    value = os.environ.get("UNIFLOW_PACKAGE_ENABLE_CUDA")
    if value is not None:
        return value.upper() not in {"0", "FALSE", "NO", "OFF"}

    cuda_root = _cmake_value("CUDAToolkit_ROOT") or os.environ.get("CUDA_HOME")
    rocm_root = os.environ.get("ROCM_PATH")
    has_cuda = _program_exists("nvcc", cuda_root)
    has_hip = _program_exists("hipconfig", rocm_root) or _program_exists(
        "hipcc", rocm_root
    )
    if has_cuda and has_hip:
        raise RuntimeError(
            "Both CUDA and ROCm were found; set -DUNIFLOW_GPU_PLATFORM=CUDA or HIP"
        )
    return has_cuda


def _detected_cuda_version() -> tuple[int, int] | None:
    toolkit_root = _cmake_value("CUDAToolkit_ROOT") or os.environ.get("CUDA_HOME")
    nvcc = Path(toolkit_root, "bin", "nvcc") if toolkit_root else None
    if nvcc is None or not nvcc.is_file():
        found = shutil.which("nvcc")
        nvcc = Path(found) if found else None
    if nvcc is None:
        return None

    result = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout)
    if match is None:
        raise RuntimeError(f"Could not read the CUDA version from {nvcc}")
    return int(match.group(1)), int(match.group(2))


def _cuda_version() -> tuple[int, int]:
    configured = os.environ.get("UNIFLOW_CUDA_VERSION")
    match = re.match(r"^(\d+)\.(\d+)", configured or "")
    requested = (int(match.group(1)), int(match.group(2))) if match else None
    detected = _detected_cuda_version()
    if requested and detected and requested != detected:
        raise RuntimeError(
            f"This build requests CUDA {requested[0]}.{requested[1]} but nvcc "
            f"reports CUDA {detected[0]}.{detected[1]}"
        )
    if requested:
        return requested
    if detected:
        return detected
    raise RuntimeError(
        "A CUDA wheel requires UNIFLOW_CUDA_VERSION or an nvcc executable"
    )


def dynamic_metadata(
    _settings: Mapping[str, Any], _project: Mapping[str, Any]
) -> dict[str, list[str]]:
    if _build_state == "sdist" or not _cuda_enabled():
        return {"dependencies": []}

    major, minor = _cuda_version()
    if (major, minor) < (12, 8):
        raise RuntimeError("UniFlow requires CUDA 12.8 or newer")
    requirement = _CUDA_RUNTIME_REQUIREMENTS.get(major)
    if requirement is None:
        raise RuntimeError(f"UniFlow does not define a CUDA {major} runtime package")
    return {"dependencies": [f'{requirement}; platform_system == "Linux"']}
