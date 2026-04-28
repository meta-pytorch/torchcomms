#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-3 license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os.path
import pathlib
import shlex
import shutil
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


def _valid_cuda_home(path: str) -> bool:
    """Check that a CUDA directory has nvcc and cuda_runtime.h."""
    if not os.path.isfile(os.path.join(path, "bin", "nvcc")):
        return False
    # Generic layout (works on all architectures): include/cuda_runtime.h
    # x86_64 layout: targets/x86_64-linux/include/cuda_runtime.h
    # aarch64/sbsa layout: targets/sbsa-linux/include/cuda_runtime.h
    return (
        os.path.isfile(os.path.join(path, "include", "cuda_runtime.h"))
        or os.path.isfile(
            os.path.join(path, "targets", "x86_64-linux", "include", "cuda_runtime.h")
        )
        or os.path.isfile(
            os.path.join(path, "targets", "sbsa-linux", "include", "cuda_runtime.h")
        )
    )


def detect_cuda_home() -> str:
    """Auto-detect CUDA toolkit location.

    Checks CUDA_HOME / CUDA_PATH env vars first, then searches standard
    install locations (/usr/local/cuda, /usr/local/cuda-*).
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        if _valid_cuda_home(cuda_home):
            return cuda_home
        print(
            f"WARNING: CUDA_HOME={cuda_home} does not contain a complete CUDA "
            f"toolkit (missing nvcc or cuda_runtime.h). Attempting auto-detection..."
        )

    # Try /usr/local/cuda first, then search versioned directories (newest first)
    candidates = ["/usr/local/cuda"]
    versioned = sorted(
        glob.glob("/usr/local/cuda-*"),
        key=lambda p: [int(x) for x in p.split("cuda-")[-1].split(".") if x.isdigit()],
        reverse=True,
    )
    candidates.extend(versioned)

    for candidate in candidates:
        if _valid_cuda_home(candidate):
            return candidate

    return ""


def _valid_rocm_home(path: str) -> bool:
    """Check that a ROCm directory has hipcc and hip/hip_runtime.h."""
    if not os.path.isfile(os.path.join(path, "bin", "hipcc")):
        return False
    # Modern ROCm (5.x+): include/hip/hip_runtime.h
    # Older ROCm: hip/include/hip/hip_runtime.h
    return os.path.isfile(
        os.path.join(path, "include", "hip", "hip_runtime.h")
    ) or os.path.isfile(os.path.join(path, "hip", "include", "hip", "hip_runtime.h"))


def detect_rocm_home() -> str:
    """Auto-detect ROCm toolkit location.

    Checks ROCM_HOME / ROCM_PATH env vars first, then hipcc on PATH,
    then falls back to /opt/rocm.
    """
    rocm_home: str | None = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    if rocm_home:
        if _valid_rocm_home(rocm_home):
            return rocm_home
        print(
            f"WARNING: ROCM_HOME={rocm_home} does not contain a complete ROCm "
            f"toolkit (missing hipcc or hip_runtime.h). Attempting auto-detection..."
        )

    hipcc: str | None = shutil.which("hipcc")
    if hipcc:
        candidate: str = os.path.dirname(os.path.dirname(os.path.realpath(hipcc)))
        if _valid_rocm_home(candidate):
            return candidate

    if _valid_rocm_home("/opt/rocm"):
        return "/opt/rocm"

    return ""


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
if IS_ROCM:
    CUDA_HOME = ""
    os.environ.pop("CUDA_HOME", None)
    ROCM_HOME: str = detect_rocm_home()
    if ROCM_HOME:
        was_auto: bool = ROCM_HOME != (
            os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
        )
        os.environ["ROCM_HOME"] = ROCM_HOME
        label = " (auto-detected)" if was_auto else ""
        print(f"- ROCM_HOME={ROCM_HOME}{label}")
    else:
        print("- ROCM_HOME=<not found>")
else:
    ROCM_HOME = ""
    os.environ.pop("ROCM_HOME", None)
    CUDA_HOME: str = detect_cuda_home()
    if CUDA_HOME:
        was_auto = CUDA_HOME != (
            os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        )
        os.environ["CUDA_HOME"] = CUDA_HOME
        label = " (auto-detected)" if was_auto else ""
        print(f"- CUDA_HOME={CUDA_HOME}{label}")
    else:
        print("- CUDA_HOME=<not found>")

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

        cfg: str = os.environ.get("CMAKE_BUILD_TYPE", "RelWithDebInfo")
        print(f"- Building with {cfg} configuration")

        cmake_args: list[str] = [
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
        # Help CMake find the CUDA toolkit. CUDA_TOOLKIT_ROOT_DIR is used by
        # the legacy FindCUDA module (which PyTorch's Caffe2Config.cmake uses).
        # CMAKE_CUDA_COMPILER is used by modern CMake CUDA language support.
        if CUDA_HOME:
            cmake_args += [
                f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}",
                f"-DCMAKE_CUDA_COMPILER={os.path.join(CUDA_HOME, 'bin', 'nvcc')}",
            ]
        build_args: list[str] = ["--", "-j"]

        # Ensure nvcc is on PATH for CMake's CUDA compiler detection.
        if CUDA_HOME:
            nvcc_dir: str = os.path.join(CUDA_HOME, "bin")
            path_dirs: list[str] = os.environ.get("PATH", "").split(os.pathsep)
            if nvcc_dir not in path_dirs:
                os.environ["PATH"] = nvcc_dir + os.pathsep + os.environ.get("PATH", "")

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
if USE_RCCLX:
    ext_modules += [
        CMakeExtension("torchcomms._comms_rcclx"),
    ]
if USE_XCCL:
    ext_modules += [
        CMakeExtension("torchcomms._comms_xccl"),
    ]
if USE_TRANSPORT:
    ext_modules += [
        CMakeExtension("torchcomms._transport"),
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
        "torchcomms.backends": [
            "nccl = torchcomms._comms_nccl",
            "ncclx = torchcomms._comms_ncclx",
            "gloo = torchcomms._comms_gloo",
            "rccl = torchcomms._comms_rccl",
            "rcclx = torchcomms._comms_rcclx",
            "xccl = torchcomms._comms_xccl",
            "dummy = torchcomms._comms",
        ]
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=install_requires,
    extras_require=extras_require,
)
