#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-3 license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os.path
import pathlib
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable

from packaging_utils import validate_core_dynamic_search_paths
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as build_ext_orig
from setuptools.command.build_py import build_py as build_py_orig

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
PROJECTION_MARKER = pathlib.Path(ROOT) / ".torchcomms_private_projection"
NATIVE_SOURCE_MANIFEST = "_native_source_manifest.json"
FORBIDDEN_RETAINED_PREFIXES = (
    b"/data/users/",
    b"/home/",
    b"/opt/conda/conda-bld",
    b"/tmp/torchcomms",
    b"/usr/local/src/conda",
)


def compiler_prefix_map_flags(
    mappings: Iterable[tuple[pathlib.Path, str]],
) -> list[str]:
    expanded: dict[pathlib.Path, str] = {}
    for source, destination in mappings:
        lexical = pathlib.Path(os.path.abspath(source))
        physical = lexical.resolve(strict=True)
        for candidate in (lexical, physical):
            if candidate == pathlib.Path("/"):
                continue
            prior = expanded.get(candidate)
            if prior is not None and prior != destination:
                raise ValueError(f"conflicting prefix mappings for {candidate}")
            expanded[candidate] = destination

    flags = []
    for source, destination in sorted(
        expanded.items(), key=lambda item: (len(str(item[0])), str(item[0]))
    ):
        flags.extend(
            (
                f"-ffile-prefix-map={source}={destination}",
                f"-fdebug-prefix-map={source}={destination}",
            )
        )
    return flags


def reproducible_compiler_flags(build_temp: pathlib.Path) -> list[str]:
    mappings = [
        (pathlib.Path(sys.prefix), "/python-env"),
        (pathlib.Path(ROOT), "/torchcomms-source"),
        (pathlib.Path(TORCH_ROOT), "/torch"),
        (build_temp, "/torchcomms-build"),
    ]
    for variable, destination in (
        ("CONDA_PREFIX", "/python-env"),
        ("CUDA_HOME", "/cuda"),
    ):
        value = os.environ.get(variable)
        if value:
            mappings.append((pathlib.Path(value), destination))
    return compiler_prefix_map_flags(mappings)


def retained_forbidden_prefixes(build_temp: pathlib.Path) -> tuple[bytes, ...]:
    if not PROJECTION_MARKER.is_file():
        return ()
    paths = [
        pathlib.Path(ROOT),
        build_temp,
        pathlib.Path(sys.prefix),
        pathlib.Path(TORCH_ROOT),
    ]
    for variable in ("CONDA_PREFIX", "CUDA_HOME"):
        value = os.environ.get(variable)
        if value:
            paths.append(pathlib.Path(value))
    prefixes = set(FORBIDDEN_RETAINED_PREFIXES)
    for path in paths:
        lexical = pathlib.Path(os.path.abspath(path))
        for candidate in (lexical, lexical.resolve(strict=True)):
            if candidate != pathlib.Path("/"):
                prefixes.add(str(candidate).encode())
    return tuple(sorted(prefixes, key=lambda prefix: (len(prefix), prefix)))


def validate_dynamic_search_paths(
    path: pathlib.Path, dynamic: str, strict: bool
) -> None:
    try:
        validate_core_dynamic_search_paths(dynamic, strict=strict)
    except ValueError as error:
        raise RuntimeError(f"{error}: {path}") from error


def validate_native_artifacts(
    root: pathlib.Path,
    forbidden_prefixes: tuple[bytes, ...],
    strict_dynamic_paths: bool,
) -> None:
    if not root.is_dir():
        raise RuntimeError(f"core native artifact directory is missing: {root}")
    elf_paths = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as candidate:
            data = candidate.read()
            if not data.startswith(b"\x7fELF"):
                continue
        retained = [prefix.decode() for prefix in forbidden_prefixes if prefix in data]
        if retained:
            raise RuntimeError(f"core native artifact retains paths: {retained}")
        elf_paths.append(path)
    if not elf_paths:
        raise RuntimeError(f"core build produced no ELF artifacts under {root}")
    command = os.environ.get("READELF", "readelf")
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(f"READELF executable was not found: {command}")
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    for path in elf_paths:
        dynamic = subprocess.check_output(
            [executable, "-dW", str(path)],
            env=environment,
            stderr=subprocess.STDOUT,
            text=True,
        )
        validate_dynamic_search_paths(path, dynamic, strict_dynamic_paths)


def cuda_toolkit_version(cuda_root: pathlib.Path) -> str:
    nvcc = (cuda_root / "bin/nvcc").resolve()
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        raise RuntimeError("CUDA_HOME must identify a toolkit with bin/nvcc")
    output = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", output)
    if match is None:
        raise RuntimeError("could not determine CUDA_HOME toolkit version")
    return match.group(1)


def validate_cuda_toolkit(cuda_root: pathlib.Path, exact: bool) -> None:
    toolkit_version = cuda_toolkit_version(cuda_root)
    torch_cuda = torch.version.cuda
    if torch_cuda is None:
        if exact:
            raise RuntimeError("private projections require CUDA-enabled PyTorch")
        return
    if toolkit_version.partition(".")[0] != torch_cuda.partition(".")[0]:
        raise RuntimeError(
            f"PyTorch CUDA {torch_cuda} and toolkit {toolkit_version} "
            "have different major versions"
        )
    if exact and toolkit_version != torch_cuda:
        raise RuntimeError(
            f"PyTorch CUDA {torch_cuda} does not exactly match private "
            f"projection toolkit {toolkit_version}"
        )


def validate_retained_build_environment() -> None:
    if PROJECTION_MARKER.is_symlink() or (
        PROJECTION_MARKER.exists() and not PROJECTION_MARKER.is_file()
    ):
        raise RuntimeError("private projection marker must be a regular file")
    if not PROJECTION_MARKER.is_file():
        return
    value = os.environ.get("CONDA_PREFIX")
    if not value:
        raise RuntimeError("private projections require CONDA_PREFIX")
    prefix = pathlib.Path(value).resolve(strict=True)
    for label, path in (
        ("Python executable", pathlib.Path(sys.executable)),
        ("Python prefix", pathlib.Path(sys.prefix)),
        ("PyTorch package", pathlib.Path(TORCH_ROOT)),
    ):
        if not path.resolve(strict=True).is_relative_to(prefix):
            raise RuntimeError(f"{label} is outside CONDA_PREFIX: {path}")
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        raise RuntimeError("private projections require CUDA_HOME")
    validate_cuda_toolkit(pathlib.Path(cuda_home).resolve(strict=True), exact=True)


validate_retained_build_environment()


def merged_flags(variable: str, generated: list[str]) -> str:
    values = [os.environ.get(variable, "").strip(), *generated]
    return " ".join(value for value in values if value)


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_source_records() -> dict[str, str]:
    root = pathlib.Path(ROOT).resolve()
    source_root = root / "comms/torchcomms"
    suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inc", ".inl"}
    records = {}
    for path in sorted(source_root.rglob("*")):
        if path.suffix not in suffixes:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(source_root) or not resolved.is_file():
            raise RuntimeError(f"native TorchComms source is unsafe: {path}")
        relative_source = path.relative_to(source_root)
        if "tests" in relative_source.parts or "examples" in relative_source.parts:
            continue
        if relative_source.parts[:1] == ("fb",):
            continue
        records[path.relative_to(root).as_posix()] = file_sha256(resolved)
    if not records:
        raise RuntimeError("no native TorchComms sources were found")
    return records


def native_source_manifest() -> dict[str, object]:
    records = native_source_records()
    digest = hashlib.sha256()
    for relative, recorded_hash in records.items():
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(recorded_hash.encode())
        digest.update(b"\0")
    return {"files": records, "schema": 1, "sha256": digest.hexdigest()}


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
USE_NCCL = flag_enabled("USE_NCCL", True)
USE_NCCLX = flag_enabled("USE_NCCLX", True)
USE_GLOO = flag_enabled("USE_GLOO", True)
USE_RCCL = flag_enabled("USE_RCCL", False)
USE_RCCLX = flag_enabled("USE_RCCLX", False)
USE_XCCL = flag_enabled("USE_XCCL", False)
IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
# Transport is CUDA-only; disable by default on ROCm but allow explicit opt-in.
USE_TRANSPORT = flag_enabled("USE_TRANSPORT", not IS_ROCM)
# Minimal RDMA CCA-hook extension. CUDA-only and requires the NCCLX static lib;
# default ON when NCCLX is built (and not ROCm).
USE_TRANSPORT_CCA_HOOK = flag_enabled(
    "USE_TRANSPORT_CCA_HOOK", USE_NCCLX and not IS_ROCM
)
USE_TRITON = flag_enabled("USE_TRITON", False)


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
        build_temp = pathlib.Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).absolute()

        prefix_map_flags = reproducible_compiler_flags(build_temp)
        build_flags = list(prefix_map_flags)
        if detect_hipify_v2():
            build_flags += ["-DHIPIFY_V2"]
        linker_flags = merged_flags("LDFLAGS", ["-Wl,--build-id=none"])
        pybind11_include_root = get_torch_pybind11_include_root(build_temp)

        cfg = os.environ.get("CMAKE_BUILD_TYPE", "RelWithDebInfo")
        print(f"- Building with {cfg} configuration")

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_PREFIX={extdir.parent.absolute()}",
            f"-DCMAKE_INSTALL_DIR={extdir.parent.absolute()}",
            f"-DCMAKE_PREFIX_PATH={TORCH_ROOT}",
            f"-DTORCHCOMMS_PYBIND11_INCLUDE_DIR={pybind11_include_root}",
            f"-DCMAKE_C_FLAGS={merged_flags('CFLAGS', prefix_map_flags)}",
            f"-DCMAKE_CXX_FLAGS={merged_flags('CXXFLAGS', build_flags)}",
            f"-DCMAKE_CUDA_FLAGS={merged_flags('CUDAFLAGS', [f'-Xcompiler={flag}' for flag in prefix_map_flags])}",
            f"-DCMAKE_SHARED_LINKER_FLAGS={linker_flags}",
            f"-DCMAKE_MODULE_LINKER_FLAGS={linker_flags}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DLIB_SUFFIX={os.environ.get('LIB_SUFFIX', 'lib')}",
            f"-DUSE_NCCL={flag_str(USE_NCCL)}",
            f"-DUSE_NCCLX={flag_str(USE_NCCLX)}",
            f"-DUSE_GLOO={flag_str(USE_GLOO)}",
            f"-DUSE_RCCL={flag_str(USE_RCCL)}",
            f"-DUSE_RCCLX={flag_str(USE_RCCLX)}",
            f"-DUSE_XCCL={flag_str(USE_XCCL)}",
            f"-DUSE_TRANSPORT={flag_str(USE_TRANSPORT)}",
            f"-DUSE_TRANSPORT_CCA_HOOK={flag_str(USE_TRANSPORT_CCA_HOOK)}",
            f"-DUSE_TRITON={flag_str(USE_TRITON)}",
        ]
        if PROJECTION_MARKER.is_file():
            cmake_args.append("-DCMAKE_SKIP_BUILD_RPATH=ON")
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            cuda_root = pathlib.Path(cuda_home).resolve(strict=True)
            nvcc = cuda_root / "bin/nvcc"
            if not nvcc.is_file():
                raise RuntimeError("CUDA_HOME must identify a toolkit with bin/nvcc")
            validate_cuda_toolkit(cuda_root, exact=PROJECTION_MARKER.is_file())
            cmake_args.extend(
                (
                    f"-DCMAKE_CUDA_COMPILER={nvcc}",
                    f"-DCUDAToolkit_ROOT={cuda_root}",
                    f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_root}",
                )
            )
        build_args = ["--", "-j"]

        os.chdir(str(build_temp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "--target", "install"] + build_args)
            if not extdir.is_file():
                raise RuntimeError(f"CMake did not produce the extension at {extdir}")
            validate_native_artifacts(
                extdir.parent.absolute(),
                retained_forbidden_prefixes(build_temp),
                strict_dynamic_paths=PROJECTION_MARKER.is_file(),
            )
            extdir.chmod(0o755)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


class build_py(build_py_orig):
    def run(self):
        super().run()
        destination = (
            pathlib.Path(self.build_lib) / "torchcomms" / NATIVE_SOURCE_MANIFEST
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(native_source_manifest(), sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        destination.chmod(0o644)


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
    cmdclass={"build_ext": build_ext, "build_py": build_py},
    install_requires=install_requires,
    extras_require=extras_require,
)
