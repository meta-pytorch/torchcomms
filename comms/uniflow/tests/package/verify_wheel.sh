#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -euo pipefail
shopt -s nullglob

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd -- "${script_dir}/../.." && pwd)"
work_dir=
rc=
for sig in 1 2 3 13 15; do
  eval "trap 'exit $((sig + 128))' ${sig}"
done
trap 'rc=$?; set +e; rm -rf -- "${work_dir}"; exit $rc' EXIT
work_dir="$(mktemp -d)" || exit 1

python="${PYTHON:-python3}"
package_cuda="${UNIFLOW_PACKAGE_ENABLE_CUDA:-OFF}"
if [[ "${package_cuda}" != ON && "${package_cuda}" != OFF ]]; then
  echo "UNIFLOW_PACKAGE_ENABLE_CUDA must be ON or OFF" >&2
  exit 1
fi
if [[ "${package_cuda}" == ON ]]; then
  package_gpu_platform=CUDA
else
  package_gpu_platform=NONE
fi
package_gpu_cmake_args=("-DUNIFLOW_GPU_PLATFORM=${package_gpu_platform}")
package_cmake_args="${CMAKE_ARGS:-} ${package_gpu_cmake_args[*]}"
package_test_env=(env -u LD_LIBRARY_PATH -u PYTHONPATH)

build_consumer() {
  local prefix="$1"
  local name="$2"
  local build_dir="${work_dir}/${name}-consumer"

  CMAKE_PREFIX_PATH="${prefix}" cmake \
    -S "${source_dir}/tests/cmake/consumer" \
    -B "${build_dir}" \
    -G Ninja
  CMAKE_PREFIX_PATH="${prefix}" cmake --build "${build_dir}"
  "${build_dir}/uniflow_consumer"
}

build_cpp_sdk() {
  local name="$1"
  local shared="$2"
  local build_dir="${work_dir}/${name}-build"
  local prefix="${work_dir}/${name}-install"

  cmake \
    -S "${unpacked_source}" \
    -B "${build_dir}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="${prefix}" \
    -DUNIFLOW_BUILD_BENCHMARKS=OFF \
    -DUNIFLOW_BUILD_PYTHON=OFF \
    -DUNIFLOW_BUILD_SHARED_LIBS="${shared}" \
    -DUNIFLOW_BUILD_TESTS=OFF \
    "${package_gpu_cmake_args[@]}" \
    -DUNIFLOW_INSTALL_CPP=ON \
    -DUNIFLOW_USE_SYSTEM_SPDLOG=OFF
  cmake --build "${build_dir}" --target install
}

create_venv() {
  local name="$1"
  local venv_dir="${work_dir}/${name}"

  "${python}" -m venv "${venv_dir}"
  if [[ "${OSTYPE}" == msys* || "${OSTYPE}" == cygwin* ]]; then
    echo "${venv_dir}/Scripts/python.exe"
  else
    echo "${venv_dir}/bin/python"
  fi
}

run_package_test() (
  cd -- "${work_dir}"
  "${package_test_env[@]}" "$@"
)

mkdir \
  "${work_dir}/python-only-wheel" \
  "${work_dir}/sdist" \
  "${work_dir}/unpacked" \
  "${work_dir}/wheel"

CMAKE_ARGS="${package_cmake_args}" "${python}" -m build \
  --sdist \
  --outdir "${work_dir}/sdist" \
  "${source_dir}"

sdists=("${work_dir}"/sdist/torchcomms_uniflow-*.tar.gz)
if [[ ${#sdists[@]} -ne 1 ]]; then
  echo "Expected one source distribution, found ${#sdists[@]}" >&2
  exit 1
fi

tar -xzf "${sdists[0]}" -C "${work_dir}/unpacked"
unpacked_sources=("${work_dir}"/unpacked/torchcomms_uniflow-*)
if [[ ${#unpacked_sources[@]} -ne 1 ]]; then
  echo "Expected one unpacked source directory, found ${#unpacked_sources[@]}" >&2
  exit 1
fi
unpacked_source="${unpacked_sources[0]}"

build_cpp_sdk static OFF
build_cpp_sdk shared ON

CMAKE_ARGS="${package_cmake_args} -DUNIFLOW_INSTALL_CPP=ON -DCMAKE_INSTALL_BINDIR=bin -DCMAKE_INSTALL_DATADIR=share -DCMAKE_INSTALL_INCLUDEDIR=include -DCMAKE_INSTALL_LIBDIR=lib" \
  "${python}" -m build \
  --wheel \
  --outdir "${work_dir}/wheel" \
  "${unpacked_source}"
CMAKE_ARGS="${package_cmake_args} -DUNIFLOW_INSTALL_CPP=OFF" \
  "${python}" -m build \
  --wheel \
  --outdir "${work_dir}/python-only-wheel" \
  "${unpacked_source}"

mv "${unpacked_source}" "${work_dir}/source-hidden"
build_consumer "${work_dir}/static-install" static
build_consumer "${work_dir}/shared-install" shared

wheels=("${work_dir}"/wheel/torchcomms_uniflow-*.whl)
python_only_wheels=(
  "${work_dir}"/python-only-wheel/torchcomms_uniflow-*.whl)
if [[ ${#wheels[@]} -ne 1 || ${#python_only_wheels[@]} -ne 1 ]]; then
  echo "Expected one SDK wheel and one Python-only wheel" >&2
  exit 1
fi

"${python}" - "${sdists[0]}" "${wheels[0]}" \
  "${python_only_wheels[0]}" "${source_dir}" <<'PY'
import sys
import tarfile
from pathlib import Path, PurePosixPath
import zipfile

sdist, wheel, python_only_wheel, source_dir = sys.argv[1:]
expected_version = (Path(source_dir) / "VERSION").read_text().strip()
portable_tests = {
    "tests/py/__init__.py",
    "tests/py/test_public_api.py",
    "tests/py/test_uniflow.py",
}

assert PurePosixPath(sdist).name == (
    f"torchcomms_uniflow-{expected_version}.tar.gz"
)
with tarfile.open(sdist) as archive:
    found_tests = set()
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        relative = PurePosixPath(*path.parts[1:])
        parts = relative.parts
        assert not str(relative).startswith("fbcode/comms/uniflow/"), path
        assert ".claude" not in parts, path
        assert ".llms" not in parts, path
        assert "benchmarks" not in parts, path
        assert path.name != "BUCK", path
        assert path.suffix != ".bzl", path
        if "tests" in parts and member.isfile():
            assert str(relative) in portable_tests, path
            found_tests.add(str(relative))
    assert found_tests == portable_tests, found_tests


def inspect_wheel(filename: str, expect_sdk: bool) -> None:
    package_files = {
        PurePosixPath("__init__.py"),
        PurePosixPath("_build_config.py"),
        PurePosixPath("_core.pyi"),
        PurePosixPath("py.typed"),
        PurePosixPath("tests/__init__.py"),
        PurePosixPath("tests/test_public_api.py"),
        PurePosixPath("tests/test_uniflow.py"),
    }
    with zipfile.ZipFile(filename) as archive:
        names = set(archive.namelist())
        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_files) == 1, metadata_files
        metadata = archive.read(metadata_files[0]).decode()
        assert f"Version: {expected_version}\n" in metadata

        for name in names:
            path = PurePosixPath(name)
            root = path.parts[0]
            assert root == "uniflow" or root.endswith(".dist-info"), path
            assert "torchcomms" not in path.parts, path
            assert ".llms" not in path.parts, path
            assert path.name != "BUCK", path
            assert path.suffix != ".bzl", path

            if root == "uniflow" and not name.endswith("/"):
                relative = PurePosixPath(*path.parts[1:])
                is_extension = (
                    relative.parent == PurePosixPath(".")
                    and relative.name.startswith("_core")
                    and relative.suffix in {".pyd", ".so"}
                )
                is_header = str(relative).startswith("include/comms/uniflow/")
                is_spdlog_runtime = (
                    relative.parent == PurePosixPath(".")
                    and relative.name.startswith("libspdlog.so")
                )
                is_spdlog_file = is_spdlog_runtime or expect_sdk and (
                    str(relative).startswith("include/spdlog/")
                    or str(relative).startswith("lib/cmake/spdlog/")
                    or str(relative).startswith("lib64/cmake/spdlog/")
                    or relative.name == "libspdlog.a"
                    or relative.name.startswith("libspdlog.so")
                    or relative
                    in {
                        PurePosixPath("lib/pkgconfig/spdlog.pc"),
                        PurePosixPath("lib64/pkgconfig/spdlog.pc"),
                        PurePosixPath("share/licenses/spdlog/LICENSE"),
                    }
                )
                is_fmt_runtime = (
                    relative.parent == PurePosixPath(".")
                    and relative.name.startswith("libfmt.so")
                )
                is_fmt_file = is_fmt_runtime or expect_sdk and (
                    str(relative).startswith("include/fmt/")
                    or str(relative).startswith("lib/cmake/fmt/")
                    or str(relative).startswith("lib64/cmake/fmt/")
                    or relative.name.startswith("libfmt")
                    and relative.suffix == ".a"
                    or relative.name.startswith("libfmt.so")
                    or relative
                    in {
                        PurePosixPath("lib/pkgconfig/fmt.pc"),
                        PurePosixPath("lib64/pkgconfig/fmt.pc"),
                        PurePosixPath("share/licenses/fmt/LICENSE"),
                    }
                )
                is_library = (
                    relative.parent
                    in {
                        PurePosixPath("bin"),
                        PurePosixPath("lib"),
                        PurePosixPath("lib64"),
                    }
                    and relative.name.startswith(("libuniflow.", "uniflow."))
                )
                is_cmake_config = (
                    str(relative).startswith("lib/cmake/uniflow/")
                    or str(relative).startswith("lib64/cmake/uniflow/")
                )
                is_sdk_metadata = expect_sdk and str(relative).startswith(
                    "share/uniflow/"
                )
                assert (
                    relative in package_files
                    or is_extension
                    or is_header
                    or is_spdlog_file
                    or is_fmt_file
                    or is_library
                    or is_cmake_config
                    or is_sdk_metadata
                ), path

            is_native_library = (
                path.suffix in {".a", ".dll", ".dylib", ".lib", ".pyd", ".so"}
                or ".so." in path.name
            )
            if is_native_library:
                is_extension = (
                    path.parent == PurePosixPath("uniflow")
                    and path.name.startswith("_core")
                )
                is_uniflow_library = (
                    path.parent
                    in {
                        PurePosixPath("uniflow/bin"),
                        PurePosixPath("uniflow/lib"),
                        PurePosixPath("uniflow/lib64"),
                    }
                    and path.name.startswith(("libuniflow.", "uniflow."))
                )
                is_spdlog_library = (
                    path.name.startswith("libspdlog.so")
                    and (
                        path.parent == PurePosixPath("uniflow")
                        or expect_sdk
                        and path.parent
                        in {
                            PurePosixPath("uniflow/lib"),
                            PurePosixPath("uniflow/lib64"),
                        }
                    )
                    or expect_sdk
                    and path.parent
                    in {
                        PurePosixPath("uniflow/lib"),
                        PurePosixPath("uniflow/lib64"),
                    }
                    and path.name == "libspdlog.a"
                )
                is_fmt_library = (
                    path.name.startswith("libfmt.so")
                    and (
                        path.parent == PurePosixPath("uniflow")
                        or expect_sdk
                        and path.parent
                        in {
                            PurePosixPath("uniflow/lib"),
                            PurePosixPath("uniflow/lib64"),
                        }
                    )
                    or expect_sdk
                    and path.parent
                    in {
                        PurePosixPath("uniflow/lib"),
                        PurePosixPath("uniflow/lib64"),
                    }
                    and path.name.startswith("libfmt")
                    and path.suffix == ".a"
                )
                assert (
                    is_extension
                    or is_uniflow_library
                    or is_spdlog_library
                    or is_fmt_library
                ), path

            if not name.endswith("/"):
                contents = archive.read(name)
                for forbidden in (
                    source_dir,
                    "/data/users/",
                    "/home/",
                    "/mnt/gvfs/",
                    "/opt/facebook/",
                    "fbsource/",
                ):
                    assert forbidden.encode() not in contents, (path, forbidden)

    required = {
        "uniflow/__init__.py",
        "uniflow/_build_config.py",
        "uniflow/_core.pyi",
        "uniflow/py.typed",
        "uniflow/tests/__init__.py",
        "uniflow/tests/test_public_api.py",
        "uniflow/tests/test_uniflow.py",
    }
    assert required <= names, required - names
    assert any(
        name.startswith("uniflow/_core") and name.endswith((".so", ".pyd"))
        for name in names
    ), "native Python extension"

    dependency_library_directories = {
        PurePosixPath("uniflow/lib"),
        PurePosixPath("uniflow/lib64"),
    }
    if not expect_sdk:
        dependency_library_directories = {PurePosixPath("uniflow")}
    assert any(
        PurePosixPath(name).parent in dependency_library_directories
        and PurePosixPath(name).name.startswith("libfmt.so")
        for name in names
    ), "fmt runtime library"
    assert any(
        PurePosixPath(name).parent in dependency_library_directories
        and PurePosixPath(name).name.startswith("libspdlog.so")
        for name in names
    ), "spdlog runtime library"

    sdk_headers = {
        name for name in names if name.startswith("uniflow/include/comms/uniflow/")
    }
    sdk_configs = {
        name for name in names if "/cmake/uniflow/" in name
    }
    sdk_libraries = {
        name
        for name in names
        if PurePosixPath(name).name.startswith(("libuniflow.", "uniflow."))
    }
    if expect_sdk:
        assert "uniflow/include/comms/uniflow/Uniflow.h" in sdk_headers
        assert any(name.endswith("/uniflowConfig.cmake") for name in sdk_configs)
        assert sdk_libraries
    else:
        assert not sdk_headers
        assert not sdk_configs
        assert not sdk_libraries

inspect_wheel(wheel, expect_sdk=True)
inspect_wheel(python_only_wheel, expect_sdk=False)
PY

venv_python="$(create_venv wheel-venv)"
run_package_test "${venv_python}" -m pip install \
  --no-deps "${wheels[0]}"
run_package_test "${venv_python}" -m pip install "pytest>=7"
run_package_test "${venv_python}" -m pytest --pyargs uniflow

wheel_prefix="$(
  run_package_test "${venv_python}" -c \
    'import uniflow; print(uniflow.cmake_prefix_path)'
)"
build_consumer "${wheel_prefix}" wheel

python_only_venv_python="$(create_venv python-only-venv)"
run_package_test "${python_only_venv_python}" -m pip install \
  --no-deps "${python_only_wheels[0]}"
run_package_test "${python_only_venv_python}" -m pip install "pytest>=7"
run_package_test "${python_only_venv_python}" -m pytest --pyargs uniflow
