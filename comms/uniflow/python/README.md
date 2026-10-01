# Building the `uniflow` Python package

`uniflow._core` is the pybind11 extension that the `uniflow` Python package
imports. Anything that uses the package needs it importable in every process
that will transfer. Check an environment with:

```bash
python -c "import uniflow._core"
```

This directory builds that extension. It is off by default; pass
`-DUNIFLOW_BUILD_PYTHON=ON`.

## Requirements

- CMake 3.20 or newer, and a C++20 compiler.
- A GPU toolkit: either the CUDA toolkit, or ROCm. The build detects which is
  present and reports it as `UNIFLOW_GPU_PLATFORM`; override with
  `-DUNIFLOW_GPU_PLATFORM=CUDA|HIP`. On ROCm the CUDA-API sources are translated
  with `hipify-perl`, which ships with ROCm, and the compiler must be
  `amdclang++`.
- Python development headers for the interpreter that will import the
  extension, and pybind11. pybind11 is taken from the environment when present
  and fetched otherwise.
- spdlog and fmt, supplied one of the two ways below. Supply them: with
  neither, the build falls back to fetching spdlog, and that fallback does not
  produce a working build.

## Choosing where spdlog, fmt and pybind11 come from

The extension is loaded into an interpreter, so its dependencies have to be
loadable next to that interpreter. Pick whichever case describes yours:

- **The interpreter uses the distribution's C++ runtime** -- a system `python3`,
  or a virtual environment built on one. Install `spdlog` and `fmt` as packages
  and let the build find them. Note the extension then keeps a dependency on
  those shared libraries, so the same packages have to be present wherever it is
  installed.

- **The interpreter ships its own C++ runtime**, as relocatable toolchains and
  some vendored Python builds do. Linking the distribution's shared `spdlog` and
  `fmt` then mixes two C libraries in one process and the import fails. Point
  the build at header trees instead, which leaves the extension depending on
  nothing but the GPU runtime:

  ```
  -DUNIFLOW_SPDLOG_INCLUDE_DIR=<spdlog>/include
  -DUNIFLOW_FMT_INCLUDE_DIR=<fmt>/include
  -DUNIFLOW_PYBIND11_INCLUDE_DIR=<pybind11>/include
  ```

  `fmt` must be a release that still formats unscoped enums implicitly; 9.x
  does, 10 and later do not.

Confirm which way it resolved by checking the extension's dependencies:
`ldd` on the built `_core*.so` should list the GPU runtime and `libuniflow.so`,
and, in the header-only case, neither spdlog nor fmt.

## Build

`<repo>` throughout is the directory holding `comms/`; the build takes its
include root from there, or from `$ROOT` when that is set. Pass the interpreter
that will import the extension as `Python3_EXECUTABLE`: the extension is tagged
with that interpreter's ABI and no other can load it.

Collect the settings once. This is the ROCm, header-only case -- the one that
applies when the target interpreter ships its own C++ runtime:

```bash
export ROCM_PATH=/opt/rocm
export UNIFLOW_CMAKE_FLAGS="\
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DUNIFLOW_BUILD_PYTHON=ON \
-DPython3_EXECUTABLE=$(command -v python3) \
-DCMAKE_PREFIX_PATH=$ROCM_PATH \
-DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/amdclang++ \
-DUNIFLOW_SPDLOG_INCLUDE_DIR=<repo>/third-party/spdlog/include \
-DUNIFLOW_FMT_INCLUDE_DIR=<repo>/third-party/fmt/9.1.0/fmt/include \
-DUNIFLOW_PYBIND11_INCLUDE_DIR=<repo>/third-party/pybind11/2.13.6/include"
```

On CUDA, drop the two ROCm lines. With a distribution interpreter, drop the
three header-tree lines and install spdlog and fmt as packages instead; the
section above says which case you are in.

```bash
cmake -S <repo>/comms/uniflow -B build $UNIFLOW_CMAKE_FLAGS
cmake --build build -j
```

The build stages an importable package at `build/python/uniflow`, so it can be
used without installing:

```bash
PYTHONPATH=build/python python -c "import uniflow._core"
```

## Install as a package

`pyproject.toml` in the parent directory builds the same extension into an
installable distribution, which is what puts `uniflow._core` on a consumer's
import path without a manual build tree.

`CMAKE_ARGS` carries the same settings to the build backend, so reuse what the
Build section collected:

```bash
CMAKE_ARGS="$UNIFLOW_CMAKE_FLAGS" pip install <repo>/comms/uniflow
python -c "import uniflow._core"
```

or, to keep the artifact -- note `--wheel`, which builds in place:

```bash
CMAKE_ARGS="$UNIFLOW_CMAKE_FLAGS" python -m build --wheel <repo>/comms/uniflow
pip install <repo>/comms/uniflow/dist/torchcomms_uniflow-*.whl
```

Plain `python -m build`, without `--wheel`, builds the wheel from an sdist and
fails: sources include each other as `comms/uniflow/...`, and an unpacked sdist
has no such prefix above it. Build in place, or from a checkout.

Without `CMAKE_ARGS` the build falls back to whatever CMake discovers, which on
a ROCm host is usually not what you want.

The distribution is named `torchcomms-uniflow` because `uniflow` on PyPI is an
unrelated project. The import name is unaffected: the package it installs is
`uniflow`, so consumers still write `import uniflow._core`.

The wheel is tagged for the interpreter and platform it was built on, and it
carries `libuniflow.so` beside the extension with an `$ORIGIN` rpath. It is not
a manylinux wheel: it resolves the GPU runtime from the host at import, so it is
portable only to hosts with a compatible one. Producing a redistributable wheel
is a separate release step.

## Install the build tree

```bash
cmake --install build --prefix <prefix>
```

This writes `uniflow/` into `<prefix>`, holding the extension, `__init__.py`,
`_core.pyi` and `libuniflow.so`. The extension finds the library beside itself,
so the directory can be copied into a virtual environment's `site-packages`, or
`<prefix>` can be put on `PYTHONPATH`.

## Runtime compatibility

The extension does not bundle the GPU runtime; it records the soname it was
linked against and resolves it from the host at import. That soname, not a
release number, is the compatibility boundary, and it is visible in the
artifact:

```bash
readelf -d <prefix>/uniflow/_core*.so | grep NEEDED
```

A build against the ROCm 7 series records `libamdhip64.so.7`, so it loads on a
host whose HIP runtime provides that soname and needs a rebuild on one that does
not. The same holds for the C++ runtime: the extension is compiled with the host
toolchain and keeps a `libstdc++.so.6` dependency, which is the other half of
why this is not a portable wheel.

What has been exercised: built with ROCm 7.0.2.1 (AMD clang 20) and imported on
MI350X (gfx950) hosts running that same ROCm, including one whose PyTorch was
built for ROCm 7.2 -- a 7.0-linked extension and a 7.2-linked PyTorch coexist in
one process there. Nothing on the CUDA side has been exercised; that path builds
the same way but is untested.
