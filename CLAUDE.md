# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

torchcomms is an experimental collective communications API for PyTorch, providing pluggable backends (NCCL, NCCLX, Gloo, RCCL, RCCLX, XCCL) for distributed GPU/CPU computing. It includes a compatibility layer (`distwrap`) that serves as a drop-in replacement for `torch.distributed`.

## Environment Setup

Prerequisites: Python 3.10+, CMake 3.22+, Ninja 1.10+, CUDA toolkit (for GPU backends).

### Using uv (preferred)

```bash
uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch nightly with CUDA 12.6
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126

# Other CUDA versions available:
# uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
# uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129
# uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# Install build and dev dependencies (setuptools/packaging needed for --no-build-isolation)
uv pip install setuptools packaging pyyaml
uv pip install pytest numpy psutil lintrunner parameterized pydot

# Install torchcomms from source (default: NCCL + NCCLX + Gloo + Transport)
uv pip install --no-build-isolation -v .
```

### Using conda

```bash
conda create -n torchcomms python=3.10
conda activate torchcomms

# Install PyTorch nightly with CUDA
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126

# Install required native deps (glog, gflags, fmt are needed for all builds; nccl for NCCL backend)
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt conda-forge::nccl -y

# Install torchcomms from source (USE_SYSTEM_LIBS=1 required with conda to link shared libs)
pip install -r requirements.txt
USE_NCCLX=OFF USE_SYSTEM_LIBS=1 pip install --no-build-isolation -v .
```

### Installing from PyPI (nightly wheels, no source build needed)

```bash
pip install --pre torch torchcomms --index-url https://download.pytorch.org/whl/nightly/cu126
```

## Building with Different Backends

Backend selection is controlled by environment variables (ON/OFF or 1/0) set before `pip install`:

| Variable | Default | Description |
|---|---|---|
| `USE_NCCL` | ON | Standard NCCL (uses PyTorch's bundled library, no extra build needed) |
| `USE_NCCLX` | ON | Meta's extended NCCL fork (built from vendored source in `comms/ncclx/`) |
| `USE_GLOO` | ON | CPU backend |
| `USE_RCCL` | OFF | AMD ROCm |
| `USE_RCCLX` | OFF | Meta's extended RCCL |
| `USE_XCCL` | OFF | Intel XPU |
| `USE_TRANSPORT` | ON (OFF on ROCm) | RDMA transport layer |
| `USE_SYSTEM_LIBS` | unset | When set, uses conda/system libs instead of building from source |

### NCCL-only (fastest build — skips NCCLX third-party dep compilation)

```bash
USE_NCCLX=OFF pip install --no-build-isolation -v .
```

### NCCLX build

`build_ncclx.sh` is the main build script. It builds ~20 third-party dependencies from source (fmt, zlib, boost, openssl, glog, gflags, folly, fbthrift, etc.), generates nccl_cvars files, builds the comms tracing service, and then compiles the vendored NCCL fork via `make`.

```bash
# Build everything from source (recommended for first time):
./build_ncclx.sh

# Or use conda/system libs for third-party deps:
USE_SYSTEM_LIBS=1 ./build_ncclx.sh

# Clean rebuild:
CLEAN_BUILD=1 ./build_ncclx.sh

# Skip third-party dep rebuild (if already built):
NCCL_BUILD_SKIP_DEPS=1 ./build_ncclx.sh
```

Key env vars for `build_ncclx.sh`:
- `CUDA_HOME` — CUDA installation (default: `/usr/local/cuda`)
- `NVCC_ARCH` — GPU architectures (default: `a100,h100`; auto-adds `b200` if CUDA 12.8+)
- `BUILDDIR` — build output directory (default: `build/ncclx`)
- `NCCL_HOME` — NCCLX source directory (default: `comms/ncclx/stable`)

### RCCL build (AMD ROCm)

```bash
export ROCM_HOME=/opt/rocm
export RCCL_INCLUDE=$ROCM_HOME/include/rccl
./build_rccl.sh
```

### RCCLX build (AMD ROCm, Meta extended)

```bash
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y
export BUILD_DIR=${PWD}/comms/rcclx/develop/build/release/build
export ROCM_HOME=/opt/rocm
export RCCLX_INCLUDE=${BUILD_DIR}/include/rccl
export RCCLX_LIB=${BUILD_DIR}/lib

# Narrow to your GPU arch to speed up build:
./build_rcclx.sh --amdgpu_targets gfx942  # MI300X/MI325X
./build_rcclx.sh --amdgpu_targets gfx950  # MI350X/MI355X
```

### XCCL build (Intel XPU)

```bash
source $INTEL_ONEAPI/compiler/latest/env/vars.sh
source $INTEL_ONEAPI/ccl/latest/env/vars.sh
USE_XCCL=ON USE_NCCL=OFF USE_NCCLX=OFF USE_TRANSPORT=OFF pip install --no-build-isolation -v .
```

### Install after backend build

For non-default backend combinations, disable the backends you don't need:

```bash
# Example: RCCLX-only
USE_NCCL=OFF USE_NCCLX=OFF USE_GLOO=OFF USE_RCCL=OFF USE_RCCLX=ON USE_TRANSPORT=OFF pip install --no-build-isolation -v .
```

### CMake direct build (C++ only, no Python package)

```bash
cmake -G Ninja -B build -DBUILD_TESTS=ON -DUSE_NCCL=ON -DUSE_NCCLX=OFF
cmake --build build
```

## Linting

Linting uses `lintrunner` configured in `.lintrunner.toml` with three linters:

- **CLANGFORMAT** — C++ formatting (`comms/torchcomms/**/*.{hpp,cpp}`) via `clang-format==21.1.2`
- **UFMT** — Python formatting (`comms/torchcomms/**/*.py`, `tools/**/*.py`) via `ufmt` (ruff-api + usort), invoked through `uv run --script` (requires `uv` to be installed)
- **PYRE** — Python type checking (`comms/torchcomms/**/*.py`)

```bash
# Initialize lintrunner (installs linter deps, first time only)
lintrunner init

# Run all linters on changed files
lintrunner

# Run and auto-fix formatting issues
lintrunner -a

# Run on specific files
lintrunner comms/torchcomms/some_file.py

# Run a specific linter only
lintrunner --take UFMT
lintrunner --take CLANGFORMAT
```

**When making changes, always run `lintrunner -a` before committing to auto-fix formatting.**

## Testing

### C++ unit tests (GoogleTest)

Enabled via `-DBUILD_TESTS=ON` cmake flag. Tests are in `comms/torchcomms/tests/unit/cpp/`.

```bash
# Full build-and-test from repo root
ctest --build-and-test ./ ./build --build-generator "Ninja" \
  --build-options -DBUILD_TESTS=ON \
  --test-command ctest --output-on-failure

# Or build then test separately
cmake -G Ninja -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure

# Run a single test suite by regex
ctest --test-dir build -R Factory --output-on-failure
ctest --test-dir build -R Options --output-on-failure

# Rebuild and rerun a specific test target
ninja -C build TorchCommFactoryTest && ctest --test-dir build -R Factory --output-on-failure
```

Available C++ test targets: `TorchCommFactoryTest`, `TorchCommOptionsTest`.

### Python unit tests (pytest)

Tests are in `comms/torchcomms/tests/unit/py/`.

```bash
# Run all Python unit tests
pytest comms/torchcomms/tests/unit/py/

# Run a single test file
pytest comms/torchcomms/tests/unit/py/test_factory.py

# Run a single test by name
pytest comms/torchcomms/tests/unit/py/test_factory.py -k "test_name"
```

### Integration tests (require multiple GPUs)

Tests are in `comms/torchcomms/tests/integration/py/`. Backend-specific integration tests live in their respective directories (e.g., `comms/torchcomms/ncclx/tests/`, `comms/torchcomms/distwrap/tests/`).

```bash
torchrun --nproc_per_node=N comms/torchcomms/tests/integration/py/<test_file>.py
```

### Performance benchmarks

Tests are in `comms/torchcomms/tests/perf/py/`.

```bash
torchrun --nproc_per_node=N comms/torchcomms/tests/perf/py/<benchmark>.py
```

**When making changes, always run the relevant unit tests and linting before considering the change complete.**

## Architecture

### Core Class Hierarchy (Strategy Pattern)

- **`TorchComm`** (`comms/torchcomms/TorchComm.hpp`) — Main user-facing class. Holds a `TorchCommBackend` and delegates all collective operations. Supports pre/post hooks. Non-thread-safe.
- **`TorchCommBackend`** (`comms/torchcomms/TorchCommBackend.hpp`) — Abstract base class that each backend implements. Pure virtual methods for every collective, point-to-point, and management operation.
- **`TorchWork`** (`comms/torchcomms/TorchWork.hpp`) — Async work handle using `c10::intrusive_ptr_target`. Concrete variants: `TorchWorkCompleted` (sync), `TorchWorkThread` (`std::async`-backed).
- **`TorchCommWindow`** (`comms/torchcomms/TorchCommWindow.hpp`) — One-sided RDMA/put operations with tensor registration and signal/wait primitives.

### Backend Loading

`TorchCommFactory` (singleton) maintains a registry of backend factories. Backends register either statically (compile-time linking) or dynamically (`dlopen` via `TORCHCOMMS_BACKEND_LIB_PATH_<BACKEND>` env var). Python-side discovery uses `entry_points` under the `torchcomms.backends` group. ABI version checking ensures compatibility (`TORCHCOMM_BACKEND_ABI_VERSION = "1.0"`).

### distwrap Compatibility Layer

`comms/torchcomms/distwrap/` provides drop-in replacements for `torch.distributed` functions. `BackendWrapper` (`BackendWrapper.cpp`) wraps `TorchComm` as a `c10d::Backend`, translating between the two APIs. Enabled when `torchcomms_is_enabled()` returns true; otherwise falls back to `torch.distributed`.

### Options Pattern

Every collective has its own Options class (e.g., `AllReduceOptions`, `SendOptions`) with a `hints` map for backend-specific configuration and a `timeout`. `CommOptions` adds `abort_process_on_timeout_or_error`, `high_priority_stream`, `store`.

### RDMA Transport

`comms/torchcomms/transport/` provides `RdmaTransport` for point-to-point RDMA operations using `folly::EventBase` for async I/O and `folly::SemiFuture` for results.

### torch.compile Support

`comms/torchcomms/functional/` enables compile-time graph representation of collectives, gated behind `TORCHCOMMS_PATCH_FOR_COMPILE=1` and PyTorch >= 2.12.

## Build System Details

- C++20 standard throughout. Default CMake build type is `RelWithDebInfo`.
- `build_ncclx.sh` handles third-party dependency building (cloning and compiling ~20 libraries into `$CONDA_PREFIX`), CVars generation, comms tracing service compilation, and the NCCLX make build.
- `setup.py` uses a custom `CMakeExtension` that invokes CMake from setuptools.
- `TORCH_CUDA_ARCH_LIST` env var controls CUDA architectures (defaults to `9.0`).

## Code Style

- **C++**: 2-space indentation, 80-char lines, enforced by `.clang-format` (run via `lintrunner`).
- **Python**: formatted with `ufmt` (ruff-api + usort), 88-char lines, enforced via `lintrunner`.

## Development Workflow

When making feature changes:
1. Update or add relevant documentation (docstrings, README, API docs) alongside code changes.
2. Run `lintrunner -a` to auto-fix formatting.
3. Run the relevant unit tests (`pytest` for Python, `ctest` for C++).
4. For changes to collectives or backends, run the relevant integration tests with `torchrun`.
