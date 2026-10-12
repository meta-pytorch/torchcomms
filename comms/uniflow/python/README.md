# Building the `uniflow` Python package

The distribution is named `torchcomms-uniflow`. Users import it with:

```python
import uniflow
```

The default wheel contains the Python API, type stubs, and matching native
libraries. Tests, benchmarks, and the C++ SDK are excluded. GPU builds contain
the transports enabled by their CMake settings.

## Requirements

- Python 3.10 or newer.
- CMake 3.20 or newer, Ninja, and a C++20 compiler.
- CUDA 12.8 or newer, or ROCm, for a GPU build.
- `hipify-perl` for a ROCm build.

The wheel build installs pybind11 in its build environment and builds the
matching fmt and spdlog libraries. System copies are not required.

## Build a wheel

Run these commands from the UniFlow source directory. To build for NVIDIA:

```bash
CMAKE_ARGS="-DUNIFLOW_GPU_PLATFORM=CUDA" python -m build
```

This creates a source archive and builds a wheel from that archive. Use
`--wheel` to build directly from the checkout instead.

For AMD, select ROCm and its compiler:

```bash
export ROCM_PATH=/opt/rocm
CMAKE_ARGS="-DUNIFLOW_GPU_PLATFORM=HIP \
  -DCMAKE_PREFIX_PATH=${ROCM_PATH} \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc" \
  python -m build
```

`UNIFLOW_GPU_PLATFORM=AUTO` selects the only GPU toolkit found. It fails if both
CUDA and ROCm are present. `NONE` builds a package without data transports for
build and API checks:

```bash
CMAKE_ARGS="-DUNIFLOW_GPU_PLATFORM=NONE" python -m build
```

Each wheel matches the Python version, CPU architecture, and GPU platform used
to build it. CUDA and HIP wheels use the same import name and must be installed
in separate environments.

## Install and check the package

Install the wheel that matches the target environment:

```bash
python -m pip install "<wheel-path>"
python -c "import uniflow; print(uniflow.__file__)"
```

## Optional tests and C++ SDK

To include Python tests in a development wheel:

```bash
CMAKE_ARGS="-DUNIFLOW_GPU_PLATFORM=NONE -DUNIFLOW_INSTALL_PYTHON_TESTS=ON" \
  python -m build
```

After installation, run `python -m pytest --pyargs uniflow.tests`. Install the
`test` extra for pytest. PyTorch and GPU tests skip when those dependencies are
unavailable. `UNIFLOW_BUILD_TESTS` controls C++ tests separately.

To include the C++ SDK, add `-DUNIFLOW_INSTALL_CPP=ON`. Its headers and CMake
package files are installed inside `uniflow`. Add that directory to
`CMAKE_PREFIX_PATH` when a C++ application uses
`find_package(uniflow CONFIG REQUIRED)`.

## Build directly with CMake

The Python extension is off in ordinary CMake builds. Enable it explicitly:

```bash
cmake -S . -B build -G Ninja \
  -DUNIFLOW_GPU_PLATFORM=NONE \
  -DUNIFLOW_BUILD_PYTHON=ON \
  -DUNIFLOW_BUILD_SHARED_LIBS=ON \
  -DUNIFLOW_INSTALL_CPP=OFF \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build
PYTHONPATH=build/python python -c "import uniflow"
```

## Runtime requirements

GPU drivers, GPU runtimes, RDMA providers, and device configuration come from
the host. GPU wheels need compatible versions on the target machine. Inspect
the installed native libraries with `readelf -d` or `ldd`. A local wheel build
does not by itself establish compatibility with every supported machine.
