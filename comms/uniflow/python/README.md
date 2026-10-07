# Building the `uniflow` Python package

The distribution is named `torchcomms-uniflow` because the `uniflow` name on
PyPI belongs to another project. It installs the `uniflow` import package and
its compiled `_core` extension.

## Requirements

- Python 3.10 or newer.
- CMake 3.20 or newer, Ninja, and a C++20 compiler.
- The CUDA toolkit or ROCm for a functional GPU build.
- `hipify-perl` for a ROCm build.

The wheel build installs pybind11 into its isolated build environment. It also
builds its matching fmt and spdlog versions, so system copies are not required.

## Choose the GPU platform

`UNIFLOW_GPU_PLATFORM` accepts `AUTO`, `NONE`, `CUDA`, or `HIP`:

- `AUTO` selects the only GPU toolkit found and fails if both CUDA and ROCm are
  present.
- `CUDA` builds the NVIDIA backend.
- `HIP` builds the AMD backend and translates the CUDA-style implementation
  sources with `hipify-perl`.
- `NONE` builds a hardware-free package for portable build and API testing.

CUDA and HIP wheels expose the same Python API, public C++ headers, and
`uniflow::uniflow` CMake target. Their binaries and runtime dependencies differ,
so they must be built and distributed separately.

## Build a wheel from the source archive

Install the frontend and build a hardware-free wheel:

```bash
python -m pip install build
CMAKE_ARGS="-DUNIFLOW_GPU_PLATFORM=NONE" python -m build
```

This first creates a source archive and then builds the wheel from the extracted
archive. That is the preferred portability check because it has no TorchComms
parent tree.

For NVIDIA:

```bash
CMAKE_ARGS="-DUNIFLOW_GPU_PLATFORM=CUDA" python -m build
```

For AMD, select ROCm and its compiler explicitly when they are not on the
default search path:

```bash
export ROCM_PATH=/opt/rocm
CMAKE_ARGS="-DUNIFLOW_GPU_PLATFORM=HIP \
  -DCMAKE_PREFIX_PATH=${ROCM_PATH} \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc" \
  python -m build
```

The default wheel contains the Python module and the matching C++ SDK: shared
libraries, public headers, and CMake package files. Build a Python-only wheel
when the C++ SDK is not wanted:

```bash
CMAKE_ARGS="-DUNIFLOW_GPU_PLATFORM=NONE -DUNIFLOW_INSTALL_CPP=OFF" \
  python -m build --wheel
```

## Install and validate

```bash
python -m pip install dist/torchcomms_uniflow-*.whl
python -c "import uniflow; print(uniflow.__file__)"
python -m pip install pytest
python -m pytest --pyargs uniflow
```

`uniflow.cmake_prefix_path` points at the SDK inside an installed SDK wheel.
Pass it through `CMAKE_PREFIX_PATH` when another CMake extension calls
`find_package(uniflow CONFIG REQUIRED)`.

## Build directly with CMake

The Python extension is off in an ordinary CMake build. Enable it explicitly:

```bash
cmake -S . -B build -G Ninja \
  -DUNIFLOW_GPU_PLATFORM=NONE \
  -DUNIFLOW_BUILD_PYTHON=ON \
  -DUNIFLOW_BUILD_SHARED_LIBS=ON \
  -DUNIFLOW_INSTALL_CPP=OFF \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build
PYTHONPATH=build/python python -c "import uniflow._core"
```

## Runtime compatibility

The wheel never bundles GPU drivers, NVML, RDMA provider plugins, kernel
libraries, or device configuration. A CUDA or HIP wheel therefore requires a
compatible vendor runtime and driver on the target host. Inspect the native
dependencies with `readelf -d uniflow/_core*.so` or `ldd` after installation.
