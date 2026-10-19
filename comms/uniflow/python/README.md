# Building the UniFlow Python package

The distribution is `torchcomms-uniflow`; the import remains unchanged:

~~~python
import uniflow._core
~~~

All variants use the same API and package name.

| Variant | CMake options | GPU transports on supported hardware |
|---|---|---|
| NVIDIA core | `UNIFLOW_GPU_PLATFORM=CUDA` | RDMA |
| NVIDIA with TCP | CUDA, `UNIFLOW_BUILD_EXTENSIONS=ON`, `UNIFLOW_ENABLE_TCP=ON` | RDMA, TCP |
| AMD | HIP, `UNIFLOW_BUILD_EXTENSIONS=ON` | P2P/XGMI slot 0, RDMA, TCP |

NVIDIA NVLink transport is deleted. NVML topology discovery remains, and enum
values remain `NVLink=0`, `RDMA=1`, `TCP=2`. NVIDIA NVLink preferences fall
through to RDMA. Unsupported TCP preferences retain their failure behavior.
TCP control messaging is part of every core build.

## Requirements

- CMake 3.20 or newer and a C++20 compiler.
- CUDA 12.8 or newer for NVIDIA, or ROCm and hipify-perl for AMD.
- Python development headers matching the importing interpreter and pybind11.
- spdlog 1.15.2 or newer and a compatible fmt build.

Shared CUDA sources are hipified for AMD. Existing HIP guards and build
helpers remain. Set `UNIFLOW_GPU_PLATFORM` explicitly when both toolkits are
installed. `NONE` configurations provide software checks and no data transport.

## Source layout

Public extensions are nested at `comms/uniflow/extensions`. Inside fbsource,
set `UNIFLOW_EXTENSIONS_DIR` to
`fbcode/neteng/network_ai/uniflow_extensions`. Public includes use
`comms/uniflow/extensions/...` with namespace `uniflow`.

The core source distribution excludes extensions. Build extension-enabled
wheels from the full exported checkout. Extensions compile in tree into the
existing library and wheel; no separate wheel or feature import is needed.

## Build and install

System interpreters can use compatible spdlog, fmt and pybind11 packages.
Interpreters that ship their own C++ runtime use the existing header options:

~~~text
-DUNIFLOW_SPDLOG_INCLUDE_DIR=<spdlog>/include
-DUNIFLOW_FMT_INCLUDE_DIR=<fmt>/include
-DUNIFLOW_PYBIND11_INCLUDE_DIR=<pybind11>/include
~~~

The current bindings use unscoped enum formatting supported by fmt 9.x.
Header builds set `UNIFLOW_INSTALL_CPP=OFF`. Native C++ installation uses
versioned dependency packages.

NVIDIA core:

~~~bash
python -m pip wheel --no-deps ./comms/uniflow   -C cmake.define.UNIFLOW_GPU_PLATFORM=CUDA
~~~

NVIDIA with TCP:

~~~bash
python -m pip wheel --no-deps ./comms/uniflow   -C cmake.define.UNIFLOW_GPU_PLATFORM=CUDA   -C cmake.define.UNIFLOW_BUILD_EXTENSIONS=ON   -C cmake.define.UNIFLOW_ENABLE_TCP=ON
~~~

AMD:

~~~bash
python -m pip wheel --no-deps ./comms/uniflow   -C cmake.define.UNIFLOW_GPU_PLATFORM=HIP   -C cmake.define.UNIFLOW_BUILD_EXTENSIONS=ON   -C cmake.define.CMAKE_PREFIX_PATH=/opt/rocm
~~~

Install the selected wheel and verify its API:

~~~bash
python -m pip install <wheel-path>
python -c "import uniflow._core"
~~~

For a native build, use the same variant options:

~~~bash
cmake -S comms/uniflow -B build -DUNIFLOW_BUILD_PYTHON=ON   -DUNIFLOW_GPU_PLATFORM=CUDA
cmake --build build -j
PYTHONPATH=build/python python -c "import uniflow._core"
~~~

The wheel contains `_core`, `libuniflow.so`, `__init__.py` and `_core.pyi`.
`$ORIGIN` finds the library beside the extension. The host supplies the GPU
runtime. Interpreter ABI, platform, C++ runtime and recorded GPU sonames must
match; these wheels are not manylinux artifacts.

See [source ownership and release](../SOURCE_MOVE.md) for export and validation
requirements.
