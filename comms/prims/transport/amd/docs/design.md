# AMD support in `comms/prims/` — design

This document describes how `comms/prims/` supports AMD GPUs (HIP/ROCm). For
the current per-component coverage matrix, see [`status.md`](status.md).

## Goal

**Single source of truth.** Every transport, collective, and test in
`comms/prims/` builds from the same `.h` / `.cc` / `.cuh` / `.cu` sources
under both nvcc (NVIDIA / CUDA / DOCA) and hipcc (AMD / HIP / RCCL). No
per-platform forks of the host or device code.

## Architecture: thin compile-time shim layer

The unified sources call NVIDIA APIs (`doca_*` for IBGDA, `cuMem*` for
fabric memory, `meta::comms::DeviceBuffer` for RAII buffers, `cudaEvent_t`
for timing). On AMD, three shim headers bridge those calls to the AMD
equivalents, plus a small set of `#ifdef __HIP_PLATFORM_AMD__` guards
around code paths that have no AMD equivalent.

### The three shim headers

All under `comms/prims/transport/amd/`. Together they replace the legacy parallel
AMD-only sources (`MultipeerIbgdaTransportAmd.{h,cu}`, etc.).

| Header | Layer | Purpose |
|---|---|---|
| `amd/HipHostCompat.h` | Host | `meta::comms::DeviceBuffer` and `meta::comms::CudaEvent` HIP-backed substitutes; `cudaEvent_t` typedef. Mirrors the NVIDIA `comms::utils` interface so test runners and benchmarks need no rewrites. |
| `amd/HipDeviceCompat.h` | Device | `__trap()` mapped to `abort()` for the HIP device pass; small device-side substitutes (warp size, etc.). Included transitively by device headers. |
| `amd/DocaCompat.h` | Both | Translates every `doca_*` symbol used by `MultipeerIbgdaTransport.{h,cc}` and `P2pIbgdaTransportDevice.cuh` to its AMD `pipes_gda_*` counterpart. Pure forwarding; the underlying impls live in `amd/pipes_gda/`. |

### The AMD-native `pipes_gda` impl

Under `comms/prims/transport/amd/pipes_gda/`:

- `PipesGdaDef.h` / `PipesGdaDev.h` / `PipesGdaOps.h` / `PipesGdaShared.h` /
  `PipesGdaUtils.h` — device-side `pipes_gda_*` API implementations
  (mlx5dv-direct WQE construction, HSA UAR mapping, etc.).
- `PipesGdaHost.{h,cc}` — host-side `pipes_gda_*` API: `pipes_gda_gpu_*`
  context, `pipes_gda_verbs_*` QP/CQ creation and modification (with full
  IBV_QP_* mask translation), HSA dmabuf export, `ibv_reg_*` wrappers.

### NIC backend (mlx5)

Under `comms/prims/transport/amd/nic/`:

- `Mlx5Hsi.h`, `Mlx5NicBackend.h`, `NicConfig.h`, `NicSelector.h` —
  hardware-specific WQE layouts and NIC-selection helpers used by the
  `pipes_gda_*` device API.

## BUCK conventions

### Single-target `select()` pattern

For libraries, tests, and any target that needs to build on both NVIDIA
and AMD: define **one** target whose platform-specific bits are routed
through `select()` on `ovr_config//gpu:amd`. Avoid sibling `*_amd` /
`*_amd_unified` target naming (legacy from the pre-unification era).

```python
gpu_cpp_library(
    name = "multi_peer_nvl_transport",
    srcs = ["MultiPeerNvlTransport.cc"],
    headers = ["MultiPeerNvlTransport.h"],
    compiler_flags = select({
        "DEFAULT": [],
        "ovr_config//gpu:amd": ["-D__HIP_PLATFORM_AMD__"],
    }),
    cuda_exported_external_deps = [("cuda", None, "cuda-lazy")],
    hip_exported_external_deps = [("rocm", None, "amdhip64-lazy")],
    deps = [...] + select({
        "DEFAULT": ["//comms/utils:cuda_raii"],
        "ovr_config//gpu:amd": ["//comms/prims:hip_compat"],
    }),
)
```

For tests using `comms_gpu_cpp_distributed_unittest`, the same
`select()` pattern applies — drop `disable_amd_ci` / `disable_nvidia_ci`
and let CI dispatch run the test under whichever build mode it picks.

### AMD-only support targets in `comms/prims:`

These exist only on AMD (no NVIDIA counterpart needed):

- `:hip_compat` — `amd/HipHostCompat.h` + `amd/HipDeviceCompat.h` shims.
- `:doca_compat_amd` — `amd/DocaCompat.h` device + host shim. Re-exports
  `:pipes_gda_device` and `:pipes_gda_host` so consumers including
  `DocaCompat.h` get the underlying impls.

### AMD `pipes_gda` library targets in `comms/prims/amd:`

- `:pipes_gda_device` — device-side AMD `pipes_gda_*` API (header-only).
- `:pipes_gda_host` — host-side `PipesGdaHost.{h,cc}`. The
  `-D__HIP_PLATFORM_AMD__` flag is gated behind `select()` so the NV
  build pass produces an empty TU (the .cc/.h content is wrapped in
  `#ifdef __HIP_PLATFORM_AMD__`).

## Conditional compilation

Two preprocessor macros gate AMD code paths. The legacy `PIPES_AMD_BUILD`
is **retired** — use only `__HIP_PLATFORM_AMD__` going forward.

| Macro | Scope | When to use |
|---|---|---|
| `__HIP_PLATFORM_AMD__` | Translation unit — auto-defined by hipcc. Also explicitly added via `compiler_flags = select({"ovr_config//gpu:amd": ["-D__HIP_PLATFORM_AMD__"]})` on targets that compile a `.cc` (not `.cu`) for AMD. | Wrap NVIDIA-only `#include`s (`<cuda.h>`, `<cuda_runtime.h>`, `comms/utils/CudaRAII.h`, DOCA headers). Wrap declarations of NVIDIA-only types (`ncclComm_t`). |
| `__HIP_DEVICE_COMPILE__` | Function body — defined only during the device-compile pass under hipcc. | Combine with `__CUDA_ARCH__` to wrap device-only intrinsics: `#if defined(__CUDA_ARCH__) \|\| defined(__HIP_DEVICE_COMPILE__)`. |

## Source-code conventions

### Wrap NVIDIA-only includes

```cpp
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/amd/HipHostCompat.h"  // unconditional — provides DeviceBuffer/CudaEvent on AMD
#ifndef __HIP_PLATFORM_AMD__
#include "comms/utils/CudaRAII.h"  // NVIDIA-only — defines DeviceBuffer/CudaEvent here
#endif
```

`HipHostCompat.h` is safe to include on both platforms (its body is gated
on `__HIP_PLATFORM_AMD__`).

### Wrap NCCL-only code

```cpp
#ifndef __HIP_PLATFORM_AMD__
#include <nccl.h>
#endif

class FooFixture : public BenchmarkTestFixture {
#ifndef __HIP_PLATFORM_AMD__
  ncclComm_t ncclComm_{};
  void initNccl() { ... }
#endif
};
```

`comms/ncclx:nccl` does not currently compile cleanly under hipcc (its
generated NCCL sources include `<cuda_runtime.h>` directly, which collides
with `<hip/hip_runtime.h>` (`uint2`/`uint3` redefinition). On AMD,
benchmarks that compare against NCCL skip the baseline.

### Device code: combine the two device-pass macros

```cpp
__device__ __forceinline__ void my_kernel_helper(...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // device-only code (intrinsics, __trap(), shfl, etc.)
#endif
}
```

## What is NOT supported on AMD

- **CUDA driver API (`cuMem*`)**: fabric handles, POSIX-FD memory exports,
  `cuMemGetAddressRange` for MR caching. AMD path uses `hipHostMalloc` for
  sink buffers and treats the user pointer as the MR base.
- **NCCL baseline in benchmarks**: see "Wrap NCCL-only code".
- **DOCA host APIs not yet in `pipes_gda::PipesGdaHost`**: anything beyond
  what `MultipeerIbgdaTransport.cc` uses.

## Adding new AMD support — recipe

1. Identify the NVIDIA target you want to build on AMD.
2. Read its source for NVIDIA-only includes / API calls. Decide whether to:
   - Wrap them in `#ifndef __HIP_PLATFORM_AMD__` (most common).
   - Add a new shim to `amd/HipHostCompat.h` if multiple consumers need
     the same substitute type.
   - Add a new entry to `amd/DocaCompat.h` if it's a new `doca_*` device
     symbol used by the IBGDA path.
   - Add a new function to `amd/pipes_gda/PipesGdaHost.{h,cc}` if it's a
     new host-side DOCA call.
3. Convert the existing NVIDIA-only library target to a unified target
   using the `select()` pattern above. Drop any `disable_amd_ci = True`.
4. Build with `buck2 build @fbcode//mode/opt-amd-gpu //path/to:foo`.
5. Update [`status.md`](status.md) to reflect the new coverage.

## File organization

```
comms/prims/
├── *.{h,cc,cuh,cu}              shared sources (NVIDIA + AMD)
├── BUCK                          unified targets — single name, select() for platform
├── amd/                          AMD-only shims and primitives
│   ├── BUCK
│   ├── HipHostCompat.h           DeviceBuffer / CudaEvent host shim
│   ├── HipDeviceCompat.h         __trap() / device-side shim
│   ├── DocaCompat.h              doca_* → pipes_gda_* translation (device + host)
│   ├── pipes_gda/                AMD-native pipes_gda_* impl
│   │   ├── PipesGdaDef.h
│   │   ├── PipesGdaDev.h
│   │   ├── PipesGdaHost.{h,cc}    host-side QP / CQ / dmabuf / MR
│   │   ├── PipesGdaOps.h
│   │   ├── PipesGdaShared.h
│   │   └── PipesGdaUtils.h
│   ├── nic/                      NIC backends (Mlx5Hsi.h, Mlx5NicBackend.h, NicConfig.h, NicSelector.h)
│   └── docs/                     this file + status.md
├── collectives/{,tests,benchmarks}/  unified collectives + tests
├── tests/                        unified tests (single targets via select)
└── benchmarks/                   unified benchmarks (NV-only at runtime; AMD benchmark targets removed pending working AMD test fleet — see status.md)
```

## Reference: HIP API mapping

HIPify auto-rewrites these inside `.cu` files cross-compiled under hipcc.
You can write `cudaXxx` in unified sources and trust the rewrite.

| CUDA | HIP |
|---|---|
| `cudaMalloc` / `cudaFree` | `hipMalloc` / `hipFree` |
| `cudaMemcpy` / `cudaMemset` | `hipMemcpy` / `hipMemset` |
| `cudaSetDevice` / `cudaGetDevice` | `hipSetDevice` / `hipGetDevice` |
| `cudaStreamCreate` / `cudaStreamSynchronize` | `hipStreamCreate` / `hipStreamSynchronize` |
| `cudaEventCreate` / `cudaEventRecord` / `cudaEventElapsedTime` | `hipEventCreate` / `hipEventRecord` / `hipEventElapsedTime` |
| `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` | `hipIpcGetMemHandle` / `hipIpcOpenMemHandle` |
| `cudaError_t` / `cudaSuccess` | `hipError_t` / `hipSuccess` |
| `clock64()` | `wall_clock64()` (semantically different — see `Timeout.cuh::gpu_clock64()`) |

NOT auto-rewritten (you must guard manually):

- `<cuda.h>`, `<cuda_runtime.h>` includes
- `cuMem*` driver API (no HIP equivalent for fabric handles)
- `meta::comms::DeviceBuffer` / `CudaEvent` (use HipHostCompat substitutes)
- NCCL types (`ncclComm_t`, `ncclResult_t`)
- DOCA types and APIs (use `DocaCompat.h`)
