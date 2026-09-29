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
| `HipHostCompat.h` | Host | `meta::comms::DeviceBuffer` and `meta::comms::CudaEvent` HIP-backed substitutes; `cudaEvent_t` typedef. Mirrors the NVIDIA `comms::utils` interface so test runners and benchmarks need no rewrites. |
| `HipDeviceCompat.h` | Device | `__trap()` mapped to `abort()` for the HIP device pass; small device-side substitutes (warp size, etc.). Included transitively by device headers. |
| `DocaCompat.h` | Both | Translates every `doca_*` symbol used by `MultipeerIbgdaTransport.{h,cc}` and `P2pIbgdaTransportDevice.cuh` to its AMD `prims_amd_gda_*` counterpart. Pure forwarding; the underlying impls live in `prims_amd_gda/`. |

### The AMD-native `prims_amd_gda` impl

Under `comms/prims/transport/amd/prims_amd_gda/`:

- `PrimsAmdGdaDef.h` / `PrimsAmdGdaDev.h` / `PrimsAmdGdaOps.h` / `PrimsAmdGdaShared.h` —
  device-side `prims_amd_gda_*` API implementations (mlx5dv-direct WQE
  construction, HSA UAR mapping, etc.).
- `PrimsAmdGdaHost.{h,cc}` — host-side `prims_amd_gda_*` API: `prims_amd_gda_gpu_*`
  context, `prims_amd_gda_verbs_*` QP/CQ creation and modification (with full
  IBV_QP_* mask translation), HSA dmabuf export, `ibv_reg_*` wrappers.
- `PrimsAmdGdaDmaBuf.{h,cc}` — dma-buf export helpers used by the host API.

### NIC backends

Under `comms/prims/transport/amd/nic/`, one subdirectory per RDMA NIC family
plus two shared headers:

- `NicConfig.h`, `NicSelector.h` — NIC-selection helpers; `NicSelector.h`
  typedefs the active backend off the `-DNIC_*` flag.
- `mlx5/Mlx5Hsi.h`, `mlx5/Mlx5NicBackend.h`
- `bnxt/BnxtHsi.h`, `bnxt/BnxtNicBackend.h`, `bnxt/BnxtReDv.h`
- `ionic/IonicHsi.h`, `ionic/IonicNicBackend.h`, `ionic/IonicReDv.h`,
  `ionic/IonicGidDiscovery.h`

Each carries the hardware-specific WQE layouts used by the `prims_amd_gda_*`
device API. Exactly one is compiled in per build — see "NIC backend selection"
below.

### NIC backend selection

`transport/amd/nic_config.bzl` reads `-c hpc_comms.nic={mlx5,bnxt,ionic}`
(default `bnxt`) at parse time and exports `NIC_DEFINE_AMD` (`-DNIC_MLX5` /
`-DNIC_BNXT` / `-DNIC_IONIC`) plus the matching `rdma-core` dep. The three host
impls define overlapping symbols, so they cannot co-exist in one binary; the
flag picks which `#ifdef` block of `PrimsAmdGdaHost.cc` compiles. Because the
choice happens at parse time, a default build proves nothing about the other
two backends — build all three when touching this code.

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

- `:hip_compat` — the `transport/amd/HipHostCompat.h` shim. Note it does *not*
  carry `HipDeviceCompat.h`; that ships with the `:prims_amd_gda*` targets below.
- `:doca_compat_amd` — `transport/amd/DocaCompat.h` device + host shim (plus
  `transport/amd/nic/ionic/IonicGidDiscovery.h`). Re-exports
  `:prims_amd_gda_device` and `:prims_amd_gda_host` so consumers including
  `DocaCompat.h` get the underlying impls.

### AMD `prims_amd_gda` library targets in `comms/prims/transport/amd:`

- `:prims_amd_gda` / `:prims_amd_gda_device` — device-side AMD
  `prims_amd_gda_*` API plus the NIC backend headers and `HipDeviceCompat.h`
  (header-only; both targets export the same header set).
- `:prims_amd_gda_host` — host-side `PrimsAmdGdaHost.{h,cc}` and
  `PrimsAmdGdaDmaBuf.{h,cc}`. The `-D__HIP_PLATFORM_AMD__` flag is gated
  behind `select()` so the NV build pass produces an empty TU (the .cc/.h
  content is wrapped in `#ifdef __HIP_PLATFORM_AMD__`).

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
- **DOCA host APIs not yet in `prims_amd_gda::PrimsAmdGdaHost`**: anything beyond
  what `MultipeerIbgdaTransport.cc` uses.

## Adding new AMD support — recipe

1. Identify the NVIDIA target you want to build on AMD.
2. Read its source for NVIDIA-only includes / API calls. Decide whether to:
   - Wrap them in `#ifndef __HIP_PLATFORM_AMD__` (most common).
   - Add a new shim to `transport/amd/HipHostCompat.h` if multiple consumers
     need the same substitute type.
   - Add a new entry to `transport/amd/DocaCompat.h` if it's a new `doca_*`
     device symbol used by the IBGDA path.
   - Add a new function to `transport/amd/prims_amd_gda/PrimsAmdGdaHost.{h,cc}`
     if it's a new host-side DOCA call.
3. Convert the existing NVIDIA-only library target to a unified target
   using the `select()` pattern above. Drop any `disable_amd_ci = True`.
4. Build with `buck2 build @fbcode//mode/opt-amd-gpu //path/to:foo`.
5. Update [`status.md`](status.md) to reflect the new coverage.

## File organization

```
comms/prims/
├── BUCK                          unified targets — single name, select() for platform
├── core/                         shared sources (NVIDIA + AMD)
├── transport/                    one directory per transport
│   ├── ibgda/  ibrc/  ll/  ll128/  llx/  nvl/  rdma/  self/
│   └── amd/                      AMD-only shims and primitives
│       ├── BUCK
│       ├── nic_config.bzl        -c hpc_comms.nic=... → -DNIC_* + rdma-core dep
│       ├── HipHostCompat.h       DeviceBuffer / CudaEvent host shim
│       ├── HipDeviceCompat.h     __trap() / device-side shim
│       ├── DocaCompat.h          doca_* → prims_amd_gda_* translation (device + host)
│       ├── prims_amd_gda/        AMD-native prims_amd_gda_* impl
│       │   ├── PrimsAmdGdaDef.h
│       │   ├── PrimsAmdGdaDev.h
│       │   ├── PrimsAmdGdaDmaBuf.{h,cc}  dma-buf export helpers
│       │   ├── PrimsAmdGdaHost.{h,cc}    host-side QP / CQ / dmabuf / MR
│       │   ├── PrimsAmdGdaOps.h
│       │   └── PrimsAmdGdaShared.h
│       ├── nic/                  NicConfig.h, NicSelector.h + mlx5/ bnxt/ ionic/
│       └── docs/                 this file + status.md
├── collectives/{,ib,tests,benchmarks}/  unified collectives + tests
├── tests/                        unified tests (single targets via select)
└── benchmarks/                   unified benchmarks; carry disable_amd_ci = True, so
                                  AMD coverage is local-only — see status.md
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
| `clock64()` | `wall_clock64()` (semantically different — see `AbortCheck.cuh::gpu_clock64()`) |

NOT auto-rewritten (you must guard manually):

- `<cuda.h>`, `<cuda_runtime.h>` includes
- `cuMem*` driver API (no HIP equivalent for fabric handles)
- `meta::comms::DeviceBuffer` / `CudaEvent` (use HipHostCompat substitutes)
- NCCL types (`ncclComm_t`, `ncclResult_t`)
- DOCA types and APIs (use `DocaCompat.h`)
