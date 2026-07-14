# Uniflow AMD — Neutral Zones

This document records the **neutral zones** of uniflow: modules that are
platform-agnostic and must remain free of any GPU-vendor dependency (CUDA, HIP,
or NVML). Keeping these modules vendor-neutral is what lets the AMD backend be a
drop-in addition behind the capability seams (see `AMD_ROCM_IMPLEMENTATION_SUMMARY.md`)
rather than a cross-cutting `#ifdef` change.

## Frozen modules

| Zone | Targets (representative) | Why it is neutral |
|------|--------------------------|-------------------|
| Executor | `executor:event_base`, `executor:scoped_event_base_thread` | Pure C++/Linux async primitives (eventfd/poll); no device code |
| Controller | `controller:controller`, `controller:tcp-controller` | Control-plane connection setup over TCP only |
| Core | `:result`, `:segment` | Core abstractions and memory model (no device runtime) |
| Logging | `logging:logging` | Logging only |
| sysfs | `drivers/sysfs:sysfs-api` | Reads PCI topology from sysfs; no GPU runtime |
| ibverbs core verbs | `drivers/ibverbs:ibv-api`, `drivers/ibverbs:ibv-core` | libibverbs QP/CQ/MR primitives; NIC-only, no GPU runtime |

GPU-specific code lives **only** behind the seams in:

- `drivers/cuda/` — four targets forming the Phase 2 GPU runtime seam, all written
  once in CUDA (`cuda*`/`cu*`) and translated to HIP at build time via two
  mechanisms in `drivers/cuda/BUCK`:
  * `cuda-api` — runtime API seam (`CudaApi` class wrapping `cuda_runtime_api.h`:
    device management, host alloc, memcpy async/peer/batch, stream/event). Built
    with `oss_gpu_cpp_library` (= fbcode `gpu_cpp_library` + OSS guard) which
    hipifies sources **and headers**, handles accelerator selection, HIP toolchain,
    ROCm arch flags, and CI labels. `rename_cpp_to_hip` emits `.hip` on AMD;
    NVIDIA keeps original `.cpp`.
  * `cuda-device-adapter` — `DeviceAdapter` implementation for pinned host memory
    (`pinnedHostAlloc`, `pinnedHostFree`, `hostGetDevicePointer`), also via
    `oss_gpu_cpp_library` with the same hipify path as `cuda-api`.
  * `cuda-driver-api` — driver API seam (`CudaDriverApi` class wrapping `cuda.h` /
    cuMem VMM, stream write-value, dma-buf export). Still built with uniflow-local
    `hipify` rule (hipify-perl) + `hip_toolchain_override`, **not** `gpu_cpp_library`,
    due to symbol-translation blocker: fbcode gpu_cpp_library hipify renames
    `CU_STREAM_WRITE_VALUE_DEFAULT` → `hipStreamWriteValueDefault` and mishandles
    other driver symbols, breaking RDMA consumers (`CopyEngine`) that still use
    hipify-perl and expect CUDA spellings. Until RDMA moves to gpu_cpp_library,
    driver seam stays on hipify-perl to preserve CUDA spellings in exported header.
  * `cuda-topology-discovery` — `CudaTopologyDiscovery` backend wiring CUDA,
    NVML, ibverbs and sysfs into `TopologyDiscovery` interface; plain C++ library
    selecting GPU seam targets via deps.

  There is no separate hand-written HIP twin; AMD/ROCm is mechanical translation
  at build time. The unified `//comms/uniflow:uniflow` target builds for both
  NVIDIA (`@mode/opt`) and AMD (`@mode/opt-amd-gpu -m rocm70`).

- `drivers/nvml/` — topology/management seam (`NvmlApi` interface, `createNvmlApi()`
  factory). NVIDIA uses real NVML (`nvml-lazy`); AMD uses no-op stub returning
  empty topology (amdsmi backend is future work). Factory keeps neutral code free
  of direct NVML linkage.

`drivers/GpuRuntimeTypes.h` (`drivers:gpu-runtime-types`) is intentionally
vendor-**neutral** (opaque handles + `enum class MemcpyKind`, std-only) and is
therefore allowed anywhere; it is not a CUDA/HIP/NVML dependency.

The aggregate targets `:_core`, `:connection`, `:multi-transport`, and
`:uniflow` are **not** neutral zones: they wire up the transports and therefore
transitively reach the GPU seams. Only the leaf core abstractions (`:result`,
`:segment`) are frozen.

## Enforcement

The freeze is enforced by build-time dependency guards in `amd/BUCK`
(`neutral_zones.bzl`). Each frozen target gets a `check_dependencies_test` in
`blocklist` mode that fails if the target's transitive dependency closure ever
reaches a first-party GPU seam target **or an external GPU runtime library**:

```
# First-party seams
fbcode//comms/uniflow/drivers/cuda(/|:).*   # CUDA/HIP: cuda-api runtime seam,
                                            # cuda-driver-api driver seam,
                                            # cuda-device-adapter, cuda-topology-discovery
fbcode//comms/uniflow/drivers/nvml(/|:).*   # NVML / amdsmi topology seam (factory + stub)

# External GPU runtime libraries (the cuda-lazy / nvml-lazy / amdhip64-lazy
# external_deps). The CUDA/ROCm build toolchain (nvcc, cuda_path, rocm_path,
# clang_root, ...) is a parse-time dep of every C++ target and is NOT matched.
.*//third-party.*/cuda(/[^:]*)?:(cuda-lazy|cuda|nvml-lazy|nvml)
.*//third-party.*/rocm(/[^:]*)?:(amdhip64-lazy|amdhip64|amdsmi-lazy|amdsmi)
```

Run them with:

```bash
buck2 test @fbcode//mode/opt fbcode//comms/uniflow/amd:
```

If a change makes a neutral module depend on a GPU seam (directly or
transitively), the corresponding `neutral_zone_dep_guard_*` test fails. The fix
is almost always to move the GPU-touching code behind a seam interface rather
than to relax the guard.
