# AMD ROCm/HIP Implementation Summary

## Overview

This document summarizes AMD ROCm/HIP support for the uniflow communication
library. uniflow's GPU layer is written **once in CUDA** (`cuda*`/`cu*`) and is
mechanically translated to HIP at build time; NVIDIA keeps compiling the original
sources. The **same `//comms/uniflow:uniflow` target builds for both NVIDIA and
AMD** — platform selection is the `ovr_config//gpu:amd` constraint (set by the
AMD build modes / modifier), not a separate library.

On AMD the GPU transport is **RDMA (RoCEv2) with GPUDirect**; the NVLink
transport is NVIDIA-only and is compiled out on AMD.

## Build-time translation (hipify-first)

There is no hand-written AMD twin of the GPU runtime. CUDA sources/headers are
translated to `.hip` and compiled with amdclang (`-x hip`); NVIDIA compiles
the original `.cpp`/`.h`. Four targets in `drivers/cuda/BUCK` form the Phase 2
GPU runtime seam, using two mechanisms:

- **`oss_gpu_cpp_library`** (in `comms/strict_oss_defs.bzl`) — the standard
  fbcode `gpu_cpp_library` plus uniflow's OSS transitive-dependency guard. Used
  for:
  * `cuda-api` — runtime API seam (`CudaApi` wrapping `cuda_runtime_api.h`:
    device management, host alloc/free, memcpy async/peer/batch, stream/event).
    Header declares raw `cuda*` types so it is hipified on AMD.
  * `cuda-device-adapter` — `DeviceAdapter` implementation for pinned host memory
    (`pinnedHostAlloc`, `pinnedHostFree`, `hostGetDevicePointer`). Header has no
    vendor types.
  It provides accelerator selection (CUDA/HIP/MTIA), hipify generation for
  sources **and** vendor-typed headers, HIP toolchain, ROCm arch flags, ROCm
  deps, and CI labels (`rename_cpp_to_hip`, `cuda_/hip_exported_external_deps`,
  `hip_preprocessor_flags` exporting `__HIP_PLATFORM_AMD__`).
- **uniflow-local `hipify` rule + `hip_toolchain_override`** (in
  `comms/uniflow/defs.bzl`) — used for:
  * `cuda-driver-api` — driver API seam (`CudaDriverApi` wrapping `cuda.h`:
    cuMem VMM, address reserve/map/unmap, set access, export/import shareable
    handle, dma-buf export, streamWriteValue64, device attributes). **Blocker:**
    fbcode `gpu_cpp_library` hipify translates some CUDA Driver-API symbols
    differently than `hipify-perl` — e.g. renames `CU_STREAM_WRITE_VALUE_DEFAULT`
    → `hipStreamWriteValueDefault`, maps `cuGetErrorName` to wrong-arity
    `hipGetErrorName`, leaves `cuStreamWriteValue64` and
    `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` untranslated.
    `CudaDriverApi.h` is consumed by RDMA `CopyEngine` still on hipify-perl,
    so driver seam stays on hipify-perl to preserve CUDA spellings until RDMA
    consumers migrate to `gpu_cpp_library`.
- **Plain `oss_cpp_library`** — used for:
  * `cuda-topology-discovery` — `CudaTopologyDiscovery` backend wiring `CudaApi`,
    `NvmlApi` factory, `IbvApi`, and `SysfsApi` into `TopologyDiscovery`
    interface. Selects GPU seam targets via deps, no vendor types in its own
    header.

`__HIP_PLATFORM_AMD__` selects the few platform-divergent code paths that hipify
cannot translate (e.g. int→`CUdeviceptr` cast via
`drivers/cuda/CudaDevicePtr.h::toDevicePtr`, dma-buf handle type aliases in
`CudaDriverApi.h`).

## Implementation stack (Phabricator)

Ordered from base to top matching the stacked diff series:

| Diff | Summary |
|------|---------|
| D107220750 | GPU runtime seam via `oss_gpu_cpp_library` + OSS ROCm allowlist. Adds `oss_gpu_cpp_library` macro (=`gpu_cpp_library`+OSS guard). Migrates `cuda-api` and `cuda-device-adapter` to `oss_gpu_cpp_library` with `rename_cpp_to_hip` and `cuda_/hip_exported_external_deps`; keeps `cuda-driver-api` on hipify-perl + `hip_toolchain_override` due to Driver-API symbol translation blocker (`CU_STREAM_WRITE_VALUE_DEFAULT`, `cuGetErrorName` arity, `cuStreamWriteValue64` untranslated). Retains `defs.bzl` hipify rule for driver seam and benchmarks. |
| D107220748 | NVML factory + AMD no-op stub + topology discovery. Adds `createNvmlApi()` factory selecting real `NvmlApi` on NVIDIA vs `NvmlApiStub` no-op on AMD (`ovr_config//gpu:amd`). `CudaTopologyDiscovery` constructs via factory, ordered after CUDA seam. No AMD amdsmi backend yet — stub returns empty topology. |
| D107220757 | AMD RDMA/GPUDirect + unified `:uniflow` target. Makes single `//comms/uniflow:uniflow` build on both NVIDIA and AMD via `ovr_config//gpu:amd` constraint select. Gates `NVLinkTransport` behind `#ifndef __HIP_PLATFORM_AMD__` in `MultiTransport.cpp` with BUCK deps select'd out. Enables AMD RDMA/GPUDirect in `transport/rdma` (CopyEngine VRAM path with `streamWriteValue64`, RdmaTransport, BUCK). Enables `uniflow_bench` on AMD. |
| D107346920 | Peer-to-peer transfer integration test. Platform-agnostic intra-node GPU P2P test over GPU interconnect (XGMI on AMD, NVLink on NVIDIA) via neutral `CudaApi::memcpyPeerAsync`. Allocates device memory so test exercises real device-to-device traffic; `hip_ci=True` enables AMD GPU CI. |
| D107220749 | AMD build documentation. Adds `AMD_BUILD.md` and initial `AMD_ROCM_IMPLEMENTATION_SUMMARY.md` describing unified target build, ROCm versions, modes, and troubleshooting. |
| D108381013 | Freeze neutral zones with GPU dep-guards. Adds `amd/neutral_zones.bzl` + `amd/BUCK` emitting per-target `check_dependencies_test` in blocklist mode for platform-agnostic modules (executor, controller, core result/segment, logging, sysfs, ibverbs core). Blocks first-party GPU seams (`drivers/cuda/*`: cuda-api, cuda-driver-api, cuda-device-adapter, cuda-topology-discovery; `drivers/nvml/*`) and external runtime libs (`cuda-lazy`, `nvml-lazy`, `amdhip64-lazy`, `amdsmi`). Documents architecture in `NEUTRAL_ZONES.md`. |
| D108675437 | RoCE GID auto-selection, netdev-prefix NIC selection, and MultiTransportFactoryOptions consolidation. Wires `ibv_query_gid_ex` via `IbvApi`/`IbvCore`/`MockIbvApi` for GID table introspection. `RdmaResources` auto-selects RoCEv2 GID skipping link-local (fe80::/10 IPv6 and 169.254 IPv4-mapped), preferring IPv4-mapped then first global. Configuration via `MultiTransportFactoryOptions` struct (NicFilter, netdevPrefix, gidIndex, trafficClass) — no library-internal env vars. Topology captures backing netdev name; netdev-prefix NIC selection defaults to `"beth"` with predicate skipped when netdev names unknown. Makes single-host RDMA test vendor-agnostic. Unit tests for GID selection and netdev-prefix. |
| D108942542 | Migrate landed uniflow GPU tests to `oss_gpu_cpp_unittest`. Migrates pre-existing GPU C++ test targets from `comms_gpu_cpp_unittest` to `oss_gpu_cpp_unittest` (=`comms_gpu_cpp_unittest` + OSS dep guard) for parity with production `oss_gpu_cpp_library` and CPU `oss_cpp_unittest`. AMD/HIP compile path, GPU CI labels, and RE config unchanged. Migrated: `drivers/cuda/tests`, `drivers/ibverbs/tests`, `drivers/nvml/tests`, `tests/integration`, `transport/tests/integration`. In-stack GPU test BUCKs folded into owning diffs (peer-to-peer test in D107346920, RDMA unit tests in D107220757/D108675437). Distributed rule left as-is. |

## Transports on AMD

### RDMA / GPUDirect (the AMD GPU transport)

RDMA is the GPU transport on AMD as well as NVIDIA (`transport/rdma/`):

- **GPUDirect RDMA** registers GPU (VRAM) memory with the NIC via a dma-buf fd
  exported with `cuMemGetHandleForAddressRange` (hipified to
  `hipMemGetHandleForAddressRange` on AMD).
- **RoCEv2 GID auto-selection** picks a valid RoCEv2 GID, with caller-supplied
  options (via MultiTransportFactoryOptions) and **netdev-prefix (`beth`) NIC selection** (D108675437).
- **GPU↔NIC PCIe affinity** (`selectGpuNics`) maps each GPU to its
  topologically-closest NIC, so per-GPU bandwidth tracks per-NIC bandwidth.
- The `CopyEngine` VRAM path uses `streamWriteValue64` for completion signaling.

Validated on MI350 (gfx950, ROCm 7.0) over Broadcom (bnxt) NICs with the
2-host `cross_host_test` and `rdma_bandwidth` benchmarks (DRAM + GPUDirect,
single-pair and 8-GPU-pair aggregate).

### NVLink (NVIDIA-only)

`NVLinkTransport` (NVML-backed topology, fabric/FD IPC) is NVIDIA-only and is
compiled out on AMD via `#ifndef __HIP_PLATFORM_AMD__` in `MultiTransport.cpp`,
with its BUCK deps `select()`'d out on AMD. Intra-node GPU-to-GPU P2P over the
GPU interconnect (XGMI on AMD, NVLink on NVIDIA) is exercised at the `CudaApi`
level by `PeerToPeerTransferTest` (`hipMemcpyPeerAsync` on AMD), but it is not a
registered uniflow transport on AMD.

## Build

```bash
# AMD (ROCm 7.0 recommended; gfx942 / gfx950 = MI300 / MI350)
buck build @fbcode//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow

# NVIDIA (same target, no GPU constraint)
buck build @fbcode//mode/opt fbcode//comms/uniflow:uniflow
```

See `AMD_BUILD.md` for ROCm versions, build modes, components, and
troubleshooting.

## Tests

- GPU C++ tests use **`oss_gpu_cpp_unittest`** (= `comms_gpu_cpp_unittest` +
  the OSS dependency guard), migrated in D108942542 for parity with production
  `oss_gpu_cpp_library` and CPU `oss_cpp_unittest`. AMD/HIP compile path, GPU
  CI labels, and RE config unchanged. CPU/neutral tests use `oss_cpp_unittest`.
- Migrated test targets:
  * `drivers/cuda/tests:cuda_device_adapter_test`, `:cuda_driver_api_test` — driver seam and device adapter via oss_gpu_cpp_library path
  * `drivers/ibverbs/tests:ibv_api_test` — ibverbs core verbs mocked
  * `drivers/nvml/tests:nvml_api_test` — NVML factory mocked, AMD stub covered
  * `transport/nvlink/tests/integration:peer_to_peer_transfer_test` — intra-node P2P (XGMI/NVLink) via CudaApi, hip_ci enabled (D107346920)
  * `transport/rdma/tests/integration:cross_host_test` — 2-host RDMA real NICs
  * `transport/rdma/tests/unit/*` — RDMA transport, slab pool, registration mocked; includes GID selection and netdev-prefix unit tests from D108675437
- Neutral zone dependency guards: `buck2 test @fbcode//mode/opt fbcode//comms/uniflow/amd:` runs `check_dependencies_test` per neutral target to ensure no GPU dep leakage (D108381013).

## Neutral zones

Platform-agnostic modules (executor, controller, core `:result`/`:segment`,
logging, sysfs, ibverbs core verbs) are kept free of any GPU-vendor dependency
and are enforced by `check_dependencies_test` guards in `amd/BUCK`
(see `NEUTRAL_ZONES.md` for full Phase 2 seam architecture). GPU code lives only
behind the four `drivers/cuda` seam targets (`cuda-api` runtime via
`oss_gpu_cpp_library`, `cuda-driver-api` via hipify-perl with symbol blocker,
`cuda-device-adapter` via `oss_gpu_cpp_library`, `cuda-topology-discovery` plain)
and `drivers/nvml` factory seam (`NvmlApi` interface with `createNvmlApi()`,
real NVML on NVIDIA, no-op stub on AMD).

## Known limitations / follow-ups

- `cuda-driver-api` remains on `hipify-perl` (see the blocker above); migrating
  it to `oss_gpu_cpp_library` requires moving the RDMA consumers
  (`transport/rdma`) to `gpu_cpp_library` so both ends hipify consistently.
- NVLink/XGMI is not exposed as a uniflow transport on AMD.
- `resolveGidIndex` readability: extract GID-table scan into a separate
  helper to de-duplicate the fallback logic (non-blocking, noted by reviewer).
- Forced GID index path does not bound `configuredGidIndex` to <= 255
  before `static_cast<uint8_t>` (follow-up to add the same guard as the auto path).

## References

- **rcclx:** `fbcode/comms/rcclx/` — HIP support patterns
- **pipes:** `fbcode/comms/pipes/` — platform abstraction patterns
- **HIP:** https://rocm.docs.amd.com/
