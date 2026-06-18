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
translated to `.hip` and compiled with amdclang (`-x hip`); NVIDIA compiles the
original `.cpp`/`.h`. Two mechanisms are in play, both in
`drivers/cuda/BUCK`:

- **`oss_gpu_cpp_library`** (in `comms/strict_oss_defs.bzl`) — the standard
  fbcode `gpu_cpp_library` plus uniflow's OSS transitive-dependency guard. Used
  for `cuda-api` (runtime) and `cuda-device-adapter`. It provides accelerator
  selection (CUDA/HIP/MTIA), hipify generation for sources **and** vendor-typed
  headers, the HIP toolchain, ROCm arch flags, ROCm deps, and CI labels
  (`rename_cpp_to_hip`, `cuda_/hip_exported_external_deps`).
- **uniflow-local `hipify` rule + `hip_toolchain_override`** (in
  `comms/uniflow/defs.bzl`) — used for `cuda-driver-api` only. **Blocker:** the
  fbcode `gpu_cpp_library` hipify tool translates some CUDA **Driver-API**
  symbols differently than `hipify-perl` (e.g. it renames
  `CU_STREAM_WRITE_VALUE_DEFAULT` → `hipStreamWriteValueDefault`, maps
  `cuGetErrorName` to the wrong-arity `hipGetErrorName`, and leaves
  `cuStreamWriteValue64` / `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`
  untranslated). `cuda-driver-api`'s exported `CudaDriverApi.h` is consumed by
  RDMA code that is still hipified by `hipify-perl`, so `cuda-driver-api` stays
  on `hipify-perl` (which preserves the CUDA spellings consumers expect) until
  those consumers also move to `gpu_cpp_library`.

`__HIP_PLATFORM_AMD__` selects the few platform-divergent code paths that hipify
cannot translate (e.g. the int→`CUdeviceptr` cast, shared via
`drivers/cuda/CudaDevicePtr.h::toDevicePtr`).

## Implementation stack (Phabricator)

| Diff | Summary |
|------|---------|
| D107220750 | GPU runtime seam via `oss_gpu_cpp_library` + OSS ROCm allowlist (`cuda-api`, `cuda-device-adapter`; `cuda-driver-api` on hipify-perl; `CudaDevicePtr.h`) |
| D107220748 | NVML factory + AMD no-op stub + topology discovery |
| D107220757 | AMD RDMA/GPUDirect + unified `:uniflow` target |
| D107346920 | Peer-to-peer transfer integration test |
| D107220749 | AMD build documentation |
| D108381013 | Freeze neutral zones with GPU dep-guards |
| D108675437 | RoCE GID auto-selection, env tunables, netdev-prefix NIC selection |
| D108942542 | Migrate landed uniflow GPU tests to `oss_gpu_cpp_unittest` |

## Transports on AMD

### RDMA / GPUDirect (the AMD GPU transport)

RDMA is the GPU transport on AMD as well as NVIDIA (`transport/rdma/`):

- **GPUDirect RDMA** registers GPU (VRAM) memory with the NIC via a dma-buf fd
  exported with `cuMemGetHandleForAddressRange` (hipified to
  `hipMemGetHandleForAddressRange` on AMD).
- **RoCEv2 GID auto-selection** picks a valid RoCEv2 GID, with `UNIFLOW_IB_*`
  env tunables and **netdev-prefix (`beth`) NIC selection** (D108675437).
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
  the OSS dependency guard), so the AMD/HIP compile path and GPU CI/RE config
  come from the standard GPU test macro. CPU/neutral tests use
  `oss_cpp_unittest`.
- `drivers/cuda/tests:cuda_device_adapter_test`, `:cuda_driver_api_test` — driver
  seam.
- `transport/rdma/tests/unit/*` — RDMA transport/slab-pool/registration (mocked).
- `transport/rdma/tests/integration:cross_host_test` — 2-host RDMA (real NICs).
- `transport/nvlink/tests/integration:peer_to_peer_transfer_test` — intra-node
  P2P (XGMI/NVLink).

## Neutral zones

Platform-agnostic modules (executor, controller, core `:result`/`:segment`,
logging, sysfs, ibverbs core verbs) are kept free of any GPU-vendor dependency
and are enforced by `check_dependencies_test` guards in `amd/BUCK`
(see `NEUTRAL_ZONES.md`). GPU code lives only behind the `drivers/cuda` and
`drivers/nvml` seams.

## Known limitations / follow-ups

- `cuda-driver-api` remains on `hipify-perl` (see the blocker above); migrating
  it to `oss_gpu_cpp_library` requires moving the RDMA consumers
  (`transport/rdma`) to `gpu_cpp_library` so both ends hipify consistently.
- NVLink/XGMI is not exposed as a uniflow transport on AMD.

## References

- **rcclx:** `fbcode/comms/rcclx/` — HIP support patterns
- **pipes:** `fbcode/comms/pipes/` — platform abstraction patterns
- **HIP:** https://rocm.docs.amd.com/
