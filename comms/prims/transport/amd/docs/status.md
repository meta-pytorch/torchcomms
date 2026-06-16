# AMD support in `comms/prims/` — status

This document tracks which `comms/prims/` library targets, tests, and
benchmarks build and run on AMD GPUs (HIP/ROCm). For the architecture and
recipes for adding new AMD support, see [`design.md`](design.md).

**Last updated:** 2026-05-27. After unification: single targets per
library / test, platform-specific bits routed via `select()` on
`ovr_config//gpu:amd`. Legacy `*_amd_unified` and `*_amd` sibling
targets retired. NIC backend on AMD is selected at parse time via
`-c hpc_comms.nic={mlx5,bnxt,ionic}` (default `bnxt`), wired by
`//comms/prims/transport/amd:nic_config.bzl`; the chosen backend swaps the
`PipesGdaHost.cc` `#ifdef NIC_*` blocks and the `rdma-core` dep
without forking the source tree.

## Legend

- ✅ **Supported** — single target builds under both `@mode/opt` (NVIDIA)
  and `@mode/opt-amd-gpu` (AMD).
- 🚧 **Partial** — builds on AMD but some functionality is gated off
  (e.g. NCCL baseline disabled, certain methods stubbed).
- ❌ **NVIDIA-only** — depends on NVIDIA-specific APIs (cuMem driver API,
  DOCA host functions not yet shimmed, NCCL, CUDA-only headers); AMD
  build mode is excluded via `disable_amd_ci = True`.
- — **N/A** — header-only or platform-agnostic; works on both without
  per-platform plumbing.

## Library targets (`comms/prims/`)

### Device transports & utilities

All device-side transports either compile directly under hipcc (HIPify
rewrites the simple cases) or pick up the AMD shims via
`comms/prims/transport/amd/HipDeviceCompat.h` (transitively included by
`Timeout.cuh`).

| Component | Target | AMD | Notes |
|---|---|---|---|
| `P2pSelfTransportDevice.cuh` | `:p2p_self_transport_device` | ✅ | Header-only. |
| `P2pNvlTransportDevice.cuh` | `:p2p_nvl_transport_device` | ✅ | Header-only; uses `:hip_compat`. |
| `P2pIbgdaTransportDevice.cuh` | `:p2p_ibgda_transport_device` | ✅ | Cross-compiled via the `:doca_compat_amd` shim. |
| `MultipeerIbgdaDeviceTransport.cuh` | `:multipeer_ibgda_device_transport` | ✅ | Header-only. |
| `Transport.cuh` | `:transport` | — | Header-only. |
| `IbgdaBuffer.h`, `Timeout.cuh`, `ThreadGroup.cuh`, `CopyUtils.cuh`, `DeviceCheck.cuh`, `DeviceSpan.cuh`, `BarrierState.cuh`, `SignalState.cuh` | various | — | Header-only. |

### Host-side transports

| Component | Target | AMD | Notes |
|---|---|---|---|
| `MultiPeerNvlTransport.{h,cc}` | `:multi_peer_nvl_transport` | ✅ | Single target with `select()`. |
| `MultiPeerTransport.{h,cc}` | `:multi_peer_transport` | 🚧 | cuMem fabric-handle paths stubbed (throws on AMD if invoked); intra-node NVL + inter-node IBGDA paths fully functional. |
| `MultipeerIbgdaTransport.{h,cc}` | `:multipeer_ibgda_transport` | ✅ | Single target with `select()`; routes through `:doca_compat_amd` (which re-exports `:pipes_gda_host`). |
| `MultipeerIbgdaTransportCuda.{cu,cuh}` | `:multipeer_ibgda_transport_cuda` | ✅ | Single target with `select()`. |
| `GpuMemHandler.{h,cc}` | `:gpu_mem_handler` | 🚧 | Fabric-mode methods (`allocateFabricMemory`, `exchangeFabricHandles`, `importFabricPeerMemory`, `cleanupFabric`) stubbed under `__HIP_PLATFORM_AMD__`; standard `hipIpc*` IPC path works. |
| `TopologyDiscovery.{h,cc}` | `:topology_discovery` | ✅ | Single target with `select()`. |

### NVIDIA-only host targets

These depend on `<cuda.h>`, the cuMem driver API, or DOCA host APIs not
present in `pipes_gda::PipesGdaHost`. They have `disable_amd_ci = True`.

| Component | Target | Why NVIDIA-only |
|---|---|---|
| `CudaDriverLazy.{h,cc}` | `:cuda_driver_lazy` | Wraps `cuMem*` driver API symbols loaded via `cudaGetDriverEntryPoint`. |
| `DocaHostUtils.h` | `:doca_host_utils` | NVIDIA-only DOCA helpers. |

### AMD-only support targets (`comms/prims:` and `comms/prims/amd:`)

These exist only on AMD; no NVIDIA counterpart by design.

| Target | Purpose |
|---|---|
| `:hip_compat` | `amd/HipHostCompat.h` + `amd/HipDeviceCompat.h` — `__trap()` + `meta::comms::DeviceBuffer`/`CudaEvent` HIP substitutes. |
| `:doca_compat_amd` | `amd/DocaCompat.h` — `doca_*` → `pipes_gda_*` translation header (device + host). Re-exports `:pipes_gda_device` and `:pipes_gda_host`. |
| `:pipes_gda` / `:pipes_gda_device` (in `comms/prims/amd:`) | AMD-native NIC backends (`amd/nic/*`) and device-side `pipes_gda_*` API (`amd/pipes_gda/PipesGda{Def,Dev,Ops,Shared,Utils}.h`). Backends: mlx5 (`Mlx5Hsi.h`, `Mlx5NicBackend.h`) and BNXT (`BnxtHsi.h`, `BnxtNicBackend.h`, `BnxtReDv.h`); selected at parse time via `-c hpc_comms.nic=...`. |
| `:pipes_gda_host` (in `comms/prims/amd:`) | Host-side `pipes_gda_*` API (`amd/pipes_gda/PipesGdaHost.{h,cc}`) — QP / CQ / IBV_QP_* mask translation, HSA dmabuf export with isDevicePointer + page-alignment guards, `ibv_reg_*` wrappers. Single `.cc` carries both mlx5 and BNXT host paths behind `#ifdef NIC_BNXT`. BNXT path goes through `bnxt_re_dv` direct verbs (dlopened via `SysIbv`), allocates SQ/CQ/RQ in GPU uncached memory + dma-buf, and carves the MSN table from the SQ tail. |

## Collectives (`comms/prims/collectives/`)

| Component | Target | AMD | Notes |
|---|---|---|---|
| `AllGather.cuh` | `:allgather` | ✅ | Single target with `select()`. |
| `AllToAllv.{h,cu,cuh}` | `:alltoallv` | ✅ | Single target with `select()`. Device kernel guarded with `__CUDA_ARCH__ \|\| __HIP_DEVICE_COMPILE__`. |
| `AllToAllvLl128.{h,cu,cuh}` | `:alltoallv_ll128` | ❌ | Depends on `:ll128_packet` / `:alltoallv` and uses LL128-specific NV intrinsics. |
| `AllToAllvAuto.{h,cu}` | `:alltoallv_auto` | ❌ | Depends on `:alltoallv_ll128`. |

### IB-collectives (`comms/prims/collectives/ib/`)

| Component | Target | AMD | Notes |
|---|---|---|---|
| `SendRecv` | `:sendrecv` | ❌ | Uses `<nccl.h>` and `comms/torchcomms/ncclx`. Same NCCL-on-AMD blocker. |

## LL128 (`comms/prims/transport/ll128/`)

| Component | Target | AMD | Notes |
|---|---|---|---|
| `Ll128Packet.cuh` / `Ll128Ops.cuh` / `Ll128AutoTune.cuh` | various | ❌ | NVIDIA-specific PTX/intrinsics for the 128-byte packet protocol. Not on the roadmap. |

## Window (`comms/prims/window/`)

| Component | Target | AMD | Notes |
|---|---|---|---|
| `DeviceWindow.cuh` | `:device_window` | ❌ | Uses `cuMem*` for symmetric memory. |
| `HostWindow.{h,cc}` | `:host_window` | ❌ | Pairs with `DeviceWindow`. |

Window support requires a HIP equivalent for `cuMem*` fabric memory or a
fallback IPC path.

## Tests (`comms/prims/tests/`)

### Tests with unified AMD/NVIDIA targets

| Test | Target | AMD | Notes |
|---|---|---|---|
| `P2pIbgdaTransportDeviceTest` | `:p2p_ibgda_transport_device_test` | ✅ | Single target via `select()`. |
| `MultipeerIbgdaTransportTest` | `:multipeer_ibgda_transport_test` | ✅ | Single target via `select()`. Build + runtime validated on AMD: all 5 `IbgdaBenchmarkFixture` tests (`PutWaitLocal`, `PutSignalWaitLocal`, `SignalOnly`, `PutSignalComparison`, `MultiPeerCounterFanOut`) PASS cross-host on dual-MI300X with BNXT NIC. |
| `AllToAllvTest` | `:alltoallv_test` | ✅ | Single target via `select()` (post-unification — replaces the prior split). |

### Tests without AMD coverage (NVIDIA-only)

These have `disable_amd_ci = True`. Most are mechanical to add (the
underlying library targets already build for AMD); the gap is just the
test BUCK plumbing.

| Test | Reason / dependency |
|---|---|
| `:thread_group_test`, `:copy_utils_test`, `:device_span_test`, `:p2p_self_transport_device_test`, `:p2p_nvl_transport_device_test`, `:barrier_test`, `:allgather_test`, `:gpu_mem_handler_test`, `:device_check_test`, `:timeout_trap_test`, `:ibgda_buffer_test`, `:multi_peer_nvl_transport_integration_test`, `:multi_peer_transport_test`, `:multi_peer_transport_multi_node_test`, `:multi_peer_transport_integration_test`, `:topology_discovery_test`, `:topology_discovery_e2e_test`, `:topology_classify_test`, `:external_staging_buffers_test`, `:tile_test`, `:p2p_nvl_transport_test` | No AMD test target yet (kernels build under hipcc; just needs the test BUCK to use `select()`). |
| `:host_window_test`, `:device_window_test` | Blocked on AMD Window support. |
| `:doca_host_utils_test`, `:multipeer_ibgda_device_transport_test`, `:nvml_fabric_info_test` | NVIDIA-only (DOCA host helpers / DOCA GPUNetIO direct / NVML). |

## Tests (`comms/prims/collectives/tests/`)

| Test | Target | AMD | Notes |
|---|---|---|---|
| `AllToAllvLl128Test` | `:alltoallv_ll128_test` | ❌ | Blocked on AMD LL128 support. |
| `AllToAllvLl128_2GpuTest` | `:alltoallv_ll128_2gpu_test` | ❌ | Same. |

## Benchmarks (`comms/prims/benchmarks/` and `comms/prims/collectives/benchmarks/`)

| Benchmark | NVIDIA Target | AMD | Reason |
|---|---|---|---|
| `IbgdaBenchmark` | `:ibgda_benchmark` | ✅ | Runs cross-host on dual-MI300X via BNXT NIC backend (`-c hpc_comms.nic=bnxt`). All 5 fixtures PASS, with per-message latency and per-size bandwidth measured. The mlx5 path compiles from the same source; runtime coverage requires a host with the matching mlx5 + DMABUF stack. |
| `IbgdaSendRecvBenchmark` | `:ibgda_sendrecv_benchmark` | 🚧 | Underlying transport works on AMD (same path as `IbgdaBenchmark`); benchmark BUCK plumbing not yet routed through `select()`. |
| `AllToAllvBenchmark` | `:alltoallv_benchmark` | 🚧 | Underlying IBGDA cross-node path works on AMD via BNXT (proven by `IbgdaBenchmark` above and the `MultipeerIbgdaTransportTest` runtime PASS); benchmark BUCK plumbing not yet routed through `select()`. |
| `AllToAllvLl128Benchmark` | `:alltoallv_ll128_benchmark` | ❌ | Blocked on LL128 support. |
| `AllGatherBenchmark` | `:allgather_benchmark` | ❌ | Could be added (deps available); not yet plumbed. |
| `IbSendRecvBenchmark*` | various | ❌ | Pulls in `<nccl.h>` and torchcomms; NCCL-on-AMD blocker. |
| `:p2p_nvl_*_benchmark`, `:multi_peer_benchmark*`, `:tile_bench` | various | ❌ | Need AMD test BUCK entries. |
| Servicelab benchmarks (`:copy_kernel_bench`, `:p2p_sync_bench`, etc.) | various | ❌ | Servicelab bench framework not yet verified on AMD. |
