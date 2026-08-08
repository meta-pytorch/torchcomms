# Building Uniflow for AMD GPUs

This document describes how to build uniflow with AMD GPU support using ROCm/HIP.

Uniflow uses a **single `//comms/uniflow:uniflow` target** for both NVIDIA and AMD.
The platform is selected at build time via the `ovr_config//gpu:amd` constraint
(set by the AMD build modes / modifier). On AMD, HIP-specific code paths are
selected through `__HIP_PLATFORM_AMD__` and the CUDA runtime deps are swapped for
ROCm/HIP — there is no separate AMD library target.

> **AMD transport status:** The unified `uniflow` target builds and runs on AMD
> supporting **two transport types**: **P2P (XGMI)** intra-node and **RDMA (RoCEv2)
> + GPUDirect**. Neither is registered unconditionally — `MultiTransportFactory`
> registers P2P only when `deviceId >= 0` and `P2pTransportFactory::supported()`
> passes (it logs "P2P transport disabled for device N" otherwise), and RDMA only
> when `selectNics()` returns a non-empty NIC list. Only NVLink is NVIDIA-only.
>
> `MultiTransport::selectTransport` resolves in this order:
> 1. intra-node VRAM<->VRAM: `[intraNodeTransport]` -> on-host tier -> RDMA,
>    where the on-host tier is NVLink on NVIDIA and **P2P/XGMI on AMD**;
> 2. otherwise: RDMA.
>
> `intraNodeTransport` flips the intra-node first choice; both tiers stay
> registered, so it is a per-transfer selection override rather than a
> kill-switch.
>
> **P2P (XGMI)** — `transport/p2p`, takes the NVLink tier's place on AMD when its
> capability checks pass
> (`MultiTransport.cpp` registers `P2pTransportFactory` under
> `__HIP_PLATFORM_AMD__`). Uses `hipIpc` memory handles.
> `P2pTransportFactory::supported()` gates on an all-to-all-XGMI arch allowlist
> (`gfx942` MI300, `gfx950` MI350, `gfx1250` MI450) so a non-XGMI part cannot let
> presence-driven selection prefer a slow PCIe-P2P link over RDMA
> (`hipDeviceCanAccessPeer` conflates the two).
>
> **RDMA / GPUDirect** — features from the Phase 2 stack (D107220750 D107220748
> D107220757 D107346920 D107220749 D108381013 D108675437 D108942542):
> - GPU-NIC PCIe affinity via `selectGpuNics` mapping each GPU to topologically-closest NIC
> - RoCEv2 GID auto-selection skipping link-local (fe80::/10, 169.254/16), preferring IPv4-mapped then first global RoCEv2, with validation against gid_tbl_len
> - Caller-configurable tunables via `MultiTransportFactoryOptions`: NicFilter (HCA selection), gidIndex (force GID, default auto-select), netdevPrefix (default `"beth"` for Broadcom bnxt NIC selection), trafficClass
> - Netdev-prefix NIC selection capturing backing netdev name in topology
> - GPUDirect registration probes AMD peer-mem support (`amdGpuDirectRdmaSupported()`:
>   amdkfd sysfs + `/proc/kallsyms` fallback) and falls back to plain `ibv_reg_mr`
>   when dma-buf export is unavailable
> - Validated 2-host on MI350 (gfx950, ROCm 7.0) over Broadcom bnxt NICs with `cross_host_test` and `rdma_bandwidth` benchmarks
>
> The **NVLink** transport is NVIDIA-only and is compiled out on AMD (guarded by
> `__HIP_PLATFORM_AMD__` in `MultiTransport.cpp`, BUCK deps select'd out). Note
> that the **cuMem VMM allocator** (`cuMemCreate`/`AddressReserve`/`Map`/
> `SetAccess`/`Export`/`Import`) is reached only by `NVLinkTransport` and the
> NVLink benchmark, so it never executes on AMD; the AMD P2P tier uses `hipIpc`
> handles instead.

## Prerequisites

- Access to a machine with AMD GPU and ROCm installed, OR
- Remote execution with AMD GPU workers configured

## Quick Start

### Build Uniflow for AMD

```bash
# Build uniflow for AMD (default ROCm)
buck build @//mode/opt-amd-gpu fbcode//comms/uniflow:uniflow

# Build with a specific ROCm version (recommended)
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow

# Build and show output location
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow --show-full-output
```

The same target builds for NVIDIA with no GPU constraint:

```bash
buck build fbcode//comms/uniflow:uniflow
```

### Build via the constraint modifier (no mode file)

The AMD platform can also be selected directly with the `ovr_config//gpu:amd`
modifier — handy for quick checks:

```bash
buck2 build 'fbcode//comms/uniflow:uniflow?ovr_config//gpu:amd'
```

## ROCm Version Selection

### Available ROCm Versions

The `-m` flag selects the ROCm toolchain version. The authoritative version set
is `ROCM_TP2_VERSIONS` in
`fbsource/tools/build_defs/third_party/rocm_tp2_versions.bzl`:

| Version | Flag | Status |
|---------|------|--------|
| ROCm 7.0 | `-m rocm70` | **Default and recommended.** `ROCM_DEFAULT_VERSION`; the version AMD ships backported fixes for. Required for MI350X+. Resolves to 7.0.2. |
| ROCm 7.2 | `-m rocm_latest` | The `latest` tp2 label. Newest available and receives upstream test patches, but **explicitly not guaranteed stable** — intended for early enablement work (e.g. MI450X). Prefer asking AMD to backport into 7.0 over adopting it. |
| ROCm 7.0.2.2 | — | Imported but **not publicly reachable**: its `constraint_value` visibility is restricted to buck infrastructure (see `arvr/tools/build_defs/config/third-party/rocm/constraints/BUCK`). |
| ROCm 6.2.1 / 6.4.0 / 6.4.2 | `-m rocm621` / `rocm640` / `rocm642` | Legacy, do not use — scheduled for removal and no longer receiving patches. |

There is no ROCm 6.0 or 6.1 in fbcode. Version policy is owned by AI Software
Platform; see the "ROCm Version Strategy" posts in the ROCm Users group for the
current plan of record.

### Default Behavior

If `-m` is not specified, Buck uses `ROCM_DEFAULT_VERSION` from
`rocm_tp2_versions.bzl`, which is currently **7.0**.

### Examples

```bash
# ROCm 7.0 (default, recommended)
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow

# ROCm 7.2, for early-enablement work only
buck build @//mode/opt-amd-gpu -m rocm_latest fbcode//comms/uniflow:uniflow

# Default version (7.0)
buck build @//mode/opt-amd-gpu fbcode//comms/uniflow:uniflow
```

## Build Modes

### Optimization Modes

| Mode | Command | Description |
|------|---------|-------------|
| Optimized | `@//mode/opt-amd-gpu` | Production build with optimizations |
| Debug | `@//mode/dbg-amd-gpu` | Debug build with symbols |
| Dev (no sanitizer) | `@//mode/dev-nosan-amd-gpu` | Development build |

### Examples

```bash
# Optimized build (production)
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow

# Debug build
buck build @//mode/dbg-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow

# Development build
buck build @//mode/dev-nosan-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow
```

## Building Specific Components

The Phase 2 GPU runtime seam in `drivers/cuda/BUCK` exposes four first-party targets,
all built via the unified `//comms/uniflow:uniflow` but testable individually.
On AMD they translate CUDA → HIP at build time; on NVIDIA they compile original CUDA.

### CUDA/HIP runtime API wrapper

```bash
# Builds with HIP on AMD via oss_gpu_cpp_library (hipifies sources+headers, rename_cpp_to_hip),
# CUDA on NVIDIA
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/cuda:cuda-api
# or buck2:
buck2 build @fbcode//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/cuda:cuda-api
```

`cuda-api` wraps `cuda_runtime_api.h` — device management, host alloc/free,
memcpy async/peer/batch, stream/event, plus `CudaDeviceGuard` RAII.

### Device Adapter (CUDA/HIP)

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/cuda:cuda-device-adapter
```

Implements `DeviceAdapter` interface for pinned host memory suitable for DMA.
Also via `oss_gpu_cpp_library` so header stays plain (no vendor types) while
source is hipified.

### CUDA Driver API wrapper

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/cuda:cuda-driver-api
```

Wraps `cuda.h` driver API — cuMem VMM create/release/map/unmap, address reserve/free,
set access, export/import shareable handle, dma-buf export, streamWriteValue64,
device attributes. Built with uniflow-local `hipify` rule (hipify-perl) +
`hip_toolchain_override`, **not** `gpu_cpp_library`, due to symbol-translation
blocker: `gpu_cpp_library` hipify renames `CU_STREAM_WRITE_VALUE_DEFAULT` breaking
RDMA `CopyEngine` consumers still on hipify-perl. Until RDMA migrates to
`gpu_cpp_library`, driver seam stays on hipify-perl to preserve CUDA spellings.

### Topology Discovery backend

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/cuda:cuda-topology-discovery
```

`CudaTopologyDiscovery` wires `CudaApi`, `NvmlApi` factory, `IbvApi`, and
`SysfsApi` into `TopologyDiscovery` interface. Plain C++ library selecting GPU
seam targets via deps.

### NVML factory (topology/management seam)

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/nvml:nvml-api
```

`createNvmlApi()` factory returns real NVML on NVIDIA (`nvml-lazy`) and no-op
stub on AMD (amdsmi backend is future work). Keeps neutral code free of direct
NVML linkage.

### Neutral zone dependency guards

```bash
# Verify platform-agnostic modules stay free of GPU deps
buck2 test @fbcode//mode/opt fbcode//comms/uniflow/amd:
# One check_dependencies_test per neutral-zone target (10 targets in
# NEUTRAL_ZONE_TARGETS), blocklisting drivers/cuda/*, drivers/nvml/*, and
# external runtime libs (cuda-lazy, nvml-lazy, amdhip64-lazy, amdsmi)
```

See `NEUTRAL_ZONES.md` for full architecture description of frozen modules
(executor, controller, core result/segment, logging, sysfs, ibverbs core) and
GPU seam boundaries.

### Main Library

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow
```

## Archive Types

### Thin Archives (Default)

By default, Buck produces thin archives (small files with references to object files):

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow
# Produces a thin archive (a small file referencing the object files).
# Use --show-full-output (or --show-full-json-output) to print its path.
```

### Full Archives (Normal)

For distribution or when device code linking is required, build with normal archives:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow -c fbcode.archive_contents=normal
# Produces: Full archive with embedded object files
```

## Output Locations

To find the exact path of the built artifact:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow --show-full-json-output
```

## Troubleshooting

### Build Failures

**Problem:** `rocm_path` not found or ROCm not available

**Solution:** Ensure you're building on a machine with ROCm installed or using remote execution with AMD workers.

```bash
# Check if ROCm is available
ls /opt/rocm 2>/dev/null || echo "ROCm not installed locally"
```

**Problem:** Wrong architecture or GPU target

**Solution:** To override the GPU architecture:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow -c rocm.arch=gfx942
```

### HIP/CUDA Translation Issues

If you encounter errors about CUDA functions not being recognized on AMD:

1. Check that `__HIP_PLATFORM_AMD__` guards are in place for CUDA-specific code
2. Verify the file is being compiled with the AMD toolchain (the `ovr_config//gpu:amd`
   constraint is active via `@//mode/opt-amd-gpu`)
3. New NVIDIA-only code added under `MultiTransport`/transports must be guarded with
   `#ifndef __HIP_PLATFORM_AMD__` and its BUCK dep `select()`'d out on AMD
4. **Driver-API symbol mismatch:** If you see undefined references to
   `hipStreamWriteValueDefault` or wrong-arity `hipGetErrorName`, you're likely
   mixing `gpu_cpp_library` hipify output with hipify-perl consumers.
   `cuda-driver-api` must stay on uniflow-local hipify-perl until RDMA
   `CopyEngine` migrates to `gpu_cpp_library` — see `drivers/cuda/BUCK` Phase 2
   seam comment for blocker details. Both producer and consumer must use same
   hipify tool to preserve CUDA spellings.

### Neutral Zone Guard Failures

If `buck2 test fbcode//comms/uniflow/amd:` fails with `neutral_zone_dep_guard_*`:

1. The failing target's transitive deps now reach `drivers/cuda/*` or `drivers/nvml/*`
   or external GPU runtime libs — check `buck2 uquery "alldeps(fbcode//path:target)"`
2. Move GPU-touching code behind seam interface (`CudaApi`, `CudaDriverApi`,
   `DeviceAdapter`, or `NvmlApi` factory) rather than relaxing the guard
3. See `NEUTRAL_ZONES.md` for full list of frozen modules and allowed GPU seam boundaries

### Missing Symbols

If linking fails with undefined references to HIP symbols:

1. Ensure you're using `@//mode/opt-amd-gpu` (sets `ovr_config//gpu:amd`)
2. Verify `-m rocm70` (or appropriate version) is specified
3. Check that dependencies also support AMD/HIP

## Comparison with rcclx

Uniflow's AMD support follows the same patterns as rcclx:

| Aspect | rcclx | uniflow |
|--------|-------|---------|
| Platform selection | `ovr_config//gpu:amd` | `ovr_config//gpu:amd` |
| Archive type | `archive_contents = "normal"` | `archive_contents = "normal"` |
| ROCm selection | `-m rocm70` | `-m rocm70` |
| Build mode | `@//mode/opt-amd-gpu` | `@//mode/opt-amd-gpu` |

Equivalent uniflow build:
```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow
```

## Advanced Usage

### Custom ROCm Installation

If ROCm is installed in a non-standard location:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow \
  -c rocm.path=/custom/rocm/path
```

### Verbose Build Output

To see detailed compilation commands:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow -v 2
```

## Integration with Applications

Because uniflow is a single target for both platforms, applications simply depend on
`fbcode//comms/uniflow:uniflow` — no per-GPU `select()` is required:

```python
# In your BUCK file
cpp_library(
    name = "my_app",
    srcs = [...],
    deps = ["fbcode//comms/uniflow:uniflow"],
)
```

Build your application:

```bash
# For AMD
buck build @//mode/opt-amd-gpu -m rocm70 //your:app

# For NVIDIA (default)
buck build //your:app
```

## References

- **rcclx**: `fbcode/comms/rcclx/` - Reference implementation
- **pipes**: `fbcode/comms/pipes/` - AMD support patterns
- **HIP Documentation**: https://rocm-documentation.readthedocs.io/
- **ROCm Modes**: `fbcode/mode/*amd*`

## Support

For issues with AMD builds:
- Oncall: `ncclx`
- Slack: #ncclx or #rocm-users
- Documentation: Internal fbcode/comms documentation
