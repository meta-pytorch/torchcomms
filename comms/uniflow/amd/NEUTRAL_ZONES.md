# Uniflow AMD — Neutral Zones

This document records the **neutral zones** of uniflow: modules that are
platform-agnostic and must remain free of any GPU-vendor dependency (CUDA, HIP,
or NVML). Keeping these modules vendor-neutral is what lets the AMD backend be a
drop-in addition behind the capability seams (see `AMD_SUPPORT_DESIGN.md`)
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

- `drivers/cuda/` — runtime seam (`CudaApi` / `CudaApiImpl` / `HipApi`), driver
  seam (`CudaDriverApi`), and the device adapter (`CudaDeviceAdapter` /
  `AmdDeviceAdapter`).
- `drivers/nvml/` — topology/management seam (`NvmlApi`, and later the amdsmi
  backend).

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
fbcode//comms/uniflow/drivers/cuda(/|:).*   # CUDA / HIP runtime + driver + device adapter
fbcode//comms/uniflow/drivers/nvml(/|:).*   # NVML / amdsmi topology

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
