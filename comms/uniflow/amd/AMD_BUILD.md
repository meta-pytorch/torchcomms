# Building Uniflow for AMD GPUs

This document describes how to build uniflow with AMD GPU support using ROCm/HIP.

Uniflow uses a **single `//comms/uniflow:uniflow` target** for both NVIDIA and AMD.
The platform is selected at build time via the `ovr_config//gpu:amd` constraint
(set by the AMD build modes / modifier). On AMD, HIP-specific code paths are
selected through `__HIP_PLATFORM_AMD__` and the CUDA runtime deps are swapped for
ROCm/HIP — there is no separate AMD library target.

> **AMD transport status:** The unified `uniflow` target builds and runs on AMD
> with **RDMA (RoCEv2) + GPUDirect** as the GPU transport (GPU-NIC PCIe affinity,
> RoCE GID auto-selection, `UNIFLOW_IB_*` tunables; validated 2-host on MI350 over
> Broadcom (bnxt) NICs). The **NVLink** transport is NVIDIA-only and is compiled
> out on AMD (guarded by `__HIP_PLATFORM_AMD__` in `MultiTransport.cpp`, with its
> BUCK deps `select()`'d out). Intra-node GPU-to-GPU P2P (XGMI) is exercised at the
> `CudaApi` level by the peer-to-peer transfer test, but is not a registered
> uniflow transport on AMD.

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

The `-m` flag selects the ROCm toolchain version:

| Version | Flag | Description |
|---------|------|-------------|
| ROCm 7.0 | `-m rocm70` | Latest stable (recommended) |
| ROCm 6.0 | `-m rocm60` | Previous stable |
| ROCm 6.1 | `-m rocm61` | Previous stable |

### Default Behavior

If `-m` is not specified, Buck uses the system default ROCm version configured in the build environment.

### Examples

```bash
# ROCm 7.0 (recommended)
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow

# ROCm 6.0
buck build @//mode/opt-amd-gpu -m rocm60 fbcode//comms/uniflow:uniflow

# System default
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

### Device Adapter (CUDA/HIP)

```bash
# Builds with HIP on AMD, CUDA on NVIDIA
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/cuda:cuda-device-adapter
```

### CUDA/HIP API wrapper

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/cuda:cuda-api
```

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
   constraint is active)
3. New NVIDIA-only code added under `MultiTransport`/transports must be guarded with
   `#ifndef __HIP_PLATFORM_AMD__` and its BUCK dep `select()`'d out on AMD

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
