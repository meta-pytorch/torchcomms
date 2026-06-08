# Building Uniflow for AMD GPUs

This document describes how to build uniflow with AMD GPU support using ROCm/HIP.

## Prerequisites

- Access to a machine with AMD GPU and ROCm installed, OR
- Remote execution with AMD GPU workers configured

## Quick Start

### Build AMD Toolchain

```bash
# Build with default ROCm version
buck build @//mode/opt-amd-gpu fbcode//comms/uniflow:amd-toolchain

# Build with specific ROCm version (recommended)
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:amd-toolchain
```

### Build Uniflow for AMD

```bash
# Build uniflow with AMD support (default ROCm)
buck build @//mode/opt-amd-gpu fbcode//comms/uniflow:uniflow-amd

# Build with specific ROCm version
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd

# Build and show output location
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd --show-full-output
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
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd

# ROCm 6.0
buck build @//mode/opt-amd-gpu -m rocm60 fbcode//comms/uniflow:uniflow-amd

# System default
buck build @//mode/opt-amd-gpu fbcode//comms/uniflow:uniflow-amd
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
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd

# Debug build
buck build @//mode/dbg-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd

# Development build
buck build @//mode/dev-nosan-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd
```

## Building Specific Components

### AMD Toolchain

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:amd-toolchain
```

### Device Adapter (CUDA/HIP)

```bash
# Builds with HIP on AMD, CUDA on NVIDIA
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow/drivers/cuda:cuda-device-adapter
```

### Main Library

```bash
# Standard uniflow (uses device adapter)
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow

# AMD-specific uniflow target
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd
```

## Archive Types

### Thin Archives (Default)

By default, Buck produces thin archives (small files with references to object files):

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd
# Produces: buck-out/v2/gen/fbcode/comms/uniflow/libuniflow-amd.a (thin archive)
```

### Full Archives (Normal)

For distribution or when device code linking is required, build with normal archives:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd -c fbcode.archive_contents=normal
# Produces: Full archive with embedded object files
```

The AMD toolchain is already configured with `archive_contents = "normal"` to ensure proper device code linking.

## Output Locations

After building, artifacts are located in:

```
buck-out/v2/gen/fbcode/comms/uniflow/
├── libuniflow-amd.a          # Main AMD library (thin archive by default)
└── libcomms_uniflow_uniflow-amd.so  # Shared library (if applicable)
```

To find the exact path:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd --show-full-json-output
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

**Solution:** The build automatically detects GPU architectures via `get_rocm_arch_args()`. To override:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd -c rocm.arch=gfx942
```

### HIPIFY Translation Issues

If you encounter errors about CUDA functions not being recognized on AMD:

1. Ensure source files use `.cpp` or `.cu` extension (HIPIFY processes these)
2. Check that `__HIP_PLATFORM_AMD__` guards are in place for CUDA-specific headers
3. Verify the file is being compiled with the AMD toolchain

### Missing Symbols

If linking fails with undefined references to HIP symbols:

1. Ensure you're using `@//mode/opt-amd-gpu` (sets `ovr_config//gpu:amd`)
2. Verify `-m rocm70` (or appropriate version) is specified
3. Check that dependencies also support AMD/HIP

## Comparison with rcclx

Uniflow's AMD support follows the same patterns as rcclx:

| Aspect | rcclx | uniflow |
|--------|-------|---------|
| Toolchain | `hip_toolchain_override` | `hip_toolchain_override` |
| Archive type | `archive_contents = "normal"` | `archive_contents = "normal"` |
| ROCm selection | `-m rocm70` | `-m rocm70` |
| Build mode | `@//mode/opt-amd-gpu` | `@//mode/opt-amd-gpu` |
| HIPIFY | Automatic CUDA→HIP translation | Automatic CUDA→HIP translation |

Example rcclx build:
```bash
buck2 build @//mode/opt-amd-gpu -m rocm70 //param_bench/train/comms/cpp/rccl-tests/src:rccl_allreduce_perf
```

Equivalent uniflow build:
```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd
```

## Advanced Usage

### Custom ROCm Installation

If ROCm is installed in a non-standard location:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd \
  -c rocm.path=/custom/rocm/path
```

### Verbose Build Output

To see detailed compilation commands:

```bash
buck build @//mode/opt-amd-gpu -m rocm70 fbcode//comms/uniflow:uniflow-amd -v 2
```

### Cleaning Build Artifacts

```bash
# Clean all artifacts
buck clean

# Clean only stale artifacts (keeps daemon running)
buck clean --stale
```

## Integration with Applications

To use uniflow with AMD support in your application:

```python
# In your BUCK file
cpp_library(
    name = "my_app",
    srcs = [...],
    deps = select({
        "ovr_config//gpu:amd": ["fbcode//comms/uniflow:uniflow-amd"],
        "DEFAULT": ["fbcode//comms/uniflow:uniflow"],
    }),
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
