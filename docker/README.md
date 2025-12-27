# torchcomms Docker Development Environment

A containerized environment that builds torchcomms from source with CUDA support.

## Quick Start

```bash
# Build and run the container (uses podman by default)
./docker/run.sh --build

# Or with docker
./docker/run.sh --build --runtime docker
```

The container clones the repository, builds all enabled backends, and installs torchcomms. It's ready to use immediately.

## Usage

```bash
./docker/run.sh [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--build` | - | Build the container image before running |
| `--cuda <version>` | `12.8.1` | CUDA version (`12.6.3`, `12.8.1`, `12.9.0`) |
| `--branch <ref>` | `main` | Git branch/tag/commit to clone |
| `--runtime <runtime>` | `podman` | Container runtime (`podman` or `docker`) |
| `--name <name>` | `torchcomms-dev` | Container name |

### Backend Options

| Option | Default | Description |
|--------|---------|-------------|
| `--nccl <ON\|OFF>` | `ON` | Standard NCCL backend |
| `--ncclx <ON\|OFF>` | `ON` | Extended NCCL backend |
| `--gloo <ON\|OFF>` | `ON` | CPU-based Gloo backend |
| `--rccl <ON\|OFF>` | `OFF` | ROCm RCCL backend (AMD GPUs) |
| `--rcclx <ON\|OFF>` | `OFF` | Extended RCCL backend |

## Examples

```bash
# Default build (NCCL + NCCLX + GLOO)
./docker/run.sh --build

# Faster build without NCCLX
./docker/run.sh --build --ncclx OFF

# Build with CUDA 12.6.3
./docker/run.sh --build --cuda 12.6.3

# Build a specific version
./docker/run.sh --build --branch v1.0.0

# CPU-only with Gloo
./docker/run.sh --build --nccl OFF --ncclx OFF --gloo ON
```

## What the Container Does

The Dockerfile:

1. Sets up CUDA base image with development tools
2. Creates `torchcomms` conda environment with all dependencies
3. Clones the repository (from `--branch`, default: `main`)
4. Builds enabled backends (NCCLX, RCCL, RCCLX)
5. Installs torchcomms via `pip install`

torchcomms is fully built and installed when the container starts.

## Development Workflow

If you make changes:

```bash
# Rebuild after local edits
pip install --no-build-isolation -v .
```

## Re-running Without Rebuild

Once built, just run without `--build` to reuse the existing image:

```bash
./docker/run.sh
```

## Container Details

- **Working directory**: `/workspace/torchcomms`
- **Conda environment**: `torchcomms` (auto-activated)
- **GPU access**: Enabled via `--device nvidia.com/gpu=all` (podman) or `--gpus all` (docker)
- **Network**: Host networking enabled

