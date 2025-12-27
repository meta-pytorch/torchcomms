#!/bin/bash
#
# Launch the torchcomms development container.
#
# Usage:
#   ./docker/run.sh [OPTIONS]
#
# Options:
#   --build              Build the container image before running
#   --cuda <version>     CUDA version (default: 12.8.1, options: 12.6.3, 12.8.1, 12.9.0)
#   --branch <ref>       Git branch/tag/commit to clone (default: main)
#   --runtime <runtime>  Container runtime: podman or docker (default: podman)
#   --nccl <ON|OFF>      Enable NCCL backend (default: ON)
#   --ncclx <ON|OFF>     Enable NCCLX backend (default: ON)
#   --gloo <ON|OFF>      Enable Gloo backend (default: ON)
#   --rccl <ON|OFF>      Enable RCCL backend (default: OFF)
#   --rcclx <ON|OFF>     Enable RCCLX backend (default: OFF)
#   --name <name>        Container name (default: torchcomms-dev)
#   --help               Show this help message

set -e

# Default values (match README defaults)
BUILD=0
CUDA_VERSION="12.8.1"
GIT_REF="main"
RUNTIME="podman"
USE_NCCL="ON"
USE_NCCLX="ON"
USE_GLOO="ON"
USE_RCCL="OFF"
USE_RCCLX="OFF"
CONTAINER_NAME="torchcomms-dev"
IMAGE_NAME="torchcomms-dev"

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=1
            shift
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --branch)
            GIT_REF="$2"
            shift 2
            ;;
        --runtime)
            RUNTIME="$2"
            shift 2
            ;;
        --nccl)
            USE_NCCL="$2"
            shift 2
            ;;
        --ncclx)
            USE_NCCLX="$2"
            shift 2
            ;;
        --gloo)
            USE_GLOO="$2"
            shift 2
            ;;
        --rccl)
            USE_RCCL="$2"
            shift 2
            ;;
        --rcclx)
            USE_RCCLX="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build              Build the container image before running"
            echo "  --cuda <version>     CUDA version (default: 12.8.1)"
            echo "                       Supported: 12.6.3, 12.8.1, 12.9.0"
            echo "  --branch <ref>       Git branch/tag/commit to clone (default: main)"
            echo "  --runtime <runtime>  Container runtime (default: podman)"
            echo "                       Supported: podman, docker"
            echo ""
            echo "Backend options (defaults match README):"
            echo "  --nccl <ON|OFF>      Enable NCCL backend (default: ON)"
            echo "  --ncclx <ON|OFF>     Enable NCCLX backend (default: ON)"
            echo "  --gloo <ON|OFF>      Enable Gloo backend (default: ON)"
            echo "  --rccl <ON|OFF>      Enable RCCL backend (default: OFF)"
            echo "  --rcclx <ON|OFF>     Enable RCCLX backend (default: OFF)"
            echo ""
            echo "  --name <name>        Container name (default: torchcomms-dev)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --build                              # Build with defaults (NCCL+NCCLX+GLOO)"
            echo "  $0 --build --ncclx OFF                  # Build without NCCLX (faster)"
            echo "  $0 --build --cuda 12.6.3                # Build with CUDA 12.6.3"
            echo "  $0 --build --branch v1.0.0              # Build specific version"
            echo "  $0 --build --nccl OFF --gloo ON         # Build with only Gloo"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate runtime
if [[ "$RUNTIME" != "podman" && "$RUNTIME" != "docker" ]]; then
    echo "Error: Invalid runtime '$RUNTIME'. Use 'podman' or 'docker'."
    exit 1
fi

# Update image name to include CUDA version
IMAGE_NAME="torchcomms-dev:cuda${CUDA_VERSION}"
# Update container name to include CUDA version (if using default)
if [ "$CONTAINER_NAME" == "torchcomms-dev" ]; then
    CONTAINER_NAME="torchcomms-dev-cuda${CUDA_VERSION}"
fi

# Set runtime-specific flags
if [ "$RUNTIME" == "podman" ]; then
    GPU_FLAGS="--device nvidia.com/gpu=all --security-opt=label=disable"
    # Podman needs --format docker to support SHELL directive
    BUILD_FORMAT="--format docker"
else
    GPU_FLAGS="--gpus all"
    BUILD_FORMAT=""
fi

# Build image if requested
if [ "$BUILD" == 1 ]; then
    echo "Building container image with CUDA ${CUDA_VERSION} using ${RUNTIME}..."
    echo "  Branch: ${GIT_REF}"
    echo "  Backends: NCCL=$USE_NCCL NCCLX=$USE_NCCLX GLOO=$USE_GLOO RCCL=$USE_RCCL RCCLX=$USE_RCCLX"
    ${RUNTIME} build \
        ${BUILD_FORMAT} \
        --build-arg CUDA_VERSION="${CUDA_VERSION}" \
        --build-arg GIT_REF="${GIT_REF}" \
        --build-arg USE_NCCL="${USE_NCCL}" \
        --build-arg USE_NCCLX="${USE_NCCLX}" \
        --build-arg USE_GLOO="${USE_GLOO}" \
        --build-arg USE_RCCL="${USE_RCCL}" \
        --build-arg USE_RCCLX="${USE_RCCLX}" \
        -t "${IMAGE_NAME}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "${SCRIPT_DIR}"
fi

# Check if image exists
if ! ${RUNTIME} image inspect "${IMAGE_NAME}" &>/dev/null; then
    echo "Container image '${IMAGE_NAME}' not found. Building with CUDA ${CUDA_VERSION}..."
    echo "  Branch: ${GIT_REF}"
    echo "  Backends: NCCL=$USE_NCCL NCCLX=$USE_NCCLX GLOO=$USE_GLOO RCCL=$USE_RCCL RCCLX=$USE_RCCLX"
    ${RUNTIME} build \
        ${BUILD_FORMAT} \
        --build-arg CUDA_VERSION="${CUDA_VERSION}" \
        --build-arg GIT_REF="${GIT_REF}" \
        --build-arg USE_NCCL="${USE_NCCL}" \
        --build-arg USE_NCCLX="${USE_NCCLX}" \
        --build-arg USE_GLOO="${USE_GLOO}" \
        --build-arg USE_RCCL="${USE_RCCL}" \
        --build-arg USE_RCCLX="${USE_RCCLX}" \
        -t "${IMAGE_NAME}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "${SCRIPT_DIR}"
fi

# Stop and remove existing container if it exists
${RUNTIME} stop "${CONTAINER_NAME}" 2>/dev/null || true
${RUNTIME} rm "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting torchcomms development container..."
echo "  Runtime: ${RUNTIME}"
echo "  CUDA version: ${CUDA_VERSION}"
echo "  Branch: ${GIT_REF}"
echo "  Image: ${IMAGE_NAME}"
echo "  Container name: ${CONTAINER_NAME}"
echo "  Mounting: ${PROJECT_ROOT} -> /workspace/torchcomms"

# Run the container
${RUNTIME} run \
    ${GPU_FLAGS} \
    --name "${CONTAINER_NAME}" \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -it \
    -v "${PROJECT_ROOT}:/workspace/torchcomms" \
    -w /workspace/torchcomms \
    "${IMAGE_NAME}" \
    /bin/bash
