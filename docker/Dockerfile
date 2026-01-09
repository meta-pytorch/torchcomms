# Development Dockerfile for torchcomms
#
# Build (using podman or docker):
#   podman build --format docker -t torchcomms-dev -f docker/Dockerfile .
#   podman build --format docker -t torchcomms-dev -f docker/Dockerfile --build-arg CUDA_VERSION=12.6.3 .
#
# Build with specific branch/tag:
#   podman build --format docker -t torchcomms-dev -f docker/Dockerfile --build-arg GIT_REF=main .
#
# Build with specific backends:
#   podman build --format docker -t torchcomms-dev -f docker/Dockerfile \
#       --build-arg USE_NCCL=ON --build-arg USE_NCCLX=OFF --build-arg USE_GLOO=ON .
#
# Run:
#   podman run --device nvidia.com/gpu=all -it -v $(pwd):/workspace/torchcomms torchcomms-dev
#   docker run --gpus all -it -v $(pwd):/workspace/torchcomms torchcomms-dev
#
# Or use the run.sh script:
#   ./docker/run.sh --build                    # Uses podman by default
#   ./docker/run.sh --build --runtime docker   # Uses docker
#
# Supported CUDA versions: 12.6.3, 12.8.1, 12.9.0
#
# Backend options (default values match README):
#   USE_NCCL=ON      Standard NCCL backend
#   USE_NCCLX=ON     Extended NCCL backend
#   USE_GLOO=ON      CPU-based Gloo backend
#   USE_RCCL=OFF     ROCm RCCL backend (AMD GPUs)
#   USE_RCCLX=OFF    Extended RCCL backend

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

# Store CUDA version for runtime reference
ARG CUDA_VERSION
ENV CUDA_VERSION=${CUDA_VERSION}

# Git reference (branch, tag, or commit) to clone
ARG GIT_REF=main

# Backend build arguments (defaults match README)
ARG USE_NCCL=ON
ARG USE_NCCLX=ON
ARG USE_GLOO=ON
ARG USE_RCCL=OFF
ARG USE_RCCLX=OFF

# Store backend options as environment variables
ENV USE_NCCL=${USE_NCCL}
ENV USE_NCCLX=${USE_NCCLX}
ENV USE_GLOO=${USE_GLOO}
ENV USE_RCCL=${USE_RCCL}
ENV USE_RCCLX=${USE_RCCLX}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    curl \
    pkg-config \
    build-essential \
    ninja-build \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set default shell to bash (must be before RUN commands that need bash)
SHELL ["/bin/bash", "-c"]

# Install Miniforge (conda-forge distribution)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

# Initialize conda for bash
RUN /opt/conda/bin/conda init bash

# Create conda environment for torchcomms (as per README)
RUN conda create -n torchcomms python=3.10 -y

# Install conda packages needed for building NCCLX (USE_SYSTEM_LIBS=1)
RUN conda run -n torchcomms conda install -y \
        cmake \
        ninja \
        jemalloc \
        gtest \
        boost \
        double-conversion \
        libevent \
        conda-forge::libsodium \
        libunwind \
        snappy \
        conda-forge::fast_float \
        libdwarf-dev \
        gflags \
        glog==0.4.0 \
        xxhash \
        zstd \
        conda-forge::zlib \
        conda-forge::libopenssl-static \
        fmt

# Clone the repository (as per README)
WORKDIR /workspace
ARG GIT_REF
RUN git clone --branch ${GIT_REF} https://github.com/meta-pytorch/torchcomms.git torchcomms

WORKDIR /workspace/torchcomms

# Install PyTorch (as per README: pip install -r requirements.txt)
RUN conda run -n torchcomms pip install -r requirements.txt

# Build NCCLX backend if enabled (as per README: USE_SYSTEM_LIBS=1 ./build_ncclx.sh)
ARG USE_NCCLX
RUN if [ "$USE_NCCLX" = "ON" ] || [ "$USE_NCCLX" = "1" ]; then \
        echo "Building NCCLX backend..." && \
        source /opt/conda/etc/profile.d/conda.sh && \
        conda activate torchcomms && \
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && \
        rm -rf /tmp/third-party && \
        rm -rf /tmp/build && \
        rm -rf build/ncclx && \
        USE_SYSTEM_LIBS=1 ./build_ncclx.sh; \
    else \
        echo "Skipping NCCLX backend build"; \
    fi

# Build RCCL backend if enabled (as per README: ./build_rccl.sh)
ARG USE_RCCL
RUN if [ "$USE_RCCL" = "ON" ] || [ "$USE_RCCL" = "1" ]; then \
        echo "Building RCCL backend..." && \
        source /opt/conda/etc/profile.d/conda.sh && \
        conda activate torchcomms && \
        ./build_rccl.sh; \
    else \
        echo "Skipping RCCL backend build"; \
    fi

# Build RCCLX backend if enabled (as per README: ./build_rcclx.sh)
ARG USE_RCCLX
RUN if [ "$USE_RCCLX" = "ON" ] || [ "$USE_RCCLX" = "1" ]; then \
        echo "Building RCCLX backend..." && \
        source /opt/conda/etc/profile.d/conda.sh && \
        conda activate torchcomms && \
        ./build_rcclx.sh; \
    else \
        echo "Skipping RCCLX backend build"; \
    fi

# Install torchcomms (as per README: pip install --no-build-isolation -v .)
# USE_SYSTEM_LIBS=1 tells the build to use shared libraries (-lglog) instead of static (-l:libglog.a)
ARG USE_NCCL
ARG USE_GLOO
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate torchcomms && \
    echo "Installing torchcomms with backends: NCCL=$USE_NCCL NCCLX=$USE_NCCLX GLOO=$USE_GLOO RCCL=$USE_RCCL RCCLX=$USE_RCCLX" && \
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && \
    USE_SYSTEM_LIBS=1 \
    USE_NCCL=$USE_NCCL \
    USE_NCCLX=$USE_NCCLX \
    USE_GLOO=$USE_GLOO \
    USE_RCCL=$USE_RCCL \
    USE_RCCLX=$USE_RCCLX \
    pip install --no-build-isolation -v .

# Set the torchcomms environment as default
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate torchcomms" >> ~/.bashrc

# Set working directory for development
WORKDIR /workspace/torchcomms

CMD ["/bin/bash"]
