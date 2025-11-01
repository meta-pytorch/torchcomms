#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script emulates the GitHub Nova build CI conda environment for local
# debugging.
# Intended to be used by docker_build_wheel.sh

set -ex

if [ "$(uname -m)" = "aarch64" ]; then
    echo "Building for aarch64"
    curl -L -o /mambaforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
    chmod +x /mambaforge.sh
    /mambaforge.sh -b -p /opt/conda
    rm /mambaforge.sh
    source /opt/conda/etc/profile.d/conda.sh
fi

cd /torchcomms



CONDA_ENV=/tmp/conda_env

# Python 3.10
#conda create --yes --quiet --prefix "$CONDA_ENV" python=3.10 cmake=3.31.2 ninja=1.12.1 pkg-config=0.29 wheel=0.37

# Python 3.13
conda create --yes --quiet --prefix "$CONDA_ENV" python=3.13 cmake=3.31.2 ninja=1.12.1 pkg-config=0.29 wheel=0.37

# Python 3.13t
#conda create --yes --quiet --prefix "$CONDA_ENV" python=3.13 cmake=3.31.2 ninja=1.12.1 pkg-config=0.29 wheel=0.37 python-freethreading -c conda-forge

CONDA_RUN="conda run --no-capture-output -p ${CONDA_ENV}"

${CONDA_RUN} pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128

# Uncomment to debug the build w/ docker exec
#${CONDA_RUN} bash -c "sleep 10000000000000"

${CONDA_RUN} bash scripts/_build_wheel.sh
