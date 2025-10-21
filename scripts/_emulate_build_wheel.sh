#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script emulates the GitHub Nova build CI conda environment for local
# debugging.
# Intended to be used by docker_build_wheel.sh

set -ex

cd /torchcomms

CONDA_ENV=/tmp/conda_env
#conda create --yes --quiet --prefix "$CONDA_ENV" python=3.10 cmake=3.31.2 ninja=1.12.1 pkg-config=0.29 wheel=0.37
conda create --yes --quiet --prefix "$CONDA_ENV" python=3.13 cmake=3.31.2 ninja=1.12.1 pkg-config=0.29 wheel=0.37

CONDA_RUN="conda run --no-capture-output -p ${CONDA_ENV}"

${CONDA_RUN} pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128

${CONDA_RUN} bash scripts/_build_wheel.sh
