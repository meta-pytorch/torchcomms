#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -x # print commands before running them
set -euo pipefail # exit script on errors

# Initialize conda if not available in this shell
if ! command -v conda &> /dev/null; then
    conda_init=""
    if [ -n "${CONDA_EXE:-}" ]; then
        conda_init="${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
    fi
    for conda_path in "$conda_init" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniforge3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh" "/usr/etc/profile.d/conda.sh" "/etc/profile.d/conda.sh"; do
        if [ -n "$conda_path" ] && [ -f "$conda_path" ]; then
            source "$conda_path"
            break
        fi
    done
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found. Install miniconda and try again." >&2
        exit 1
    fi
fi

# Activate conda environment if CONDA_DEFAULT_ENV is set but not active
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_PREFIX:-}" = "" ]; then
    conda activate "$CONDA_DEFAULT_ENV"
fi

python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
conda install conda-forge::libopenssl-static conda-forge::rsync -y
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y

ONLY_FUNCS="AllReduce * * Sum f32" ./comms/github/build_rcclx.sh
