#!/bin/bash -e

export CONTINUOUS_INTEGRATION=true
export CUDA_VISIBLE_DEVICES=2

: ${COMPILER:=${CXX}}

if [ -z "$1" ]; then
    echo "Assuming compiler is ${CXX}"
else
    COMPILER=${1}
fi

module load anaconda
module load likwid

if [ -n "${COMPILER}" ]; then
    module load cuda/10.0
    export CUDA_CLANG=1
else
    module load cuda
    export CUDA_CLANG=0
fi

module list
spack env activate timemory-ci
spack load dyninst boost
source activate
conda activate timemory
echo "python: $(which python)"
