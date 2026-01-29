#!/bin/bash

conda create -yn xpu_torchcomms_ci python=3.10
source activate xpu_torchcomms_ci
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export SCCACHE_DISABLE=1

export INTEL_ONEAPI=/opt/intel/oneapi
source $INTEL_ONEAPI/compiler/latest/env/vars.sh
source $INTEL_ONEAPI/ccl/latest/env/vars.sh

export USE_XCCL=ON
export USE_NCCL=OFF
export USE_NCCLX=OFF
export USE_TRANSPORT=OFF

python3 -m pip install torch torchvision torchaudio pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir 
cd torchcomms && pip install . --no-build-isolation && cd ..

python3 -c "import torch; import torchcomms; print(f'Torch version: {torch.__version__}')"

