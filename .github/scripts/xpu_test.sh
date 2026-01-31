#!/bin/bash

apt update -y && apt install -y build-essential cmake

# Install Condaforge
BASE_URL="https://github.com/conda-forge/miniforge/releases/latest/download"  # @lint-ignore
CONDA_FILE="Miniforge3-Linux-$(uname -m).sh"
pushd /tmp
wget -q "${BASE_URL}/${CONDA_FILE}"
bash "${CONDA_FILE}" -b -f -p "/opt/conda"
popd
sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
export PATH="/opt/conda/bin:$PATH"

conda create -yn xpu_torchcomms_ci python=3.10 cmake
source activate xpu_torchcomms_ci
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y

# export CC=/usr/bin/gcc
# export CXX=/usr/bin/g++
# export SCCACHE_DISABLE=1

# Install oneAPI DLE
ONEAPI_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9065c156-58ab-41b0-bbee-9b0e229ffca5/intel-deep-learning-essentials-2025.3.1.15_offline.sh"
wget -qO /tmp/intel-deep-learning-essentials.sh ${ONEAPI_URL}
chmod +x /tmp/intel-deep-learning-essentials.sh
/tmp/intel-deep-learning-essentials.sh -a --silent --eula accept
rm -f /tmp/intel-deep-learning-essentials.sh

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
