#!/bin/bash
set -ex
# Install oneAPI DLE
ONEAPI_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9065c156-58ab-41b0-bbee-9b0e229ffca5/intel-deep-learning-essentials-2025.3.1.15_offline.sh"
wget -qO /tmp/intel-deep-learning-essentials.sh ${ONEAPI_URL}
chmod +x /tmp/intel-deep-learning-essentials.sh
/tmp/intel-deep-learning-essentials.sh -a --silent --eula accept
rm -f /tmp/intel-deep-learning-essentials.sh

export INTEL_ONEAPI=/opt/intel/oneapi
source $INTEL_ONEAPI/compiler/latest/env/vars.sh
source $INTEL_ONEAPI/ccl/latest/env/vars.sh

conda create -yn xpu_torchcomms_ci python=3.10
source activate xpu_torchcomms_ci
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y

export USE_XCCL=ON
export USE_NCCL=OFF
export USE_NCCLX=OFF
export USE_TRANSPORT=OFF
export USE_SYSTEM_LIBS=1

python3 -m pip install torch pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir
cd torchcomms && pip install . --no-build-isolation && cd ..

python3 -c "import torch; import torchcomms; print(f'Torch version: {torch.__version__}')"
