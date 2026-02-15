#!/bin/bash

set -ex
# Install oneAPI DLE
ONEAPI_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/b3e6c1bf-a6d5-4580-8b1d-80cbfd38c8af/intel-deep-learning-essentials-2025.3.2.36_offline.sh"
wget -qO /tmp/intel-deep-learning-essentials.sh ${ONEAPI_URL}
chmod +x /tmp/intel-deep-learning-essentials.sh
/tmp/intel-deep-learning-essentials.sh -a --silent --eula accept
rm -f /tmp/intel-deep-learning-essentials.sh

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

#source OneAPI components
export INTEL_ONEAPI=/opt/intel/oneapi
export TCM_ROOT=${INTEL_ONEAPI}/tcm/latest
export LD_LIBRARY_PATH="${TCM_ROOT}/lib":${LD_LIBRARY_PATH}

source $INTEL_ONEAPI/umf/latest/env/vars.sh
source $INTEL_ONEAPI/compiler/latest/env/vars.sh
source $INTEL_ONEAPI/tbb/latest/env/vars.sh
source $INTEL_ONEAPI/ccl/latest/env/vars.sh
source $INTEL_ONEAPI/pti/latest/env/vars.sh
source $INTEL_ONEAPI/mkl/latest/env/vars.sh

#Create Conda Env and install dependencies
conda create -yn xpu_torchcomms_ci python=3.10
source activate xpu_torchcomms_ci
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y

export USE_XCCL=ON
export USE_NCCL=OFF
export USE_NCCLX=OFF
export USE_GLOO=OFF
export USE_TRANSPORT=OFF
export USE_SYSTEM_LIBS=1

python3 -m pip install typing-extensions numpy sympy
python3 -m pip install --no-deps --pre torch pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir 
cd torchcomms && pip install . --no-deps --no-build-isolation && cd ..

#Check Intel XPU visibility
python3 -c "import torch; import torchcomms; print(f'Torch version: {torch.__version__}')"
python3 -c "import torch;print(\"XPU device available\"); print(torch.xpu.is_available())"
python3 -c "import torch;[print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"

#Run XCCL Python Integration Tests
torchcomms/comms/torchcomms/scripts/run_tests_integration_xccl_py.sh
