#!/bin/bash

set -ex

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

#source Deep Learning Essentials components
export TCM_ROOT=${DLE_PATH}/tcm/latest
export LD_LIBRARY_PATH="${TCM_ROOT}/lib":${LD_LIBRARY_PATH}

source ${DLE_PATH}/umf/latest/env/vars.sh
source ${DLE_PATH}/compiler/latest/env/vars.sh
source ${DLE_PATH}/tbb/latest/env/vars.sh
source ${DLE_PATH}/ccl/latest/env/vars.sh
source ${DLE_PATH}/pti/latest/env/vars.sh
source ${DLE_PATH}/mkl/latest/env/vars.sh

#Create Conda Env and install dependencies
conda create -yn xpu_torchcomms_ci-${RUNNER_NAME} python=3.10
source activate xpu_torchcomms_ci-${RUNNER_NAME}
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
#Expose ZE_AFFINITY_MASK to explicitly expose the number of Intel GPUs assigned to the runner for all tests.

echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
python3 -c "import torch; import torchcomms; print(f'Torch version: {torch.__version__}')"
python3 -c "import torch;print(\"XPU device available\"); print(torch.xpu.is_available())"
ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK} python3 -c "import torch;[print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"

#Run XCCL Python Integration Tests
ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK} torchcomms/comms/torchcomms/scripts/run_tests_integration_xccl_py.sh
