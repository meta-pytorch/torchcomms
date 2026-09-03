#!/bin/bash

set -ex

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

#Install Intel OMIX
apt-get update -y
apt-get install -y gpg wget build-essential ninja-build cmake git

# Add Intel GPG key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
  | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble/${OMIX_REPO}/${OMIX_VERSION} unified" | \
  tee /etc/apt/sources.list.d/intel-gpu-noble.list

apt-get update

# Install OMIX (includes GPU driver + oneAPI DLE components) and source OneAPI components
apt-get install -y intel-omix-dev
source /opt/intel/oneapi/setvars.sh
# Cleanup
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

source /home/jenkins/miniforge3/etc/profile.d/conda.sh
#Create Conda Env and install dependencies
conda remove -n xpu_torchcomms_ci-${RUNNER_NAME} --all -y || true
conda create -yn xpu_torchcomms_ci-${RUNNER_NAME} python=3.12.0
conda activate xpu_torchcomms_ci-${RUNNER_NAME}
conda install conda-forge::glog=0.4.0 conda-forge::gflags=2.2.2 conda-forge::fmt=12.2.0 -y

export USE_XCCL=ON
export USE_NCCL=OFF
export USE_NCCLX=OFF
export USE_GLOO=ON
export USE_TRANSPORT=OFF
export USE_SYSTEM_LIBS=1
ulimit -n 65535 # Increase the open file descriptor limit to avoid oneCCL/Level Zero
# initialization failures ("pidfd_getfd failed: Too many open files")

python3 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir

#Build and run XCCL C++ unit tests (mock-based, no XPU hardware required)
cd torchcomms
cmake -B build -G Ninja -DBUILD_TESTS=ON -DUSE_XCCL=ON -DUSE_NCCL=OFF -DUSE_NCCLX=OFF -DUSE_GLOO=ON -DUSE_TRANSPORT=OFF
cmake --build build
ctest --test-dir build --output-on-failure -R "TorchCommXCCLTest|TorchWorkXCCLQueueTest|TorchCommXCCLBootstrapTest|HintParsingTest"
cd ..

cd torchcomms && pip install '.[dev]' --no-build-isolation && cd ..

#Check Intel XPU visibility
#Expose ZE_AFFINITY_MASK to explicitly expose the number of Intel GPUs assigned to the runner for all tests.

echo "ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
python3 -c "import torch; import torchcomms; print(f'Torch version: {torch.__version__}')"
python3 -c "import torch;print(\"XPU device available\"); print(torch.xpu.is_available())"
ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK} python3 -c "import torch;[print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"

#Run XCCL Python Integration Tests
ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK} torchcomms/comms/torchcomms/scripts/run_tests_integration_xccl_py.sh
