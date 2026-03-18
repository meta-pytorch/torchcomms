#!/bin/bash

set -ex
source "$(dirname "$0")/xpu_setup_env.sh" xpu_torchcomms_ci

#Build and run C++ tests
cd torchcomms
cmake -B build -G Ninja -DBUILD_TESTS=ON -DUSE_XCCL=ON -DUSE_NCCL=OFF -DUSE_NCCLX=OFF -DUSE_TRANSPORT=OFF
cmake --build build
ctest --test-dir build --output-on-failure -R "TorchCommXCCLTest|TorchWorkXCCLQueueTest|TorchCommXCCLBootstrapTest|HintParsingTest"
cd ..

#Install torchcomms Python package
cd torchcomms && pip install . --no-deps --no-build-isolation && cd ..

#Check Intel XPU visibility
python3 -c "import torch; import torchcomms; print(f'Torch version: {torch.__version__}')"
python3 -c "import torch;print(\"XPU device available\"); print(torch.xpu.is_available())"
python3 -c "import torch;[print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"

#Run XCCL Python Integration Tests
torchcomms/comms/torchcomms/scripts/run_tests_integration_xccl_py.sh
