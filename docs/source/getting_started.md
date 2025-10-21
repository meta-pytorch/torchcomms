# Getting Started

torchcomms is an experimental, lightweight communication API for
[PyTorchDistributed(PTD)](https://docs.pytorch.org/docs/stable/distributed.html).
It provides a simplified, object-oriented interface
for distributed collective operations and offers both high-level
collective APIs and multiple out-of-the-box backends.

torchcomms provides:

- **Simplified Object-Oriented API**: A clean, intuitive interface for communication operations
- **Support for Multiple Backends**, including:
  - **NCCLX**: NVIDIA Collective Communications Library (extended) - Meta's production-tested backend that powers all generative AI services
  - **NCCL**: Standard NCCL backend for NVIDIA GPUs
  - **GLOO**: CPU-based backend for CPU tensors and metadata transfer
  - **RCCL**: AMD ROCm Collective Communications Library for AMD GPUs
- **Synchronous and Asynchronous Operations**: Flexible execution modes for different performance needs
- **Native PyTorch Integration**: Works seamlessly with PyTorch tensors and CUDA streams
- **Scalable**: Designed to scale to 100,000+ GPUs

Common use cases for torchcomms include distributed training of neural networks,
multi-GPU data parallelism, model parallelism across multiple devices,
and collective communication patterns such as AllReduce, Broadcast,
Send/Recv, and other operations.

## Prerequisites

torchcomms requires the following software and hardware:

- Python 3.10 or higher
- PyTorch 2.8 or higher
- CUDA-capable GPU (for NCCL/NCCLX or RCCL backends)
- CMake 3.22 or higher

## Installation

torchcomms is available on PyPI and can be installed using pip. Alternatively,
you can build torchcomms from source.

### Using pip (Nightly Builds)

You can install torchcomms and PyTorch nightly builds using pip:

```bash
pip install --pre torch torchcomms --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Building from Source

#### Prerequisites

- CMake 3.22 or higher
- Ninja 1.10 or higher

Alternatively, you can build torchcomms from source. If you want to build the NCCLX backend, we recommend building it under a virtual conda environment.
Run the following commands to build and install torchcomms:

```bash
# Create a conda environment
conda create -n torchcomms python=3.10
conda activate torchcomms
# Clone the repository
git clone git@github.com:meta-pytorch/torchcomms.git
cd torchcomms
```

Build the backend (choose one based on your hardware):

::::{tab-set}

:::{tab-item} Standard NCCL Backend

No build needed - uses the library provided by PyTorch

:::

:::{tab-item} NCCLX Backend

If you want to install the third-party dependencies directly from conda, run the following command:
```bash
USE_SYSTEM_LIBS=1 ./build_ncclx.sh
```

If you want to build and install the third-party dependencies from source, run the following command:
```bash
./build_ncclx.sh
```

:::

:::{tab-item} RCCL Backend

```bash
./build_rccl.sh
```

:::

::::

Install torchcomms:

```bash
# Install PyTorch (if not already installed)
pip install -r requirements.txt
pip install -v .
```

### Build Configuration

You can customize the build by setting environment variables before running pip install:

```bash
# Enable/disable specific backends (ON/OFF or 1/0)
export USE_NCCL=ON    # Default: ON
export USE_NCCLX=ON   # Default: ON
export USE_GLOO=ON    # Default: ON
export USE_RCCL=OFF   # Default: OFF
```

Then run:

```bash
# Install PyTorch (if not already installed)
pip install -r requirements.txt
pip install -v .
```

## Quick Start Example

Here's a simple example demonstrating synchronous `AllReduce` communication across multiple GPUs:

```python
#!/usr/bin/env python3
# example.py
import torch
from torchcomms import new_comm, ReduceOp

def main():
    # Initialize TorchComm with NCCLX backend
    device = torch.device("cuda")
    torchcomm = new_comm("ncclx", device, name="main_comm")

    # Get rank and world size
    rank = torchcomm.get_rank()
    world_size = torchcomm.get_size()

    # Calculate device ID
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")

    print(f"Rank {rank}/{world_size}: Running on device {device_id}")

    # Create a tensor with rank-specific data
    tensor = torch.full(
        (1024,),
        float(rank + 1),
        dtype=torch.float32,
        device=target_device
    )

    print(f"Rank {rank}: Before AllReduce: {tensor[0].item()}")

    # Perform synchronous AllReduce (sum across all ranks)
    torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

    # Synchronize CUDA stream
    torch.cuda.current_stream().synchronize()

    print(f"Rank {rank}: After AllReduce: {tensor[0].item()}")

    # Cleanup
    torchcomm.finalize()

if __name__ == "__main__":
    main()
```

### Running the Example

To run this example with multiple processes (one per GPU):

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=2 example.py

# Or using python -m torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=2 example.py
```

In the example above, we perform the following steps:

1. `new_comm()` creates a communicator with the specified backend
2. Each process gets its unique rank and total world size
3. Each rank creates a tensor with rank-specific values
4. All tensors are summed across all ranks
5. Clean up communication resources

## Asynchronous Operations

torchcomms also supports asynchronous operations for better performance.
Here is the same example as above, but with asynchronous `AllReduce`:

```python
import torch
from torchcomms import new_comm, ReduceOp

device = torch.device("cuda")
torchcomm = new_comm("ncclx", device, name="main_comm")

rank = torchcomm.get_rank()
device_id = rank % torch.cuda.device_count()
target_device = torch.device(f"cuda:{device_id}")

# Create tensor
tensor = torch.full((1024,), float(rank + 1), dtype=torch.float32, device=target_device)

# Start async AllReduce
work = torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

# Do other work while communication happens
print(f"Rank {rank}: Doing other work while AllReduce is in progress...")

# Wait for completion
work.wait()
print(f"Rank {rank}: AllReduce completed")

torchcomm.finalize()
```

## Next Steps

* Explore more examples in [torchcomms/comms/torchcomms/examples/](https://github.com/meta-pytorch/torchcomms/tree/main/comms/torchcomms/examples)
* Check out the [torchcomms API documentation](api)
