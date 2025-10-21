<p align="center">
  <img width="500" height="216" alt="torchcomms" src="logo.png">
</p>


# torchcomms

torchcomms is a new experimental communications API for PyTorch. This provides
both the high level collectives API as well as several out of the box backends.

* Documentation is located in the [docs/](./docs) folder.
* Examples are located in the [comms/torchcomms/examples](./comms/torchcomms/examples) folder.

## Prerequisites

torchcomms requires the following software and hardware:

- Python 3.10 or higher
- PyTorch 2.8 or higher
- CUDA-capable GPU (for NCCL/NCCLX or RCCL backends)
- CMake 3.22 or higher

## Installation

### Building from Source

You can build torchcomms from source.
Run the following commands to build and install torchcomms:

```bash
# Install PyTorch (if not already installed)
pip install torch
# Clone the repository
git clone git@github.com:meta-pytorch/torchcomms.git
cd torchcomms
```

#### Build the backend (choose one based on your hardware):

##### NCCLX Backend

```bash
./build_ncclx.sh
```

##### RCCL Backend

```bash
./build_rccl.sh
```

##### Standard NCCL Backend

No build needed - uses the library provided by PyTorch

#### Install torchcomms:

```bash
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
    torchcomm = new_comm("nccl", device, name="main_comm")

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

### Asynchronous Operations

torchcomms also supports asynchronous operations for better performance.
Here is the same example as above, but with asynchronous `AllReduce`:

```python
import torch
from torchcomms import new_comm, ReduceOp

device = torch.device("cuda")
torchcomm = new_comm("nccl", device, name="main_comm")

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

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

### torchcomms License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.

### Other Licenses

torchcomms backends include third-party source code may be using other licenses.
Please check the directory and relevant files to verify the license.

For convenience some of them are listed below:

* [NCCL License](./comms/ncclx/v2_27/LICENSE.txt)
