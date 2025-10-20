# TorchComm: PyTorch Communication Library

TorchComm is a communication library for PyTorch that provides a unified
interface for various communication backends. It supports point-to-point and
collective operations for distributed training and inference.

## Table of Contents

- [Installation](#installation)
- [API Overview](#api-overview)
- [Detailed API Reference](#detailed-api-reference)
  - [Constructor and Initialization](#constructor-and-initialization)
  - [Point-to-Point Operations](#point-to-point-operations)
  - [Collective Operations](#collective-operations)
  - [Scatter and Gather Operations](#scatter-and-gather-operations)
  - [Communicator Management](#communicator-management)
  - [Work Object](#work-object)
  - [Options Configuration](#options-configuration)
- [Environment Variables](#environment-variables)
- [Examples](#examples)

## Installation

TorchComm is part of the PyTorch communication libraries. It can be imported in
Python as:

```python
import torchcomms
```

## API Overview

TorchComm provides the following categories of operations:

- **Initialization and Management**: Create, initialize, and manage
  communication groups
- **Point-to-Point Operations**: Send and receive operations between ranks
- **Collective Operations**: Operations involving all ranks in a communicator
  (broadcast, reduce, all-reduce, etc.)
- **Scatter and Gather Operations**: Distribute data from one rank to many or
  collect data from many ranks to one
- **Asynchronous Operations**: All operations support asynchronous execution
  with work objects
- **CUDA Graphs Support**: All operations are compatible with CUDA graph capture
  and replay for optimized performance

## Detailed API Reference

### Constructor and Initialization

#### Constructor

```python
torchcomms.new_comm(backend, device, ...)
```

- **backend** (str): Communication backend to use (e.g., "ncclx")
- **device** (torch.device): Device to use for communication
- **options** (CommOptions, optional): Configuration options including store,
  timeout, and other settings

**Note**: The store parameter is now optional and can be provided through the
options parameter. TorchComms supports multiple bootstrap backends including
TCPStore, and Torchrun. If no store is provided, TorchComms will automatically
detect the environment and use the appropriate bootstrap mechanism.

#### Initialization Methods

```python
finalize()
```

Finalize and free all resources.

```python
get_rank()
```

Get the rank of this process. Returns an integer.

```python
get_size()
```

Get the world size (total number of processes). Returns an integer.

### Point-to-Point Operations

#### Send

```python
send(tensor, dst, async_op, hints=None, timeout=None)
```

Send a tensor to the destination rank.

- **tensor** (torch.Tensor): Tensor to send
- **dst** (int): Destination rank
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### Receive

```python
recv(tensor, src, async_op, hints=None, timeout=None)
```

Receive a tensor from the source rank.

- **tensor** (torch.Tensor): Tensor to receive data into
- **src** (int): Source rank
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

### Collective Operations

#### Broadcast

```python
broadcast(tensor, root, async_op, hints=None, timeout=None)
```

Broadcast a tensor from the root rank to all other ranks.

- **tensor** (torch.Tensor): Tensor to broadcast
- **root** (int): Root rank
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### All-Reduce

```python
all_reduce(tensor, op, async_op, hints=None, timeout=None)
```

Perform an all-reduce operation on a tensor across all ranks.

- **tensor** (torch.Tensor): Tensor to reduce
- **op** (ReduceOp): Reduction operation
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### Reduce

```python
reduce(tensor, root, op, async_op, hints=None, timeout=None)
```

Reduce a tensor across all ranks to the root rank.

- **tensor** (torch.Tensor): Tensor to reduce
- **root** (int): Root rank
- **op** (ReduceOp): Reduction operation
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### All-Gather

```python
all_gather(tensor_list, tensor, async_op, hints=None, timeout=None)
```

Gather tensors from all ranks and store them in a list.

- **tensor_list** (list of torch.Tensor): List of tensors to store gathered
  results
- **tensor** (torch.Tensor): Tensor to gather
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### All-Gather Single

```python
all_gather_single(output, input, async_op, hints=None, timeout=None)
```

Gather tensors from all ranks and store them in a single tensor.

- **output** (torch.Tensor): Output tensor to store gathered results
- **input** (torch.Tensor): Input tensor to gather
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### Reduce-Scatter

```python
reduce_scatter(output, input_list, op, async_op, hints=None, timeout=None)
```

Reduce and scatter tensors across all ranks.

- **output** (torch.Tensor): Output tensor
- **input_list** (list of torch.Tensor): List of input tensors
- **op** (ReduceOp): Reduction operation
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### Reduce-Scatter Single

```python
reduce_scatter_single(output, input, op, async_op, hints=None, timeout=None)
```

Reduce and scatter a single tensor across all ranks.

- **output** (torch.Tensor): Output tensor
- **input** (torch.Tensor): Input tensor
- **op** (ReduceOp): Reduction operation
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### All-to-All Single

```python
all_to_all_single(output, input, async_op, hints=None, timeout=None)
```

All-to-all exchange of a single tensor.

- **output** (torch.Tensor): Output tensor
- **input** (torch.Tensor): Input tensor
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### All-to-All

```python
all_to_all(output_tensor_list, input_tensor_list, async_op, hints=None, timeout=None)
```

All-to-all exchange of tensors.

- **output_tensor_list** (list of torch.Tensor): List of output tensors
- **input_tensor_list** (list of torch.Tensor): List of input tensors
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### All-to-All Variable Single

```python
all_to_all_v_single(output, input, output_split_sizes, input_split_sizes, async_op, hints=None, timeout=None)
```

All-to-all exchange with variable split sizes, allowing different amounts of
data to be sent to each peer.

- **output** (torch.Tensor): Output tensor
- **input** (torch.Tensor): Input tensor
- **output_split_sizes** (list of int): List of output split sizes for each rank
- **input_split_sizes** (list of int): List of input split sizes for each rank
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### Barrier

```python
barrier(async_op, hints=None, timeout=None)
```

Synchronize all processes.

- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

### Scatter and Gather Operations

#### Scatter

```python
scatter(output_tensor, input_tensor_list, root, async_op, hints=None, timeout=None)
```

Scatter tensors from the root rank to all ranks.

- **output_tensor** (torch.Tensor): Output tensor
- **input_tensor_list** (list of torch.Tensor): List of input tensors (only used
  on root)
- **root** (int): Root rank
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

#### Gather

```python
gather(output_tensor_list, input_tensor, root, async_op, hints=None, timeout=None)
```

Gather tensors from all ranks to the root rank.

- **output_tensor_list** (list of torch.Tensor): List of output tensors (only
  used on root)
- **input_tensor** (torch.Tensor): Input tensor
- **root** (int): Root rank
- **async_op** (bool): Whether to perform the operation asynchronously
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the operation
- **Returns**: TorchWork object

### Communicator Management

#### Split

```python
split(ranks)
```

Split the communicator into a subgroup.

- **ranks** (list of int): List of ranks to include in the new subgroup. If the list is empty, `None` will be returned. If the list is non-empty but does not include the current rank, an exception will be thrown.
  communicator
- **Returns**: New TorchComm object

### Work Object

The TorchWork object is returned by asynchronous operations and provides methods
to check completion and wait for operations.

```python
is_completed()
```

Check if the operation is completed.

- **Returns**: Boolean indicating completion status

```python
wait()
```

Wait for the operation to complete.

### CUDA Graphs Support

TorchComm supports CUDA graph capture and replay for optimized performance. All
communication operations are compatible with CUDA graphs, allowing you to
capture communication patterns and replay them with reduced CPU overhead.

When operations are captured within a CUDA graph:

- All operations execute on the current CUDA stream
- Work objects created during graph capture have special handling for graph mode
- Multiple graph replays can be performed for repeated communication patterns
- Graph mode eliminates CUDA event overhead during replay

CUDA graphs are particularly beneficial for:

- Repeated communication patterns in training loops
- Reducing CPU overhead for small, frequent operations
- Optimizing end-to-end performance in distributed training scenarios

### Configuration Options

#### Hints and Timeout Parameters

All communication operations accept optional `hints` and `timeout` keyword
arguments:

- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options
- **timeout** (timedelta, optional): Timeout for the specific operation

**Backend-Specific Hints:**

- **"torchcomm::ncclx::high_priority_stream"**: Set to "true" to enable high
  priority CUDA stream for NCCL operations (default: not set, equivalent to
  false)

#### Communicator Options

The `new_comm` function accepts several optional parameters for configuration:

- **abort_process_on_timeout_or_error** (bool, optional): Whether to abort the
  process on timeout or error
- **timeout** (timedelta, optional): Default timeout for operations
- **store** (Store, optional): Store for communication between processes
- **name** (str, optional): Communicator name
- **hints** (Dict[str, str], optional): Dictionary of string hints for
  backend-specific options

**Note**: While options classes (SendOptions, BroadcastOptions, etc.) are
available for advanced use cases, the recommended approach is to use the `hints`
and `timeout` keyword arguments directly in the operation methods.

### Reduction Operations

The following reduction operations are available:

- **ReduceOp.SUM**: Sum of elements
- **ReduceOp.PRODUCT**: Product of elements
- **ReduceOp.MIN**: Minimum element
- **ReduceOp.MAX**: Maximum element
- **ReduceOp.BAND**: Bitwise AND
- **ReduceOp.BOR**: Bitwise OR
- **ReduceOp.BXOR**: Bitwise XOR
- **ReduceOp.PREMUL_SUM**: Pre-multiplication sum

## Environment Variables

TorchComm uses the following environment variables for configuration:

- **TORCHCOMM_ABORT_ON_ERROR**: Whether to abort the process on timeout or error
  (default: "true")
- **TORCHCOMM_TIMEOUT_SECONDS**: Default timeout in seconds for operations
  (default: "30.0")

## Examples

### Basic Usage

```python
import torch
import torchcomms

# Create a store for rendezvous
store = torch.distributed.FileStore("/tmp/torchcomm_test", 2)

# Create a communicator
device = torch.device("cuda:0")
comm = torchcomms.new_comm("nccl", device, store=store)

# Get rank and world size
rank = comm.get_rank()
world_size = comm.get_size()

# Create a tensor
tensor = torch.ones(10, device=device) * rank

# Perform an all-reduce operation
comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

# Tensor now contains the sum of all ranks
print(f"Rank {rank}: {tensor}")

# Finalize the communicator
comm.finalize()
```

### Asynchronous Operations

```python
import torch
import torchcomms

# Create a communicator
store = torch.distributed.FileStore("/tmp/torchcomm_test", 2)
device = torch.device("cuda:0")
comm = torchcomms.new_comm("nccl", device, store=store)

rank = comm.get_rank()
world_size = comm.get_size()

# Create tensors
send_tensor = torch.ones(10, device=device) * rank
recv_tensor = torch.zeros(10, device=device)

# Perform asynchronous send and receive
if rank == 0:
    work = comm.send(send_tensor, dst=1, async_op=True)
else:
    work = comm.recv(recv_tensor, src=0, async_op=True)

# Do other work while communication is in progress
# ...

# Wait for the operation to complete
work.wait()

# Finalize the communicator
comm.finalize()
```

### CUDA Graphs Example

```python
import torch
import torchcomms

# Create a communicator
device = torch.device("cuda:0")
comm = torchcomms.new_comm("ncclx", device)

rank = comm.get_rank()
world_size = comm.get_size()

# Create tensors
tensor = torch.ones(10, device=device) * rank

# Capture communication operations in a CUDA graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    # Operations captured in graph mode
    work = comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=True)
    # Note: work.wait() should not be called inside graph capture

# Replay the graph multiple times for optimized performance
for _ in range(10):
    graph.replay()
    # Synchronize after replay to ensure completion
    torch.cuda.current_stream().synchronize()

# Finalize the communicator
comm.finalize()
```

### Custom Options

```python
import torch
import torchcomms

# Create a communicator with custom options
device = torch.device("cuda:0")
comm = torchcomms.new_comm(
    "ncclx",
    device,
    timeout=torch.timedelta(seconds=60.0),
    abort_process_on_timeout_or_error=False,
    hints={
        "torchcomm::ncclx::high_priority_stream": "true",
        "backend_option": "value"
    },
)

# ...
comm.finalize()
```
