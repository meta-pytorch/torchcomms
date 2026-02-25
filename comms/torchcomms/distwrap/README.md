# distwrap - Distributed Communication Wrapper

`distwrap` provides a `torch.distributed`-compatible API that can optionally
route communication operations through torchcomms instead of native
torch.distributed backends.

## Overview

The module creates wrapper functions for the torch.distributed API (with minor
extensions) that internally dispatch to either torch.distributed or torchcomms
based on configuration. It maintains a registry that maps ProcessGroups to
their associated torchcomms instances.

## Architecture

```text
+-------------------------------------------------------------------------+
|                           User Code                                     |
|         from torchcomms import distwrap as dist                         |
|         dist.all_reduce(tensor)                                         |
+--------------------------------+----------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
|                      distwrap.collectives                               |
|                                                                         |
|   def all_reduce(tensor, ...):                                          |
|       if torchcomms_is_enabled():                                       |
|           tc = get_torchcomms_instance(pg, tensor)  <-- Registry lookup |
|           return tc.all_reduce(...)                                     |
|       else:                                                             |
|           return dist.all_reduce(...)                                   |
+--------------------------------+----------------------------------------+
                                 |
              +------------------+------------------+
              |                                     |
              v                                     v
+-------------------------+           +-------------------------+
|     torchcomms          |           |   torch.distributed     |
|   (use_torchcomms=True) |           |  (use_torchcomms=False) |
+-------------------------+           +-------------------------+
```

## Module Structure

- `__init__.py` - Public API exports and attribute forwarding to torch.distributed
- `new_comm.py` - Process group management (init, new_group, split_group, destroy)
- `collectives.py` - Standard collective operations (all_reduce, broadcast, etc.)
- `collectives_extension.py` - torchcomms-only extensions (window ops, alltoallv variants)
- `pginfo.py` - ProcessGroup registry for metadata and torchcomms instances
- `utils.py` - Backend parsing, instance lookup, and helper functions

## ProcessGroup Registry

The core mechanism is a registry (`pginfo.py`) that maps `ProcessGroup` objects
to metadata and torchcomms instances:

```python
@dataclass
class _PG_INFO:
    global_ranks: list[int]    # Global ranks in this group
    group_desc: str            # Group description/name
    data: dict[str, Any]       # Arbitrary data (stores torchcomms instances)

# Registry: ProcessGroup -> _PG_INFO
_PG_INFO_REGISTRY: dict[ProcessGroup, _PG_INFO] = {}
```

When torchcomms is enabled, the `data` dictionary stores:
- `"torchcomms"`: Dict mapping device type (e.g., "cuda", "cpu") to TorchComm instance
- `"device_backends"`: Dict mapping device type to backend name (e.g., "cuda" -> "nccl")

## Usage

### Basic Usage (torch.distributed passthrough)

```python
from torchcomms import distwrap as dist

# Initialize without torchcomms - all calls pass through to torch.distributed
dist.init_process_group(backend="nccl")

# These call torch.distributed directly
dist.all_reduce(tensor)
dist.broadcast(tensor, src=0)
```

### With torchcomms Backend

```python
from torchcomms import distwrap as dist

# Initialize with torchcomms enabled
dist.init_process_group(backend="nccl", use_torchcomms=True)

# These now route through torchcomms
dist.all_reduce(tensor)  # Uses torchcomms.all_reduce internally
dist.broadcast(tensor, src=0)

# Direct torch.distributed calls are blocked when torchcomms is enabled
# import torch.distributed as torch_dist
# torch_dist.all_reduce(tensor)  # Raises NotImplementedError
```

### Creating Subgroups

```python
# With torchcomms, use split_group instead of new_group
subgroup = dist.split_group(
    split_ranks=[[0, 1], [2, 3]],
    group_desc="my_subgroup"
)

# Operations on subgroups use the split torchcomms instance
dist.all_reduce(tensor, group=subgroup)
```

### torchcomms-only Extensions

```python
# Window operations (for RMA/one-sided communication)
window = dist.new_window(group=pg)

# Specialized alltoallv variants for MoE workloads
persist_req = dist.alltoallv_dedup_init(...)
dist.alltoallv_dedup_exec(..., persist_req)
```

## Initialization Flow

When `init_process_group(use_torchcomms=True)` is called:

1. **Parse backend string**: Converts backend spec (e.g., "nccl", "cpu:gloo,cuda:nccl")
   to device->backend mapping

2. **Initialize torch.distributed**: Calls `dist.init_process_group()` with the
   appropriate backend (ncclx/rcclx are renamed to nccl/rccl for compatibility)

3. **Register world group**: Creates a `_PG_INFO` entry for `dist.group.WORLD`

4. **Create torchcomms instances**: For each device type, creates a `TorchComm`
   instance and stores it in the registry

5. **Block direct dist calls**: Patches torch.distributed collectives to raise
   `NotImplementedError` if called directly

## Collective Dispatch Logic

Each collective wrapper follows this pattern:

```python
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    pg = get_group(group)  # Default to WORLD if None

    if torchcomms_is_enabled():
        # Look up torchcomms instance based on tensor's device type
        tc = get_torchcomms_instance(pg, tensor=tensor)
        work = tc.all_reduce(tensor, op, async_op)
        return work if async_op else None
    else:
        # Pass through to torch.distributed
        return dist.all_reduce(tensor, op, pg, async_op)
```

## Supported Operations

### Standard Collectives
- `all_reduce`, `broadcast`, `reduce`
- `all_gather`, `all_gather_into_tensor`
- `reduce_scatter`, `reduce_scatter_tensor`
- `all_to_all`, `all_to_all_single`
- `scatter`, `gather`
- `barrier`

### Point-to-Point
- `send`, `recv`
- `isend`, `irecv`
- `batch_isend_irecv`

### Object Collectives
- `all_gather_object`, `gather_object`
- `scatter_object_list`, `broadcast_object_list`

### Extensions (torchcomms-only)
- `new_window` - Create RMA window
- `alltoallv_dedup_init/exec` - Deduplicated alltoallv for MoE
- `alltoallv_dynamic_dispatch/combine` - Dynamic alltoallv for MoE

## Limitations When torchcomms is Enabled

The following operations that work with torch.distributed will raise exceptions
when `use_torchcomms=True`:

### new_group() is not supported

Use `split_group()` instead. torchcomms requires splitting from an existing
communicator rather than creating arbitrary new groups.

```python
# This raises AssertionError when torchcomms is enabled:
subgroup = dist.new_group(ranks=[0, 1])

# Use split_group instead:
subgroup = dist.split_group(split_ranks=[[0, 1], [2, 3]])
```

### Wildcard recv/irecv (src=None) is not supported

torchcomms requires an explicit source rank for receive operations.

```python
# These raise ValueError when torchcomms is enabled:
dist.recv(tensor, src=None)   # Wildcard recv
dist.irecv(tensor, src=None)  # Wildcard irecv

# Specify explicit source rank instead:
dist.recv(tensor, src=0)
dist.irecv(tensor, src=0)
```

### Direct torch.distributed calls are blocked

When torchcomms is enabled, direct calls to torch.distributed collective
functions are patched to raise `NotImplementedError`. This prevents accidental
bypassing of the distwrap layer.

```python
import torch.distributed as torch_dist

# These raise NotImplementedError when torchcomms is enabled:
torch_dist.all_reduce(tensor)
torch_dist.broadcast(tensor, src=0)
torch_dist.send(tensor, dst=1)
# ... and all other collective operations

# Use distwrap instead:
from torchcomms import distwrap as dist
dist.all_reduce(tensor)
```

### Object collectives use an arbitrary torchcomms instance

For object-based collectives (`all_gather_object`, `gather_object`,
`scatter_object_list`, `broadcast_object_list`), there is no tensor to infer
the device type from. In this case, distwrap picks the first available
torchcomms instance arbitrarily. If you have multiple backends configured
(e.g., both CPU and GPU), the chosen backend may not be what you expect.

### split_group and new_window operate on all backends by default

For `split_group`, there is no way to infer which backend to use. By default,
it creates a new torchcomms instance for each backend that was configured
during `init_process_group` (e.g., both NCCL and Gloo if both were initialized).

To limit which backends are used, pass the `backend` parameter explicitly:

```python
# Split only on NCCL backend
subgroup = dist.split_group(
    split_ranks=[[0, 1], [2, 3]],
    backend="nccl"
)
```

### new_window uses an arbitrary torchcomms instance by default

Similar to object collectives, `new_window` picks the first available
torchcomms instance from the group when `backend` is not specified. Use the
`backend` parameter to explicitly choose which backend to use:

```python
# Create window on specific backend
window = dist.new_window(group=pg, backend="ncclx")
```

## Forwarded Attributes

The module forwards these attributes directly from torch.distributed:

- `get_rank`, `get_world_size`
- `is_initialized`, `is_available`
- `get_process_group_ranks`
- `ProcessGroup`, `ProcessGroupNCCL`, `GroupMember`
- `ReduceOp`, `P2POp`, `group`, `Store`, `HashStore`, `Work`
