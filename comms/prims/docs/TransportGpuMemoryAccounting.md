# MCCL and Prims GPU Memory Accounting

## Goal

Report physical GPU memory owned directly by MCCL and by its Prims transports,
per communicator, rank, and GPU. Reports include the current total, lifetime
peak, transport breakdown, and resource breakdown.

This does not count user buffers, Ctran-owned allocations, host-mapped memory,
peer mappings, virtual-address reservations, registrations, or provider-owned
QP/CQ backing. The initial IBGDA scope also excludes the CUDA-helper metadata
arrays so no provider/DOCA helper code needs to change.

## Design

`McclComm` owns one `GpuMemoryTracker`. During initialization and native MCCL
collectives, an RAII scope binds it to thread-local storage. Thin `mcclCuda*`
wrappers around the existing host-side CUDA/HIP allocation calls record:

```text
component, resource, accounted_bytes
```

Allocation registers `(backing kind, pointer or VMM handle) -> tracker and
metadata` in a process-wide synchronized registry. Free uses the registry, so
it remains correct if the communicator is later used or destroyed on another
thread. The registry is created only by the first tracked allocation; free and
teardown never initialize it. Transport objects keep their existing memory
ownership; tracker handles and per-allocation tokens are not passed through the
transport stack.

Only successful allocations and frees change accounting. Physical backings
are counted once; mappings, aliases, and logical regions inside packed
allocations do not increase totals. `accounted_bytes` is the size passed to the
CUDA/HIP allocator, including VMM rounding.

The initial resource set is:

```text
mccl.allgather.overlap_counters
common.transport_dispatch_table

nvl.p2p.data_staging
nvl.p2p.signal
nvl.p2p.channel_state
nvl.p2p.channel_progress
nvl.p2p.barrier
nvl.p2p.ll
nvl.p2p.ll128
nvl.p2p.transport_table
nvl.multimem.combined_backing
nvl.multimem.peer_signal_ptr_table

ib.eager.send_buffer
ib.eager.recv_buffer
ib.eager.control_buffer
ib.sendrecv.peer_bulk
ib.slot.signal_inbox
ib.slot.counter
ib.slot.discard_signal

ibgda.atomic_return_sink
ibgda.peer_transport_array
```

## Reporting

With `MCCL_GPU_MEMORY_ACCOUNTING_ENABLE=1`, emit:

1. `bootstrap`, after communicator and transport initialization.
2. `final`, immediately before communicator-owned transport memory is released.

The tracker also exposes a structured snapshot so a future Scuba sink can use
the same data without parsing the log.

```text
MCCL GPU memory [final] commHash=... rank=0 gpu=0
  Prims transport-owned GPU memory:
    common: 4.00 MiB (4194304 bytes)
    nvl: 256.00 MiB (268435456 bytes)
    ib: 32.00 MiB (33554432 bytes)
    ibgda: 48.00 MiB (50331648 bytes)
    total: 340.00 MiB (356515840 bytes)
    peak: 356.00 MiB (373293056 bytes)
  direct MCCL-owned GPU memory: current=8.00 MiB (8388608 bytes) peak=8.00 MiB (8388608 bytes)
  unclassified GPU memory: current=0.00 MiB (0 bytes) peak=0.00 MiB (0 bytes)
  total accounted GPU memory: current=348.00 MiB (364904448 bytes) peak=364.00 MiB (381681664 bytes)
  resource mccl.allgather.overlap_counters: current=8.00 MiB (8388608 bytes) peak=8.00 MiB (8388608 bytes)
  resource common.transport_dispatch_table: current=4.00 MiB (4194304 bytes) peak=4.00 MiB (4194304 bytes)
  resource nvl.multimem.combined_backing: current=256.00 MiB (268435456 bytes) peak=272.00 MiB (285212672 bytes)
  resource ib.sendrecv.peer_bulk: current=32.00 MiB (33554432 bytes) peak=32.00 MiB (33554432 bytes)
  resource ibgda.atomic_return_sink: current=48.00 MiB (50331648 bytes) peak=48.00 MiB (50331648 bytes)
```

Resource rows are emitted only when their current or peak usage is nonzero.

The feature is off by default. Disabled allocation and free paths perform only
TLS/atomic null checks; they create no tracker state, take no lock, format no
message, and perform no GPU query or synchronization. When enabled, registry
locking occurs only at allocation and free, never from a kernel or the
steady-state data path.

## Validation

- Unit-test totals, peaks, duplicate protection, nested scopes, cross-thread
  free, and tracker lifetime.
- Build the NVIDIA and AMD transport paths and the MCCL conda package.
- Compare accounting disabled versus enabled if performance validation is
  required.
