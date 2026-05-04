# ClogHook Log Format — AI Tool Instructions (V1)

This document describes the ClogHook log format for AI tools that parse
or analyze clog output. ClogHook is a hook-based logger for TorchComm
collective operations that records operation signatures, enqueue events,
and GPU lifecycle events to a pipe-delimited log file.

## Overview

ClogHook attaches to one or more TorchComm communicators via pre-hook
and post-hook callbacks. Each collective operation is assigned a
correlation ID (`corr_id`) based on its signature (operation type,
parameters, and communicator). Repeated calls with identical signatures
reuse the same `corr_id`, providing natural deduplication.

The log captures four categories of information:
1. **Signatures** — unique collective operation definitions
2. **Enqueue events** — when a collective is submitted (always logged)
3. **Lifecycle events** — GPU start, end, and CPU wait (when enabled
   via `LIFECYCLE` or `ALL` event options, which are equivalent)
4. **Communicator management** — creation, splits

Each rank in a distributed job writes its own log file. There is no
cross-rank coordination.

## Line Format

All lines use `|` as the field delimiter. Every event line ends with a
timestamp field `+<ts>` representing seconds elapsed since the base
timestamp in the version header.

## Version Header

The first line of every log file:

```
V|<version>|base_timestamp=<epoch_seconds>
```

- `version`: integer log format version (currently 1)
- `base_timestamp`: wall-clock time in seconds (millisecond precision)
  when the ClogHook was constructed

All `+<ts>` fields in subsequent lines are deltas relative to
`base_timestamp`. To reconstruct absolute wall-clock time:
`absolute_time = base_timestamp + ts`.

## Signature Lines

```
C<corr_id>|sig|<op>|<param1>=<val1>|<param2>=<val2>|...|comm=<comm_name>
```

Each unique combination of (operation, parameters, communicator) produces
exactly one signature line. The `corr_id` is a positive integer assigned
sequentially. All subsequent event lines for the same collective
reference this `corr_id`.

### Signature Parameters by Operation Type

**In-place collectives** (tensor is both input and output):
- `all_reduce`: `in_count`, `out_count`, `dtype`, `red_op`, `async_op`
- `broadcast`: `in_count`, `out_count`, `dtype`, `root`, `async_op`
- `reduce`: `in_count`, `out_count`, `dtype`, `red_op`, `root`, `async_op`

**Point-to-point**:
- `send`: `in_count`, `out_count=0`, `dtype`, `peer`, `async_op`
- `recv`: `in_count=0`, `out_count`, `dtype`, `peer`, `async_op`

**Distinct input/output (single tensors)**:
- `all_gather_single`: `in_count`, `out_count`, `dtype`, `async_op`
- `all_to_all_single`: `in_count`, `out_count`, `dtype`, `async_op`
- `reduce_scatter_single`: `in_count`, `out_count`, `dtype`, `red_op`, `async_op`
- `all_to_all_v_single`: `in_splits`, `out_splits`, `dtype`, `async_op`

**Input tensor to output tensor list**:
- `all_gather`: `in_count`, `out_counts`, `dtype`, `async_op`
- `all_gather_v`: `in_count`, `out_counts`, `dtype`, `async_op`

**Input tensor list to output tensor**:
- `reduce_scatter`: `in_counts`, `out_count`, `dtype`, `red_op`, `async_op`
- `reduce_scatter_v`: `in_counts`, `out_count`, `dtype`, `red_op`, `async_op`

**Tensor lists on both sides**:
- `all_to_all`: `in_counts`, `out_counts`, `dtype`, `async_op`

**Scatter/gather with root**:
- `scatter`: `in_counts`, `out_count`, `dtype`, `root`, `async_op`
- `gather`: `in_count`, `out_counts`, `dtype`, `root`, `async_op`
- `gather_single`: `in_count`, `out_count`, `dtype`, `root`, `async_op`

**Special**:
- `barrier`: `async_op`
- `batch_op_issue`: `num_ops`, `async_op`

### Parameter Value Formats

- `count`, `in_count`, `out_count`: integer element count
- `in_counts`, `out_counts`: comma-separated integer list (one per rank)
- `in_splits`, `out_splits`: comma-separated integer list
- `dtype`: short name — `f32`, `f64`, `f16`, `bf16`, `i32`, `i64`,
  `i16`, `i8`, `u8`, `bool`, `fp8e4m3`, `fp8e5m2`, `other`
- `red_op`: `sum`, `prod`, `min`, `max`, `band`, `bor`, `bxor`,
  `premul_sum`, `avg`, `unknown`
- `async_op`: `t` (true) or `f` (false)
- `root`, `peer`: integer rank
- `num_ops`: integer count of batched operations
- `comm`: communicator name string

### Verbose Fields

When `buffers` verbose mode is enabled, signature lines include
additional pointer fields:
- `buf=<hex>`: buffer address (in-place collectives)
- `in=<hex>`: input buffer address
- `out=<hex>`: output buffer address
- For tensor lists: comma-separated hex addresses

### Signature Deduplication

A signature is keyed by the full parameter string including the comm
name. Two collectives with identical parameters on different
communicators produce different signatures (different `corr_id` values).
Two collectives with identical parameters on the same communicator share
a single signature and `corr_id`.

This means the `corr_id` identifies the *type* of collective, not a
specific *instance*. Multiple Q/S/E/W event sequences may reference the
same `corr_id`.

## Event Lines

### Non-graph events

```
C<corr_id>|Q|work_id=<work_id>|+<ts>   # enqueue
C<corr_id>|S|+<ts>                      # start (GPU kernel launched)
C<corr_id>|E|+<ts>                      # end (GPU kernel completed)
C<corr_id>|W|+<ts>                      # wait (CPU waited on this work)
```

**Q** (enqueue): Always logged in the pre-hook when the collective is
submitted. This is the earliest observable event for a collective
instance. Each Q event includes a `work_id` that uniquely identifies
this specific collective instance (monotonically increasing). The
`work_id` can be used to correlate Q events with lifecycle warnings.

**S** (start): Logged when the GPU begins executing the collective.
For NCCL-based backends, this corresponds to the start CUDA event
being recorded on the collective's stream.

**E** (end): Logged when the GPU finishes executing the collective.
Corresponds to the end CUDA event completing.

**W** (wait): Logged when the CPU explicitly waits for the collective
to complete (via `work->wait()`). Not all collectives have a W event —
synchronous collectives may complete without an explicit wait.

### Graph capture events

During CUDA graph capture, events are prefixed with the graph ID:

```
G<graph_id>|C<corr_id>|Q|work_id=<work_id>|+<ts>    # enqueue during capture
G<graph_id>|C<corr_id>|S|+<ts>                       # start during capture
G<graph_id>|C<corr_id>|E|+<ts>                       # end during capture
G<graph_id>|C<corr_id>|W|+<ts>                       # wait during capture
```

`graph_id` is assigned by the CUDA runtime and is globally unique across
all graphs in the process (per the CUDA spec). ClogHook detects graph
capture by querying `cudaStreamGetCaptureInfo` on the current stream.

During capture, the collective actually executes once (to record the
graph nodes). The Q/S/E/W events logged during capture reflect this
single execution.

### Graph replay events

When a captured graph is replayed, the watchdog thread detects
completion of graph-captured collectives and fires replay events:

```
G<graph_id>|R<replay_id>|C<corr_id>|S|+<ts>    # start during replay
G<graph_id>|R<replay_id>|C<corr_id>|E|+<ts>    # end during replay
```

- `replay_id`: monotonically increasing counter per graph, starting
  at 1 for the first replay after capture.
- Replay events only include S and E. There is no Q (the graph launch
  replays all captured operations atomically) and no W (waits are not
  observable per-collective during replay).

## Communicator Events

```
new_comm|comm=<name>|rank=<rank>|world_size=<size>
```

Logged when `registerWithComm()` is called. Records the communicator
name, this rank's position, and the total number of ranks.

```
split|parent=<parent_comm>|child=<child_comm>|ranks=<r0>,<r1>,...
```

Logged when a communicator split occurs. `ranks` lists the global ranks
participating in the child communicator.

## Warning Lines

```
WARN|<description>
```

Warnings indicate unexpected states, such as:
- A post-hook firing without a matching pre-hook
- A lifecycle event for an unknown work ID
- A graph replay event for an unknown graph or stream

Warnings do not affect program execution — they are diagnostic only.

## Ordering Guarantees

### What IS ordered

1. **Signature before events**: A signature line for a `corr_id` always
   appears before any Q/S/E/W lines referencing that `corr_id`.

2. **Q before S/E/W for a non-graph collective instance**: Within a
   single non-graph collective, the Q event is logged first (in the
   pre-hook on the calling thread). S/E/W events follow, though S and
   E are logged from callback threads and W from the waiting thread.

3. **Within a single CUDA stream during capture**: Collectives captured
   on the same CUDA stream have their Q events logged in capture order.
   This ordering reflects the stream's execution order in the graph.

4. **Lifecycle ordering per collective**: For any single collective
   instance, S occurs before E on the GPU. W may occur at any point
   after Q (it depends on when the user calls wait).

### What is NOT ordered

1. **Events across different CUDA streams**: Collectives on different
   streams execute independently on the GPU. Their S/E events in the
   log may appear in any order relative to each other. For example, a
   collective on stream A may log E before a collective on stream B
   logs S, even if B was enqueued after A.

2. **Graph replay events across communicators**: Each communicator has
   its own watchdog thread that detects replay completions and fires
   hooks independently. For a graph involving multiple communicators,
   replay events from different communicators may arrive in any order
   and may be arbitrarily interleaved. **Treat replay events as "these
   events have occurred" rather than "this is the order in which they
   occurred."**

3. **Graph replay events within a communicator across streams**: When a
   communicator has collectives on multiple streams within the same
   graph, the watchdog iterates streams independently. Events from
   different streams within the same communicator may appear out of
   order.

4. **Events across ranks**: Each rank writes its own independent log
   file. There is no synchronization or ordering relationship between
   lines in different rank log files. Use `base_timestamp + ts` for
   approximate cross-rank time correlation, but note that clock skew
   between nodes limits precision.

5. **Interleaving of different collective instances**: When multiple
   collectives are in flight concurrently (e.g., async operations),
   their lifecycle events may interleave. A sequence like
   `C1|S, C2|S, C1|E, C2|E` is normal and indicates overlapping
   execution.

6. **Graph capture lifecycle vs replay lifecycle**: During capture,
   S/E/W events are delivered by the eager execution callbacks. During
   replay, S/E events are delivered by the watchdog thread polling CUDA
   events. These are fundamentally different delivery mechanisms with
   different latencies. Do not compare capture-time and replay-time
   timestamps as if they measure the same thing.

## Graph Structure Reconstruction

To reconstruct the internal structure of a captured CUDA graph from the
log:

1. **Identify captured collectives**: Collect all
   `G<graph_id>|C<corr_id>|Q|work_id=<work_id>|+<ts>` lines for the
   target graph.

2. **Resolve signatures**: For each `corr_id`, find the corresponding
   `C<corr_id>|sig|...` line to determine the operation type,
   parameters, and communicator.

3. **Determine stream assignment**: The `async_op` field in the
   signature indicates stream behavior:
   - `async_op=f` (sync): The collective executes on the user's current
     CUDA stream at capture time.
   - `async_op=t` (async): The collective executes on the communicator's
     internal stream. An implicit dependency exists from the user stream
     to the internal stream (to transfer the operation), and a
     subsequent wait creates a dependency back.

4. **Infer ordering**: Within a single stream, Q events appear in
   capture order, which is the execution order. Across streams, there
   is no inherent ordering unless dependencies are present.

5. **Correlate replays**: For each `G<graph_id>|R<replay_id>|...` line,
   the `replay_id` identifies which replay execution the event belongs
   to. All events with the same `(graph_id, replay_id)` belong to the
   same graph replay.

## Async Operation Execution Model

When `async_op=t`, the collective follows this execution path:

1. The user calls the collective on their current CUDA stream.
2. The backend records a dependency event on the user stream.
3. The backend's internal stream waits on that dependency.
4. The collective executes on the internal stream.
5. When the user calls `wait()`, a dependency is created from the
   internal stream back to the user's stream (or whichever stream the
   wait occurs on).

In graph capture mode, these stream dependencies become edges in the
captured CUDA graph. During replay, the GPU executes the graph
respecting these dependencies without CPU involvement.

In the log, async operations appear as regular Q/S/E/W events. The
stream assignment (user stream vs internal stream) is not explicitly
logged but can be inferred from the `async_op` flag and the
communicator name.

## Multiple Communicators

A single ClogHook instance can be registered with multiple
communicators. All communicators write to the same log file. The `comm`
field in signature lines identifies which communicator each collective
belongs to.

When multiple communicators participate in the same CUDA graph:
- Each communicator's collectives are captured independently.
- During replay, each communicator's watchdog fires events
  independently.
- The log may interleave events from different communicators for the
  same graph replay.

## Timestamp Precision

Timestamps are recorded using `std::chrono::system_clock` with
millisecond precision, formatted as seconds with three decimal places
(e.g., `+1.234` means 1234 milliseconds after base_timestamp).

The timestamp reflects when ClogHook recorded the event, not when the
GPU operation actually occurred. For S/E events, there is inherent
latency between the GPU completing the operation and the callback
delivering the event to ClogHook.

## Field Reference

| Field | Description |
|-------|-------------|
| `V` | Version header line marker |
| `C<id>` | Correlation ID — links events to their signature |
| `G<id>` | CUDA graph ID (globally unique per CUDA spec) |
| `R<id>` | Replay number within a graph (1-based) |
| `sig` | Signature definition line |
| `Q` | Enqueue event (always logged) |
| `work_id=<id>` | Unique per-instance work ID (in Q events) |
| `S` | Start event (GPU execution began) |
| `E` | End event (GPU execution completed) |
| `W` | Wait event (CPU waited for completion) |
| `+<ts>` | Timestamp as seconds delta from base_timestamp |
| `new_comm` | Communicator creation event |
| `split` | Communicator split event |
| `WARN` | Diagnostic warning |

## Example Log

```
V|1|base_timestamp=1714234567.890
new_comm|comm=world|rank=0|world_size=8
C1|sig|all_reduce|in_count=1024|out_count=1024|dtype=f32|red_op=sum|async_op=f|comm=world
C1|Q|work_id=1|+0.001
C1|S|+0.002
C1|E|+0.005
C2|sig|all_reduce|in_count=2048|out_count=2048|dtype=bf16|red_op=sum|async_op=t|comm=world
C2|Q|work_id=2|+0.010
C2|S|+0.011
C1|Q|work_id=3|+0.012
C2|E|+0.015
C1|S|+0.013
C1|E|+0.016
C2|W|+0.017
G42|C1|Q|work_id=4|+1.000
G42|C2|Q|work_id=5|+1.001
G42|C1|S|+1.002
G42|C2|S|+1.003
G42|C1|E|+1.005
G42|C2|E|+1.006
G42|R1|C1|S|+2.001
G42|R1|C2|S|+2.002
G42|R1|C1|E|+2.004
G42|R1|C2|E|+2.005
G42|R2|C1|S|+3.001
G42|R2|C2|S|+3.002
G42|R2|C1|E|+3.004
G42|R2|C2|E|+3.005
```
