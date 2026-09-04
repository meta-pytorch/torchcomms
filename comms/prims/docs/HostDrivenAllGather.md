# Host-Driven, SM-Free AllGather

## Summary

`mcclhostring` is an MCCL-hosted AllGather that launches no GPU kernel. It uses
the GPU Copy Engine for local and NVLink movement and CPU-posted IBRC RDMA for
inter-node movement. The design keeps the model's SMs available while moving
data directly between user buffers without transport staging.

The implementation has three shapes selected from communicator geometry:

- Without an NVLink peer domain, it is a flat inter-rank IB ring.
- With an NVLink peer domain, ranks with the same local rank form an IB rail
  ring and each arrival is fanned out to local peers over NVLink.
- When every rank is in one NVLink peer domain, it bypasses IB entirely and
  performs an NVL-only Copy-Engine gather. When a symmetric multicast receive
  allocation is available, this path uses a multicast VA so one Copy-Engine
  write reaches every destination slot in the NVL domain.

The implementation is in the review stack beginning at
[D112636432](https://www.internalfb.com/diff/D112636432). This document
describes the current 24-commit local lineage, including follow-ups beyond the
16-diff Phabricator dependency chain. None of those commits is landed, so this
is not a production contract until the stack lands.

## Goals and Scope

- Consume no SMs in the AllGather data path or its synchronization.
- Preserve stream ordering for in-place and out-of-place AllGather.
- Use the existing Prims IBRC transport and progress thread.
- Operate directly on the caller's receive allocation without staging buffers.
- Scale inter-node transfers across available NICs and QPs.
- Terminate the current execution path on peer, NIC, or host-wait failure.
- Support flat multi-node, rectangular hierarchical, and NVL-only topologies.

The design does not provide automatic algorithm selection, CUDA graph capture,
ReduceScatter, or AllReduce. The shared engine intentionally owns only
Copy-Engine and synchronization primitives; collective policy remains in MCCL.

## Architecture

| Layer | Location | Responsibility |
| --- | --- | --- |
| IBRC transport and proxy | `comms/prims/transport/ibrc/` | Own command queues and QPs, post verbs, poll completions, and publish errors. |
| `P2pIbrcHostWriter` | `comms/prims/transport/ibrc/P2pIbrcHostWriter.h` | Produce descriptors into one host-mapped `(QP, NIC)` command queue. |
| `P2pIbrcHostLanes` | `comms/prims/transport/ibrc/P2pIbrcHostLanes.h` | Split one transfer across command queues and preserve per-lane ordering. |
| `HostCollectiveEngine` | `comms/prims/HostCollectiveEngine.h` | Enqueue Copy-Engine copies and CUDA stream-memory waits and writes. |
| `AllGatherHostRing` | `comms/mccl/collectives/allgather/AllGatherHostRing.cpp` | Select geometry, manage registrations, run the IB ring or NVL-only path, and enforce cross-call ordering. |

### Host-issued IBRC work

IBRC command queues are host-mapped. `P2pIbrcHostWriter` uses the same
reserve, fill, and release-publish protocol as the device producer, so the
existing CPU proxy drains host-produced descriptors without a separate
transport path.

One writer owns one command queue. Host and device producers must not drive the
same queue concurrently: atomic producer-index updates are memory-safe, but the
queue's backpressure and ordering contract assumes one logical producer.

The writer supports RDMA write, RDMA write with immediate notification,
standalone remote atomic signal, local completion counters, and queue fences.
Its polling, fence, and backpressure waits check IBRC error state, an optional
communicator-abort predicate, and a deadline. The host-ring's direct-counter
and bridge-flag waits also check abort and deadline. All default to ten minutes
when the caller does not provide an operation timeout.

### Multi-lane striping

A single command queue is limited to one NIC and one QP. The host-lanes wrapper
stripes a transfer over multiple existing IBRC rings, with consecutive lanes
alternating NICs before reusing a NIC with another QP.

Each lane carries a deterministic, 128-byte-aligned byte range. Transfers use
only as many lanes as can carry at least 256 KiB each; smaller transfers
collapse to one lane. The configured width is bounded by the peer's available
host lanes, 128 total lanes, and `numNics * MCCL_HOSTRING_QPS_PER_NIC`. The
current checkout defaults to 8 QPs per NIC.

Completion is per lane. Different lanes use different QPs and potentially
different NICs, so completion on one lane says nothing about another lane's
payload. The receiver waits on every lane used by the transfer, and the sender
fences every lane before announcing that its source buffer can be reused.

### Copy-Engine orchestration

`HostCollectiveEngine` wraps device-to-device `cudaMemcpyAsync`,
`cuStreamWaitValue64`, and `cuStreamWriteValue64`. It does not own a stream or
the ring algorithm.

The main engine runs on the caller's stream. This orders the initial copy after
the producer of `sendbuff` and orders later caller work after the gather without
an end-of-call host synchronization. Hierarchical fanout may run on a separate
non-blocking stream and joins the caller stream with an event before return.

The NVL-only path also uses Copy-Engine work. It opens remote receive slots
with fabric handles because CUDA IPC handles do not cover cross-host peers in
one MNNVL domain. When MCCL can resolve the caller's receive allocation to a
registered multicast VA, the source chunk is copied once to the multicast VA
for that chunk offset. Otherwise the path uses the existing fabric-handle peer
mapping and unicast Copy-Engine copies.

## Algorithm

### Geometry

`HierRingGeometry` is shared with the SM-based hierarchical ring so both
algorithms accept the same topology and buffer layout. A supported IB-ring
communicator must have:

- More than one IB position.
- A rectangular topology with the same local-rank count on every node.
- Contiguous node-major rank numbering.
- NVLink transport to every local peer when the local domain has multiple
  ranks.
- No more than 72 ranks in one NVLink domain for this path.

The AllGather accepts disjoint send and receive buffers or the conventional
in-place layout where `sendbuff` is exactly this rank's receive slot. Other
overlap is rejected.

### NVL-only gather

When all ranks are in one NVLink domain, the communicator has no IB rail peers.
The host-ring implementation detects that `ibSize == 1` and takes an NVL-only
path instead of rejecting the topology. Every rank copies its own send chunk
into its receive slot when needed, then copies that chunk to the corresponding
receive slot on every other rank in the NVL domain.

The preferred implementation is Copy-Engine multicast. MCCL asks the common
registration path for the multicast VA that covers the caller's receive
allocation and chunk offset, then issues one `cudaMemcpyAsync` from the source
slot to that multicast VA. Hardware replicates the write to the participating
NVL peers. This keeps the path SM-free; no multicast kernel is launched.

If a multicast VA is not available, the same NVL-only algorithm can fall back
to fabric-handle unicast copies. That fallback is correct but does one
Copy-Engine copy per destination rank and is expected to lag the multicast path
on large NVL domains.

The NVL-only path still uses stream ordering rather than a host-side terminal
synchronization. The caller stream orders the source copy, multicast or unicast
fanout, and subsequent user work.

### Flat ring

When `nvlSize == 1`, every rank is one IB position. Each rank copies its own
chunk into its receive slot when necessary, then performs `nRanks - 1` steps:

1. Write the current chunk to the down-peer over striped IBRC lanes.
2. Wait for the up-peer's corresponding arrival on every used lane.
3. Forward that newly arrived chunk in the next step.

There is no global barrier. Monotonic per-lane arrivals provide point-to-point
progress, avoiding coupling every rank to the slowest rank.

### Hierarchical ring

When `nvlSize > 1`, ranks with the same local rank form one IB rail. The rail
performs `nNodes - 1` inter-node steps rather than `nRanks - 1` steps. At each
step:

1. The host posts a striped RDMA write of one rail chunk to the down-peer.
2. The caller and fanout streams wait for every lane of the up-peer arrival.
3. The host is notified that the arrived chunk is safe to forward.
4. The fanout stream copies the chunk into every local peer's receive buffer.

All local rails run independently. Together their NVLink fanouts populate the
complete AllGather result on every local rank.

The fanout stream is enabled by default through
`MCCL_HOSTRING_FANOUT_STREAM`. Keeping its copies off the caller stream lets
the next IB arrival bridge proceed without waiting behind the previous local
fanout. A final event joins fanout back into the caller stream.

## Arrival Signaling

The default data-arrival path uses IB `WRITE_WITH_IMM`. Every completion
increments the receiver's monotonic per-lane counter; call and step thresholds,
not the immediate-data payload, determine which arrival is awaited. The
receiver preposts zero-SGE receives and the IBRC proxy updates a per-lane,
host-mapped signal for each `RECV_RDMA_WITH_IMM` completion. Its device alias
lets CUDA streams wait on the same counter.

By default, the host polls every used lane counter directly, eliminating the
Copy-Engine bridge from the forwarding path. Setting
`MCCL_HOSTRING_DIRECT_HOST_WAIT=0` retains the bridge as an available fallback
for topology-specific performance tuning.

`MCCL_HOSTRING_DATA_WRITE_WITH_IMM=0` selects the fallback path. In that mode,
each lane appends a remote atomic fetch-add to its data write. Arrival counters
live in device memory so both the payload and signal follow the Data Direct
path; a Copy-Engine write bridges stream-visible arrival to a host-pinned flag.

Both paths retain the same per-lane rule: an arrival is usable only after every
lane carrying that chunk has signaled.

## Cross-call Correctness

Data arrival proves delivery, not consumption. Because the collective launches
no kernel, there is no kernel position that implicitly prevents call `k + 1`
from overwriting data still in use by call `k`. Three handshakes replace that
implicit ordering:

1. **Caller stream drained.** At the start of a call, a stream-ordered write to
   mapped host memory proves the caller has finished consuming the previous
   result. Only then may this rank acknowledge its up-peer.
2. **Down-peer ready.** Before the first RDMA write, the caller stream waits for
   the down-peer's acknowledgement from the previous call and bridges it to the
   host. This prevents overwriting a buffer the peer still needs.
3. **NVLink peers ready.** In the hierarchical shape, every local rank publishes
   that it consumed the previous result, then the fanout stream waits locally
   for every peer before writing into peer receive buffers.

A separate end-of-call NVLink handshake ensures every local peer's fanout for
the current call has landed before the caller can observe completion. After all
ring steps, `fence_all()` proves that every NIC has finished reading this
rank's receive buffer. The acknowledgement is deliberately sent at the start
of the next call, after the caller stream has consumed the result, rather than
at the end of the current call.

All counters are monotonic within a cache epoch. Expected values include the
call number, so a delayed signal from an earlier call cannot satisfy a later
wait.

Correctness requires the caller to consume `recvbuff` on the stream passed in
`AllGatherOpts`. Consumption on an unrelated stream is not ordered before the
next call's acknowledgement.

## Registration and Resource Lifetime

For IB-ring shapes, the NIC accesses the receive allocation directly. The send
buffer is not registered because only the Copy Engine reads it.

The per-communicator cache registers and exchanges the entire CUDA allocation
containing `recvbuff`, rather than one call's slice. This supports callers that
rotate receive-buffer offsets within a stable allocation. The cache key is:

```text
allocation base, allocation size, per-rank send size, NVLink domain size
```

Per-rank send size is part of the key because it changes how many lanes are
active and therefore which counters must advance. A cache rebuild also
materializes the rail peers, exchanges remote keys, creates signal and
acknowledgement storage, maps NVLink peer buffers, and creates the auxiliary
streams and event.

The current implementation assumes every rank makes the same cache hit or
rebuild decision, uses symmetric allocation behavior, and places `recvbuff` at
the same offset within its allocation. These restrictions are not enforced.
Integrating with MCCL's common registration cache would provide explicit
free-time invalidation and a place to coordinate per-rank registration state.

For the NVL-only multicast shape, the preferred source of registration state is
the common MCCL/CTRAN window that already covers the caller's receive
allocation. Hostring resolves the registered allocation base and size to the
window's multicast VA and applies the per-call chunk offset at use time. If
that registered multicast mapping is absent, hostring can create its own
fabric-handle peer mappings and use unicast Copy-Engine copies as a fallback.

## Failure Handling

Host waits on the current host-ring execution path terminate on timeout or
communicator abort; writer waits additionally observe IBRC NIC errors. The MCCL
entry point catches transport and CUDA exceptions and converts them to
`commInternalError` rather than allowing an exception across the NCCL-compatible
C ABI.

The recovery path releases immediate-mode arrival waits with a host write and
atomic-mode arrival plus acknowledgement waits from a dedicated non-blocking
stream before invalidating cached resources. The current hierarchical
implementation does not yet release its `freeDev` and `doneDev` waits or
quiesce the fanout stream, so a peer failure can still wedge cleanup. It also
does not clear the transport's WRITE_WITH_IMM receive target before freeing its
host-mapped backing. These are landing blockers for complete fault tolerance.

Changing the cache key can also invalidate resources while asynchronous work
from the previous call still references them. A cache rebuild must retire or
defer those resources before supporting back-to-back calls that change buffer
allocation or message size.

## Selection and Requirements

Select the algorithm with:

```text
NCCL_ALLGATHER_ALGO=mcclhostring
NCCL_CTRAN_USE_PIPES=1
MCCL_IB_MODE=ibrc
```

`NCCL_CTRAN_ENABLE` is not required. MCCL owns dispatch for this name before
CTRAN or registered-algorithm selection. Builds without Prims support report
the algorithm unsupported.

The current support probe assumes the transport was created in IBRC mode and
can throw when the default IBGDA backend is active. It should use a non-throwing
IBRC capability check and report unsupported instead.

The following development controls are not selection APIs:

- `MCCL_HOSTRING_DATA_WRITE_WITH_IMM=0` uses atomic arrival signaling.
- `MCCL_HOSTRING_FANOUT_STREAM=0` keeps fanout on the caller stream.
- `MCCL_HOSTRING_QPS_PER_NIC=<N>` changes the maximum QPs used per NIC.
- `MCCL_HOSTRING_NVL_CE_MULTICAST=0` disables Copy-Engine multicast in the
  NVL-only path.
- `MCCL_HOSTRING_NVL_MULTIMEM=0` disables multicast-VA resolution and forces
  the unicast fabric-handle fallback.

For the NVL-only multicast path, the receive allocation must be registered in
the symmetric multicast-capable path used by the NCCL test suite's `-R 2 -M 1`
mode. The data path itself does not require IB peers.

## Performance Evidence

Copy-Engine microbenchmarks established the feasibility of replacing SM copy
kernels: on H100, large local copies reached about 1.1 TB/s and NVLink peer
copies reached about 397 GB/s, matching or exceeding the measured SM-copy path.

The following GB300 results were recorded on 2026-08-25. They predate the
current checkout's default WRITE_WITH_IMM signaling and 8-QP-per-NIC setting,
so they motivate the design but do not validate the current tip. Sizes are per
rank and bandwidth is AllGather bus bandwidth.

| Topology | Size | `mcclhostring` | Comparison |
| --- | ---: | ---: | ---: |
| 16 nodes x 4 GPUs | 4 MiB | 180 GB/s | `ctwin`: 125 GB/s |
| 16 nodes x 4 GPUs | 16 MiB | 265 GB/s | `ctwin`: 188 GB/s |
| 16 nodes x 4 GPUs | 64 MiB | 352 GB/s | `ctwin`: 209 GB/s |
| 16 nodes x 1 GPU | 4 MiB | 52 GB/s | `ctring`: 70 GB/s |
| 16 nodes x 1 GPU | 16 MiB | 80 GB/s | `ctring`: 93 GB/s |
| 16 nodes x 1 GPU | 64 MiB | 91 GB/s | `ctring`: 96 GB/s |

The hierarchical path was fastest among the measured algorithms from 2 MiB
per rank upward. The flat path approached the SM ring as message size grew,
reaching about 95% of `ctring` bandwidth at 64 MiB. Results from the two
topologies are not directly comparable because they gather different total
volumes.

The following GB300 NVL-only results were recorded on 2026-09-03 with
`MCCL_HOSTRING_NVL_CE_MULTICAST=1` and `MCCL_HOSTRING_NVL_MULTIMEM=1`.
Sizes are per rank. Bandwidth is average AllGather bus bandwidth from
successful correctness-checked samples.

| Topology | Size | `mcclhostring` CE multicast | NCCL | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 NVL domain x 16 GPUs | 64 MiB | 702.2 GB/s | 702.9 GB/s | 1.00x |
| 1 NVL domain x 16 GPUs | 256 MiB | 731.8 GB/s | 735.7 GB/s | 1.00x |
| 1 NVL domain x 32 GPUs | 64 MiB | 733.8 GB/s | 698.8 GB/s | 1.05x |
| 1 NVL domain x 32 GPUs | 256 MiB | 668.3 GB/s | 729.4 GB/s | 0.92x |

The 1x32, 256 MiB/rank case remains below NCCL and needs follow-up before using
the multicast path as a complete performance replacement for NCCL on every NVL
shape.

## Testing and Completion Criteria

Existing coverage:

- `P2pIbrcHostLanesTest.cc` unit-tests deterministic splitting, alignment,
  threshold collapse, full byte coverage, and non-empty lanes.
- `HostAllGatherTwoNodeTest.cc` exercises host-issued put, fence, atomic signal,
  and counter completion over two IB peers.
- `HostAllGatherHierTest.cc` exercises a 4-by-2 logical topology with IB rails,
  NVLink fanout, and atomic arrival synchronization.
- `AllGatherHostRingTwoNodeTest.cpp` runs back-to-back calls with distinct
  payload generations and no barrier between calls, then compares supported
  algorithms across message sizes. Despite its name, the cross-call test needs
  at least three ranks to exercise forwarding safely.
- Manual RTP NVL-only runs exercise 1x16 and 1x32 topologies with
  Copy-Engine multicast enabled and correctness checking on.

The multi-node tests are manual hardware tests rather than ordinary remote
execution tests. The current tip has only build and lint coverage for the
WRITE_WITH_IMM transport change, and its host-ring integration is recorded as
not run.

Before landing:

- Run cross-call and rotating-buffer correctness on flat and hierarchical
  topologies with both immediate and atomic signaling, and on NVL-only
  topologies with multicast enabled and disabled.
- Exercise abort during arrival, acknowledgement, `freeDev`, and `doneDev`
  waits, including cache cleanup.
- Repeat performance sweeps for the current 8-QP-per-NIC configuration and
  validate the idle-command-queue optimization under the same load.
- Investigate the 1x32, 256 MiB/rank multicast gap against NCCL.
- Run targeted Buck builds and tests, `arc lint`, the required local rcclx ROCm
  package build, and the H100/GB200 device-API MAST matrix.
- Resolve the cache-coherence and resource-retirement restrictions above, or
  enforce them as explicit API preconditions.

## Limitations and Future Work

- The host thread still blocks while bridging GPU progress and posting the next
  RDMA step. Fully stream-ordered host execution remains future work.
- The ring advances one inter-node chunk at a time. A deeper credit-window
  pipeline could overlap more steps.
- The NVL-only unicast fallback performs one Copy-Engine copy per destination
  rank. It is a correctness fallback, not the expected high-performance path
  for large NVL domains.
- Calls on one communicator are assumed to be serialized; the mutable cache,
  call counter, and stream rebinding have no internal locking.
- `mcclhostring` shares `Algorithm::Ring` with the fused ring and therefore uses
  a bespoke dispatch branch. It should receive its own MCCL algorithm enum.
- Explicit host-ring operations are currently labeled `ctran` in MCCL operation
  traces, and host-ring failures do not attach their error to the trace guard.
- ReduceScatter and AllReduce can be built from the same host-driven primitives.
- Runtime selection should consider message size, topology, and SM pressure.

## References

- [Implementation stack beginning at D112636432](https://www.internalfb.com/diff/D112636432)
- [Host IBRC WRITE_WITH_IMM support (D118428297)](https://www.internalfb.com/diff/D118428297)
- [Host-ring WRITE_WITH_IMM integration (D118428298)](https://www.internalfb.com/diff/D118428298)
- [NVL-only hostring path (D118700363)](https://www.internalfb.com/diff/D118700363)
- [CE multicast for NVL-only hostring (D118700364)](https://www.internalfb.com/diff/D118700364)
- [Original design and experiment log](https://docs.google.com/document/d/1HaEG5SmauQREa4XLVCxQuveTZPp_ZfovqIoebX3BhKs/edit)
