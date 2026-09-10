# PRIMS Lazy Channels

Status: Target design for the dependent implementation changes in this stack

## Summary

The existing peer-lazy connection path creates backend-global NIC/control state
and fixed descriptor tables at communicator initialization, but defers
peer/channel-specific QPs, buffers, CQs, and FIFOs until MCCL first uses a peer.
Without lazy channels, that first use creates every configured channel for the
peer.

Lazy channels add a channel-prefix dimension to the same materialization path:

```text
kEager:
  first positive demand for peer P -> create channels [0, capacity)

kLazyPrefix:
  demand (peer P, N channels) -> create missing channels [oldK, N)
```

For `kLazyPrefix`, PRIMS validates `N <= capacity` and materializes the exact
requested prefix. Channels are always a prefix, never a sparse set. V1 is
grow-only: a peer's materialized prefix can increase but never decrease, and
resources are released only at communicator teardown. For example, a demand
for three channels materializes `[0, 3)`.

`MCCL_PRIMS_LAZY_CHANNELS` selects the mode and defaults to off. There is no
user-facing lazy API.

## Scope and ownership

This design covers the MCCL API -> MCCL launcher/kernel -> PRIMS transport path
for IBGDA and IBRC, plus PRIMS-backed CTRAN launchers using the same readiness
contract.

- MCCL selects the algorithm, peers, and channel counts.
- The MCCL algorithm prepares all required peer/channel prefixes before
  obtaining device transport pointers and launching the kernel.
- PRIMS allocates, exchanges, connects, and publishes transport resources.
- PRIMS kernels consume prepared transport descriptors; they never allocate or
  connect.

Tree, ring, and other algorithms do not own separate transport buffers. They
reuse the same PRIMS resource for a `(peer, channel)` and differ only in which
peers and channel prefixes they request.

NVL lazy storage is out of scope.

## Data model

### Peer/channel readiness

The existing peer materialization state is extended with one host-side
readiness watermark per peer:

```cpp
struct PendingPeer {
  int rank;
  uint32_t targetChannels;
};

std::mutex materializationMutex;
std::vector<PendingPeer> pendingPeers;
std::vector<uint32_t> materializedChannels;
std::atomic<bool> materializationFailed;
```

`materializedChannels[peerIndex] == K` means every channel in `[0, K)` is ready
for a kernel. A request for `N <= K` is a local no-op. A larger request appends
only the exact missing range `[K, N)`.

The mutex protects request merging, distributed materialization, and watermark
publication. Backend range allocations are immutable teardown ownership; there
is no separate per-range state machine. The terminal bit is atomic so the IBRC
progress thread can poison the transport without waiting behind an in-flight
bootstrap exchange.

### Stable device descriptors

Communicator initialization reserves lightweight descriptor storage for the
full logical capacity:

```text
outerTransport[peer]
channelDescriptors[peer][channel]
ibgdaQpSlots[peer][NIC][main|companion][channel][direction][lane]
ibrcQueueSlots[peer][NIC][channel][direction][lane]
```

IBGDA keeps separate fixed spans for main and companion QP pointers; IBRC keeps
a fixed command-queue table. All outer and inner addresses remain stable.
Entries begin empty and are populated only when their channel is materialized.
Kernels and older CUDA graphs can therefore keep using an existing prefix while
later launches append channels.

Each channel descriptor contains exact resource bindings and mutable progress
state for that `(peer, channel)`, including:

- local send and receive staging
- remote receive staging
- DATA_READY, SLOT_FREE, and NIC_DONE state
- completion and QP state

Device code selects the descriptor directly. It does not derive channel
addresses by offsetting a peer-wide buffer. Allocation, publication, and lookup
share one canonical `(channel, direction, lane)` to backend-slot mapping.

### Physical resources

| Lifetime | Resources |
| --- | --- |
| Communicator initialization | NIC/PD state, fixed-capacity tables, and backend-global control resources |
| First positive peer demand | peer-wide signal/counter state; start the IBRC progress thread |
| Each materialized channel range | staging/control buffers, completion storage, MRs, and backend transport objects |
| Communicator teardown | every materialized range and fixed descriptor table |

An IBGDA range owns main and companion QPs, loopback responders, completion
slots, and registered staging/control memory. An IBRC range owns QPs, CQs,
command FIFOs, completion storage, registered staging/control memory, and
host-mapped NIC_DONE counters. Each published channel descriptor contains the
exact pointer and key for that channel.

## MCCL submission contract

### Existing MCCL boundary

The existing MCCL architecture remains unchanged:

- `McclComm` dispatches to an `ICollective` algorithm.
- `CommContext` carries the communicator state and `MultiPeerTransport`.
- `ICollective::run()` selects the final topology and launch geometry, prepares
  transport readiness, binds device transport pointers, and enqueues the
  kernel.

Dispatch in `McclComm` is too early because final peers and channel width are
known inside `run()`. The readiness call must occur before device transport
pointers are embedded in launch parameters. Therefore `ICollective::run()` is
the host readiness boundary; algorithms may fill transport-independent launch
fields before that call.

After selecting topology and geometry, the algorithm requests its IB resources
through `MultiPeerTransport`:

```cpp
struct PeerChannelDemand {
  int peerRank;
  uint32_t ibChannels;
};

MultiPeerDeviceHandle get_device_handle(
    std::span<const PeerChannelDemand> demands);
```

`get_device_handle()` validates the complete batch, sorts peers, max-merges
duplicate demands, materializes missing ranges, and returns the stable device
handle. Only then does the algorithm obtain and embed device transport pointers
and launch. Kernels and kernel ABIs do not change.

The demand overload is the canonical MCCL readiness path. The compatibility
peer-list overload converts each preferred-route IB peer to a full-capacity
demand and forwards to the same implementation.

The materialization mutex serializes host growth. No lock is held through
kernel enqueue because growth only appends resources and never invalidates an
in-flight kernel's prefix. Operation overlap follows MCCL's existing
same-communicator execution contract.

### Initial algorithm demands

V1 uses each launcher's existing launch geometry, including any existing
message-size-dependent `numBlocks` selection. It does not add an independent
channel planner. Lazy growth follows whole launch-block resource bundles: for
`B` blocks, the collective includes every channel used by blocks `[0, B)`, even
when one block maps to multiple channels. It then requests the exact highest
channel ID used plus one. PRIMS receives this flattened channel prefix; it does
not reinterpret or round the block demand.

| Launcher | IB demand |
| --- | --- |
| CTRAN Direct ReduceScatter | every remote peer: `numBlocks` |
| Ring ReduceScatter | ring peers: `kNumBlocks * kNumRings` |
| Fused Tree AllReduce | tree peers: `numBlocks * virtualStripes * kTreeLanes` |
| Fused Ring AllReduce | participating prev/next: `numBlocks * virtualStripes` |
| Fused Direct AllReduce | participating IBGDA peers: `numBlocks` |
| Hierarchical Ring AllGather | participating prev/next: `numBlocks` |
| Batched SendRecv | each IB peer: maximum `op.blocks` across that peer's operations |
| Legacy blocking SendRecv | full capacity until matched per-edge geometry is implemented |

Both endpoints of an IB edge must derive the same effective prefix. Batched
SendRecv derives `op.blocks` from the matched edge's message size and channel
capacity; send and receive operations for the same peer are max-merged.

## Materialization protocol

The implementation extends the existing PRIMS lazy-peer hierarchy:

```text
ICollective::run()
  -> MultiPeerTransport::get_device_handle(demands)
     -> MultiPeerTransport::materializePeerChannels()
        -> MultiPeerIbTransport<Backend>::connectPeers()
           -> Backend::materializePeerChannelRange(peer, oldK, newK)
```

`kEager` retains the existing two backend phases: exchange/connect QPs, then
exchange/bind buffers. Any positive demand is promoted to full capacity, and no
channel REQUEST/READY records are exchanged.

`kLazyPrefix` adds agreement around those same backend phases:

1. Validate, sort, and max-merge the complete demand batch before mutation.
2. Merge each exact target with pending state under the materialization mutex.
3. Exchange an exact `PeerChannelRequest` containing backend, range, capacity,
   NIC, direction, QP, pipeline, slot, and staging geometry.
4. Allocate QPs and related backend objects for `[oldK, newK)`. Exchange the
   fixed-capacity `PeerQpPayload` with only that range's entries populated and
   consumed, then connect those QPs.
5. Allocate staging/control resources for the range and exchange buffer
   addresses and keys.
6. Populate the stable device tables for the range and complete device-side
   initialization.
7. Exchange READY by echoing the same `PeerChannelRequest`, then advance the
   peer's materialized-channel watermark to `newK`.

Every growth generation uses a distinct peer-pair bootstrap tag range. This is
required by store-backed bootstraps, whose received keys remain present after a
round completes.

Device entries may be populated before READY, but the host does not return a
launchable handle until READY succeeds and the watermark advances. Existing
bindings in `[0, oldK)` are never replaced or rewritten by growth.

IBRC also maintains a backend publication watermark in command-queue-slot
units. After populating its fixed tables, it release-publishes
`publishedCmdQueueCounts[peerIndex]`; the progress thread acquire-loads this
prefix and never scans partially initialized queues. READY and the common
`materializedChannels` watermark remain the later kernel-admission boundary.

## CUDA graphs

Cold first use while the user's launch stream is under capture is supported:

1. PRIMS enters relaxed thread capture mode for host-side materialization.
2. Allocation, bootstrap exchange, and QP connection remain graph-external host
   work.
3. Device initialization and descriptor uploads use short-lived nonblocking
   initialization streams where required. They are synchronized and destroyed
   before readiness is published.
4. The MCCL kernel is then enqueued on the user's captured stream.

Materialization contributes no graph nodes; the algorithm's normal device work
is captured. Replay performs no allocation, connection, bootstrap exchange, or
descriptor upload. Stable append-only tables also allow an older graph to
replay after a later launch grows the prefix.

PRIMS does not add a communicator-owned control stream or couple growth to a
CUDA event.

## Failure behavior

Batch validation before materialization is non-terminal. Once `connectPeers()`
starts a materialization transaction, any exception is terminal, including a
local allocation failure before the first wire exchange:

1. atomically latch `materializationFailed`
2. clear pending work and unlock the materialization mutex when the failing
   path owns it
3. run backend terminal handling and rethrow the original error
4. reject later readiness calls and device-transport accessors

There is no retry or rollback. Partial resources remain owned for best-effort
communicator teardown. IBRC also publishes failed device status and stops its
progress thread. IBGDA has no backend-specific device poison path; the host
poison prevents later submissions but does not itself stop an already-running
IBGDA kernel.

A cancellation-aware bootstrap can wake both endpoints after an asymmetric
materialization failure. A generic blocking bootstrap such as `MpiBootstrap`
cannot cancel a peer already blocked in an exchange, so that peer may remain
blocked until the upper-layer operation or job watchdog expires. The transport
does not guarantee bilateral fail-fast behavior for such bootstrap
implementations.

## Configuration

`MCCL_PRIMS_LAZY_CHANNELS` is a process-wide cvar and defaults to off. MCCL
samples it while creating each communicator and stores the resulting immutable
mode in that communicator's PRIMS configuration. MCCL has no public
per-communicator option today, while direct/internal transport construction can
set the per-communicator configuration explicitly.

Ranks exchange a fixed `ChannelProtocolRecord` containing mode and capacity
during communicator initialization. Backend, NIC, QP, staging, pipeline, and
slot geometry are validated bilaterally before an IB range is published.

## Deferred extensions

- suffix shrink and reclaim
- independent transport-level channel planning
- matched per-edge demand for legacy blocking SendRecv
- NVL VMM/fabric-handle prefix mapping
