# PRIMS Lazy Channels

Status: Design contract for PRIMS lazy IB channel materialization.

## Overview

PRIMS already connects peers lazily. Before this change, the first use of an IB
peer created every configured channel for that peer.

Lazy channels extend the existing `get_device_handle()` readiness boundary:

```text
before:
  get_device_handle(peers)
    -> connect each requested peer
    -> materialize all configured channels
    -> return the device handle

after:
  get_device_handle(PeerChannelDemand{peer, N})
    -> connect the peer if needed
    -> materialize the peer's channel prefix [0, N)
    -> return the same stable device handle
```

The outer device handle and route selection remain stable. Lazy growth changes
internal IB backend descriptor storage, but does not change the
collective-visible transport operations. The new behavior only changes how much
IB transport state must be ready before the handle is returned.

`MCCL_PRIMS_LAZY_CHANNELS` selects the behavior and defaults to off:

```text
lazy channels disabled:
  first positive demand -> materialize [0, capacity)

lazy channels enabled:
  first demand N        -> materialize [0, N)
  later demand N > K    -> materialize [K, N)
  later demand N <= K   -> no-op
```

V1 is exact-prefix, grow-only, and append-only. It does not round channel
demand to powers of two, reclaim a suffix, or support sparse channel sets.

## Units and invariants

| Term | Meaning |
| --- | --- |
| Launch block | A collective CUDA block/CTA and resource-planning unit, not a physical SM. |
| Logical channel | A PRIMS IB resource slot selected by device code. One block may use multiple channels. |
| Capacity `C` | Maximum logical channel prefix supported by the communicator. |
| Demand `N` | Target prefix for one peer, not the number of channels to add. |
| Watermark `K` | Prepared prefix length; every channel in `[0, K)` is launchable. |
| Growth range | The missing suffix `[K, N)` when `N > K`. |
| Protocol slot | One `Simple` or `LL` resource slot inside a channel. The protocols share channel QPs but have separate staging, signal, and counter state. |

`K` is tracked independently for each remote peer. Host readiness state has
`nRanks - 1` entries indexed by compact peer index; the local rank has no
entry. Each peer also has an independent successful-growth generation used to
derive bootstrap tags.

In this document, materialization is the complete host transaction.
Device-state publication is the local backend step that writes descriptors into
fixed device tables. Allocated or published resources are not launchable until
the mode-specific commit updates the peer's watermark.

Some historical names use `group` or `group_id` for the logical channel index
space. In this document, `channel` is the canonical term. These names do not
mean CUDA block.

The required invariants are:

1. Each peer is prepared as one contiguous prefix `[0, K)`.
2. Growth satisfies `0 <= K < N <= C`; `N <= K` is a no-op.
3. Existing allocations, rkeys, and descriptor bindings never move.
4. Both endpoints agree on the range and geometry before launch.
5. Kernels only consume prepared descriptors; allocation and bootstrap are
   host-side work.
6. In lazy-prefix mode, successful READY followed by the local watermark
   update commits the range. In lazy-channel-disabled mode, backend
   materialization completion followed by the watermark update commits full
   capacity.
7. Failure after distributed materialization starts is terminal.

### Example

For capacity `C = 8` and current watermark `K = 2`, suppose two launch blocks
use two channels each:

```text
target N = 2 blocks * 2 channels per block = 4

before: [ready ready empty empty empty empty empty empty]
grow:               [2, 4)
after:  [ready ready ready ready empty empty empty empty]
```

The demand is `N = 4`, not `2`. PRIMS allocates exactly the two missing
channels as one range and leaves `[0, 2)` unchanged.

## API and ownership

The canonical readiness API is:

```cpp
struct PeerChannelDemand {
  int peerRank;
  uint32_t ibChannels;
};

MultiPeerDeviceHandle get_device_handle(
    std::span<const PeerChannelDemand> demands);
```

The call performs both kinds of laziness:

```text
first positive demand with K = 0: connect the peer and materialize [0, N)
N > K:                            materialize only [K, N)
N <= K:                           return the existing stable handle
```

Ownership remains layered:

- The collective selects peers, launch blocks, and required logical channels.
- `MultiPeerTransport` validates the complete demand batch and dispatches it to
  the selected IB transport.
- `MultiPeerIbTransport` applies mode-dependent promotion, max-merges per-peer
  targets, orders peers, runs bootstrap, and owns readiness state.
- IBGDA or IBRC creates and publishes the requested backend range.
- Device code consumes the resulting descriptors.

The host flow is:

```text
ICollective::run()
  -> derive PeerChannelDemand from final launch geometry
  -> get_device_handle(demands)
     -> materialize missing peer/channel ranges
  -> bind the stable handle
  -> launch the kernel
```

Duplicate demands for one peer are max-merged, and the complete local batch is
validated before distributed mutation begins. For every positive-demand IB
edge, both endpoints must enter the same materialization round and request the
same `[K, N)`. Given this edge-symmetric demand graph, peers are processed in a
deterministic deadlock-safe order.

## Demand derivation

Final peers, blocks, stripes, rings, and lanes are known inside the collective,
so `ICollective::run()` is the readiness boundary. For `B` blocks, the
collective requests every logical channel used by blocks `[0, B)`:

```text
N = 1 + highest channel index used by blocks [0, B)
```

PRIMS receives this prefix and does not reinterpret or round it.

| Current caller | Target prefix for each participating IB peer |
| --- | --- |
| CTRAN Direct ReduceScatter | `numBlocks` |
| Ring ReduceScatter | `kNumBlocks * kNumRings` |
| Fused Tree AllReduce | `numBlocks * virtualStripes * kTreeLanes` |
| Fused Ring AllReduce | `numBlocks * virtualStripes` |
| Fused Direct AllReduce | `numBlocks` |
| Hierarchical Ring AllGather | `numBlocks` |
| Batched SendRecv | maximum `op.blocks` among that peer's operations |
| Legacy blocking SendRecv | full capacity until matched per-edge geometry exists |

## Growth protocol

One missing range `[K, N)` is one bilateral host-side transaction between the
two peers:

```text
Rank A                    Bootstrap                    Rank B
  |<-------------------- REQUEST [K,N) ---------------->|
  |<---------------------- QP payload ----------------->|
  |<--------------- BUFFER addresses/rkeys ------------>|
  |            populate local fixed tables              |
  |<---------------------- READY [K,N) ---------------->|
  |                 commit watermark K = N              |
```

1. **REQUEST** exchanges the exact range, capacity, backend, NIC, QP,
   pipeline, protocol-control, and staging geometry. A mismatch fails before
   allocation.
2. **QP** creates the backend objects for `[K, N)`, exchanges one fixed-size
   payload containing all new QPNs, and connects the QPs.
3. **BUFFER** allocates and registers the range backing, then exchanges one
   payload containing its addresses and per-NIC rkeys.
4. **READY** runs after local descriptor initialization. Only then does each
   endpoint publish watermark `N` and return a launchable handle.

Each phase uses `IBootstrap` point-to-point send/receive, not a communicator-wide
all-gather. A typed exchange performs one send and one receive per endpoint;
rank ordering prevents deadlock with blocking bootstrap implementations.

The QP payload includes all new QP table entries. The BUFFER payload covers the
entire range, so adding two channels does not exchange two rkey messages.
Distinct `(peer pair, generation, phase)` tags prevent growth rounds from
colliding.

Lazy-channel-disabled mode keeps the QP and BUFFER phases, promotes a positive
demand to `[0, C)`, and does not run the per-peer REQUEST or READY phases.

## Memory and device state

Communicator setup allocates physically backed, fixed-capacity metadata tables:

```text
peerTransportEntries[peer]
channelDescriptors[peer][channel]
ibgdaQpTable[peer][NIC][main|companion][channel][direction][lane]
ibrcCommandQueues[peer][NIC][channel][direction][lane]
```

These tables are physically backed and begin empty; an IBGDA QP table entry
initially contains a null pointer, not a pre-created QP.

Heavy resources are deferred:

| Time | Resources created |
| --- | --- |
| Communicator setup | NIC/PD state, global control state, and fixed metadata tables |
| First positive peer demand | Peer-wide slot-control signal/counter state; IBRC progress when needed |
| Each growth range | QPs/CQs, protocol staging/control, completion storage, MRs, and backend progress objects |
| Teardown | Release every fixed table, peer-wide allocation, and range |

When send/receive buffers are enabled, one range normally owns one allocation
containing local send and receive staging and one control allocation containing
protocol signal state and the IBGDA device counter. IBRC uses a host-mapped
completion counter.

The allocations are registered with every participating NIC. One BUFFER
payload exchanges addresses and per-NIC rkeys for the range. Later growth gets
new allocations and registrations; earlier addresses, MRs, rkeys, and
descriptor bindings remain valid.

When configured, the first range also creates peer-wide slot-control
signal/counter state. Its buffer payload includes the remote-visible
`slotSignal` and, for IBGDA, `slotDiscard`. Later ranges leave those fields
empty.

### Device-state publication

| Aspect | IBGDA | IBRC |
| --- | --- | --- |
| Progress | GPU/NIC-direct | CPU progress thread |
| Range resources | Main/companion QPs, loopback responders, completion state, and device buffers | Verbs QPs/CQs, command FIFOs, completion state, buffers, and host-mapped counters |
| Device-state publication | Populate the fixed range; publish the peer transport entry after the first prefix | Publish the peer transport entry after the first prefix; release-publish the command-queue prefix after each range |
| Terminal handling | Common host poison | Host/device poison and progress-thread shutdown |

For both backends, device-state initialization and publication finish before
READY.

## Concurrency, graphs, and failure

The materialization mutex covers demand merging, peer ordering, bootstrap, and
watermark updates. It is released before kernel enqueue. Lazy growth inherits
MCCL's existing host-submission contract; it does not add support for concurrent
collective submission. Growth explicitly selects the communicator's configured
CUDA device.

Append-only device-state publication lets an in-flight kernel keep using
`[0, K)` while a later host submission prepares `[K, N)`, subject to MCCL's
existing same-communicator execution contract.

CUDA Graph capture is supported directly. During capture,
`get_device_handle()` materializes the required channel prefix as
graph-external host work before kernel enqueue. Device initialization completes
before READY, so the graph records only normal collective work. Replay performs
no materialization.

Local demand validation failures are non-terminal. Once distributed
materialization starts, any exception is terminal:

1. latch `materializationFailed`;
2. clear pending work and run backend terminal handling;
3. rethrow the original error; and
4. reject later readiness and device-transport access.

The failed suffix is not committed, so the watermark remains `K`, but terminal
poison prevents subsequent host access from falling back to it. Partial
resources remain owned for best-effort teardown. IBRC also publishes device
failure and stops its progress thread; IBGDA host poison cannot stop an already
running kernel.

A cancellation-aware bootstrap can wake both endpoints after asymmetric
failure. A generic blocking bootstrap such as `MpiBootstrap` may leave the peer
blocked until an upper-layer watchdog expires.

## Configuration and compatibility

`MCCL_PRIMS_LAZY_CHANNELS` is process-wide and defaults to off. MCCL snapshots
it into immutable communicator configuration. Ranks validate the same mode and
capacity during setup, and each REQUEST validates its peer-edge geometry.

Existing configuration is first normalized into channel capacity `C` and QPs
per channel. `MCCL_PRIMS_LAZY_CHANNELS` then controls whether a positive demand
materializes the requested prefix `N` or full capacity `C`.

The demand overload is the canonical MCCL path. The compatibility peer-list
overload requests full capacity. A backend direct accessor materializes full
capacity when no prefix exists; with an existing prefix it returns the stable
peer transport entry without expanding it.

The per-edge rendezvous requirement does not require global route-matrix
symmetry or change route selection.

## Validation

Validation must cover:

- exact transitions such as `0 -> 1 -> 3` and `N <= K` no-ops;
- duplicate-peer max-merge, bilateral mismatch, and terminal failure;
- launch blocks that map to multiple channels;
- stable old descriptors across IBGDA and IBRC growth rounds;
- IBRC release/acquire device-state publication;
- CUDA Graph capture-time materialization followed by replay with no
  additional materialization;
- compatibility and lazy-disabled behavior; and
- matched lazy-enabled/lazy-disabled correctness for representative
  collectives, with no data mismatch, out-of-bounds access, or teardown error.

## Deferred extensions

Deferred extensions are suffix reclamation, transport-level channel planning,
matched legacy SendRecv demand, lazy NVL/window storage, and generic
cancellation for blocking bootstraps.
