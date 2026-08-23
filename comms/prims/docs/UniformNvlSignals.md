# Uniform NVL Signals

## Scope

Uniform NVL signals provide one compile-time device API over the internal signal storage owned by `MultimemNvlTransport`.
The API supports unicast and multicast access without changing the physical signal allocation.
It does not use or modify `P2pNvlTransportDevice`, `p2pSignalCount`, or pair-specific P2P signal buffers.

The signal API has four compile-time dimensions:

- Access selects a unicast mapping or the multicast mapping of the same backing allocation.
- Topology selects per-peer slots or aggregate counters.
- Phase selects ready, acknowledgment, or consumed storage.
- Wait policy selects how participating threads observe a per-peer stripe.

Compile-time selection keeps access-path and topology branches out of measured device code.

## Storage

For `R` NVL ranks, pipeline depth `P`, and `B` channels, the internal allocation contains:

```text
signalsPerPeer = 3
signalsPerLane = 4
signalsPerChannel = 3R + 4P
internalSignalCount = B * signalsPerChannel
```

Each channel starts with three peer stripes:

```text
channelBase = channel * signalsPerChannel

ready(peer)    = channelBase + 0 * R + peer
ack(peer)      = channelBase + 1 * R + peer
consumed(peer) = channelBase + 2 * R + peer
```

Each pipeline lane then owns four aggregate slots:

```text
laneBase = channelBase + 3 * R + lane * 4

readyCounter(lane) = laneBase + 0
readyEpoch(lane)   = laneBase + 1
ackCounter(lane)   = laneBase + 2
ackEpoch(lane)     = laneBase + 3
```

The peer stripes are shared across pipeline lanes and carry monotonic round values.
Every `StageRound` must carry a nonzero value.
Per-peer round values begin at one and advance monotonically, with successive comparisons remaining within half of the `uint64_t` sequence space.
Callers skip the reserved zero value when a per-peer sequence wraps.
Aggregate counters are lane-private so concurrent pipeline rounds cannot mix their arrivals.
Aggregate consumed signaling is unsupported because the layout has no consumed counter or epoch.

## Address Views

`MultimemNvlTransport` owns one combined physical allocation per NVL rank:

```text
[data][alignment padding][user signals][internal signals]
```

The transport exposes the same internal signal offsets through three address views:

```text
internalLocalSignals
    This NVL rank's local unicast mapping.

internalUnicastSignalsByRank[rank]
    This process's unicast mapping of NVL rank `rank`'s backing allocation.

internalMultimemSignals
    The multicast mapping over every rank's backing allocation.
```

The peer pointer table is indexed by NVL-local rank rather than communicator rank.
It includes the local rank so signal code uses one indexing contract for local and remote destinations.
The host transport owns the device pointer table and keeps it valid for the lifetime of every device handle derived from that transport.

The unicast mappings and multicast mapping retain the same physical allocations.
No signal API creates a second signal buffer or aliases the legacy P2P signal allocation.

## Public Device Contract

The API uses these compile-time selectors:

```cpp
enum class NvlSignalAccess {
  Unicast,
  Multimem,
};

enum class NvlSignalTopology {
  PerPeer,
  Aggregate,
};

enum class NvlSignalPhase {
  Ready,
  Ack,
  Consumed,
};

enum class NvlPerPeerWaitPolicy {
  WaitAll,
  SerialMin,
  TreeMin,
  ButterflyMin,
};
```

`signal_publish`, `signal_wait`, and `signal_publish_and_wait` take the transport device handle, a channel and round description, participant roles, a thread group, and a timeout where waiting is required.
`NvlSignalParticipants` identifies the active publishers, active waiters, and expected arrival count without embedding benchmark-specific control flow in the primitive.

Aggregate multimem operations require `signal_publish_and_wait` on every rank because every multicast destination must reserve the same epoch increment.
The split `signal_publish` and `signal_wait` entry points reject aggregate multimem at compile time.
They remain available for per-peer protocols and aggregate unicast protocols whose counter updates target only selected waiters.

The following combinations are rejected at compile time:

- Aggregate topology with a non-default per-peer wait policy.
- Aggregate topology with consumed phase.
- Split aggregate multimem publication or waiting.
- A phase or access operation that has no storage or instruction mapping.

Host launch sizing uses 64 threads through 64 ranks and 128 threads through 72 ranks.
Device protocol validation rejects per-peer teams above 72 ranks and traps on an out-of-range channel, lane, rank, or signal offset before issuing a signal operation.

## Execution Ownership

One cooperative `ThreadGroup` owns one logical channel.
Signal kernels use one-dimensional grids and blocks, and both topologies
require `group.group_id == channel`. The logical group id is authoritative even
when a caller has deliberately renumbered groups; `block_id` continues to name
the physical CUDA block rather than the channel.
Concurrent operations may use separate streams only when they own disjoint
channels; concurrent launches targeting the same channel are invalid.

Aggregate topology uses one full owning warp per channel, so multiple warp
groups in one CUDA block may own distinct channels.
Pipeline depth determines the active lane owners:

```text
thread 0     owns pipeline lane 0
thread 1     owns pipeline lane 1
...
thread P - 1 owns pipeline lane P - 1
threads P..31 do not publish or wait
```

Per-peer topology uses one whole block per channel: two warps through 64
ranks and four warps for 65-72 ranks:

```text
thread 0     owns NVL peer 0
thread 1     owns NVL peer 1
...
thread R - 1 owns NVL peer R - 1
threads R..127 own no peer slot
```

Per-peer launches use 64 threads through 64 ranks and 128 threads for 65-72 ranks.
The block contains exactly that many threads.
Every launched thread participates in required block synchronization.
`TreeMin` and `ButterflyMin` additionally use all threads for their reductions.

Per-peer slots are shared across pipeline lanes, so per-peer operations use pipeline depth one.
Their round value is monotonic across reuse of the same stripe.

## Per-Peer Topology

For unicast publication, a publisher writes its sender-owned slot through the destination rank's entry in `internalUnicastSignalsByRank`.
Fan-in publication targets only the coordinator.
Global publication targets every destination rank.

For multicast publication, one publisher writes its sender-owned slot through `internalMultimemSignals`.
The multicast store replicates that slot value into every rank's local backing.

Waiters observe their local peer stripe through `internalLocalSignals`.
The wait policies have identical completion and timeout semantics:

1. `WaitAll` assigns one required peer slot to each participating peer-owner thread.
2. `SerialMin` has one thread scan every required peer slot.
3. `TreeMin` distributes slot loads and reduces completion through a block-wide tree.
4. `ButterflyMin` distributes slot loads and reduces completion through a block-wide butterfly exchange.

The completion predicate is always that every required peer slot has reached or advanced beyond the selected round within half of the `uint64_t` sequence space.
This modular comparison remains correct across wraparound and prevents a faster publisher from stranding a waiter on an earlier exact value.
Wait-policy selection changes observation geometry, not protocol semantics.

## Aggregate Topology

Aggregate multicast publication issues one `multimem.red.release.sys.global.add.u64` for each active lane.
Every publisher targets the counter owned by its channel and lane.
Global warp group `C` owns aggregate channel `C`.
The multicast instruction advances the corresponding counter on every rank, regardless of which ranks wait for that operation.
Global completion advances each rank's counter by `R`.
Fan-in completion advances each rank's counter by `R - 1`.

Aggregate unicast publication atomically adds through the destination's unicast mapping.
Fan-in publishers target the coordinator's lane counter.
Global publishers update every destination's lane counter, including their local destination.

Each active lane owner waits on the corresponding local counter with acquire semantics.
Aggregate counters carry anonymous cumulative credits. They do not identify
which publisher produced a credit or encode `StageRound::value`.

The caller defines the credit contract for each acknowledged producer-consumer
epoch. For epoch `k`, let `quota[k][publisher]` be the maximum number of Ready
credits that publisher may contribute before Ack(k), and let
`readyDelta[k]` be the sum of those quotas. A valid contract requires:

```text
budget:
    every Ready publisher has a fixed credit quota for epoch k

bound:
    no publisher exceeds its quota before Ack(k)

completion:
    the consumer waits for previousReady + readyDelta[k]
    the consumer performs the protected payload work

acknowledgment:
    publish Ack(k) only after Ready(k) and payload consumption complete

turnstile:
    every rank that may contribute to epoch k + 1
    wait for Ack(k) before publishing the next Ready credit
```

Because each publisher cannot exceed its quota, reaching the sum of all quotas
proves that every publisher met its assigned contribution. A total-credit
threshold without per-publisher bounds is insufficient: one fast publisher
could substitute extra credits for a delayed publisher. The Ack waiter mask
must include every rank that may contribute to the next epoch, including ranks
introduced by a participant-mask transition.

For example, an `N`-producer-to-one-consumer protocol may assign two Ready
credits to every producer before each Ack:

```text
initial state: Ready = 0, Ack = 0

epoch 1:
    each of N producers contributes at most 2 Ready credits
    consumer waits for Ready >= 2 * N
    consumer performs the protected work
    consumer publishes Ack(1); every next-epoch producer waits for it

epoch 2:
    each of N producers contributes at most 2 more Ready credits
    consumer waits for Ready >= 4 * N
    consumer performs the protected work
    consumer publishes Ack(2)
```

The quota may be one, two, or another caller-defined value, and may vary by
publisher or epoch. Payload protected by the full epoch must not be consumed at
an intermediate anonymous Ready threshold that does not establish the complete
credit budget.

Ready-only aggregate fan-in is unsupported when a publisher is not also a
waiter and no external turnstile bounds its credits. Such a publisher could
enter a later epoch and contribute another anonymous Ready credit before a
delayed current-epoch publisher contributes.
Ready-only aggregate barriers remain valid when every publisher is also a
waiter for the same operation.
For multicast access, every rank advances its local epoch by the expected arrival count for every operation.
A selected waiter advances the epoch after observing the counter, while a nonwaiter reserves the same arrival count without blocking.
This keeps each rank's counter and epoch accounting aligned when the waiter mask changes on a reused channel and lane.
For unicast access, only selected waiters receive counter updates and advance their epochs.
If the counter already contains later arrivals, the waiter completes the current epoch and leaves the additional arrivals available to later epochs.
For `B` channels and `P` active lanes, the protocol uses `B * P` independent counters.
It never redirects all channel arrivals into one counter.

## Ordering

Unicast publication uses the existing system-scope release store or atomic add in `SignalState`.
Waiting polls `SignalState::load` with system-scope acquire semantics and the modular sequence predicate.
A caller that enables a timeout starts the `Timeout` before entering the primitive.

Multicast per-peer publication uses `multimem.st.release.sys.global.u64`.
Multicast aggregate publication uses `multimem.red.release.sys.global.add.u64`.
Waiters read their local backing with system-scope acquire semantics.

The primitive performs the topology-required warp or block synchronization before publication and before receipt.
The caller remains responsible for ensuring payload writes precede ready publication and payload reads complete before consumed publication.

## Fixed RTT Return

Signal benchmarks use one return protocol for every forward access path and topology:

```text
access = Multimem
topology = Aggregate
phase = Ack
publisher rank count = 1
publisher rank = designated coordinator
operations per publisher rank = active pipeline lanes
expected increment per active lane = 1
```

The coordinator issues one multicast aggregate acknowledgment per active channel and lane.
Every producer waits on its local acknowledgment counter.
The return path has no per-peer wait-policy dimension and does not reproduce the forward topology.

## Lifetime And Failure

The transport exchanges all unicast mappings before constructing its multicast overlay and device pointer table.
Every rank executes the same collective setup order.
Device handles are unavailable until every setup phase succeeds.

If any exchange phase fails, the transport instance is poisoned for subsequent `exchange()` calls.
Callers must destroy it and construct a new transport before retrying.
Destruction releases the device pointer table, multicast mapping, peer unicast mappings, local unicast mapping, and retained physical allocations in dependency order.

## Non-Goals

This API does not change collective algorithms, payload staging, user signal storage, or automatic memory registration.
It does not add operation tracing.
It does not add a P2P transport adapter.
It does not define performance acceptance thresholds.
