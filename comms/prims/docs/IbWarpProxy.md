# IB Warp Proxy

`IbgdaWarpProxy` (`transport/ibgda/IbgdaWarpProxy.cuh`) is a block-resident,
topology-independent stream engine for staged IBGDA send and receive progress.
The current collective integration is fused Ring AllReduce. Ring chooses peers,
logical channels, buffers, and operation order.

IBRC uses a separate host-side progress architecture. The warp proxy is
NVIDIA-only and is compiled out on AMD/ROCm.

## Architecture decision

The final design has one 640-thread cooperative Ring worker group and one
32-thread IB proxy warp, for 672 threads total. There is no dedicated CQ warp.

```text
worker warps                         IB proxy warp
------------                        -------------
stage/copy/reduce                    poll DATA_READY
publish/consume stream counters  -> post SLOT_FREE
wait for arrived/retired          <- post RDMA + DATA_READY
                                     poll send CQ and publish retired
```

The workers never poll a CQ. Before reusing a physical send slot, they acquire
the stream's `retired` frontier. The IB proxy warp advances that frontier after
observing the corresponding completion.

The redesign changes how work is described, not the peer-visible wire
protocol:

| | Previous command-queue proxy | Final stream proxy |
|---|---|---|
| Work publication | Per-operation descriptors in depth-sized send and receive queues | Monotonic counters in one duplex stream per `(transport, logical channel)` |
| Slot identity | Carried in each command | Derived from `ordinal % pipelineDepth` and `ordinal / pipelineDepth` |
| Forwarding | Receive-to-send dependency represented inside proxy commands | Collective composes an upstream Rx stream and downstream Tx stream |
| Progress ownership | One IB proxy warp interprets collective-shaped commands | One topology-independent IB proxy warp scans ready streams |
| CQ ownership | Workers wait for completion while preparing a reused slot | The posting IB proxy warp also polls send CQs and publishes `retired` |

Each Tx stream advances `produced -> posted -> retired`; each Rx stream advances
`requested -> arrived -> consumed -> credited`. These absolute frontiers are
the producer/consumer contract. `DATA_READY` and `SLOT_FREE` remain cumulative
peer-visible byte counters; the shared-memory counters do not replace them.

## Execution model

A proxy launch contains one cooperative worker group followed by one IB proxy
warp:

```text
threads [0, WorkerThreads)                    worker group
threads [WorkerThreads, WorkerThreads+32)     IB proxy warp
```

The worker group executes the collective and performs local staging copies. The
IB proxy warp owns peer-visible progress:

- polling remote `DATA_READY` counters;
- posting `SLOT_FREE` credits;
- posting RDMA writes and their fused `DATA_READY` updates; and
- polling send CQs to retire reusable staging slots.

While a proxy run owns a stream, its IB proxy warp is the only SQ producer and
CQ consumer. The worker and proxy sides communicate through block-scope
acquire/release counters in shared memory.

`WorkerThreads` must be a non-zero multiple of the warp size, and
`WorkerThreads + 32` must not exceed the CUDA threads-per-block limit. Workers
use named barrier 1; barrier 0 remains available for the final full-block join.

## Streams

A stream is bound lazily by this key:

```text
(P2pIbgdaTransportDevice*, logical channel)
```

The proxy does not encode Ring neighbors or a forward operation. Each key owns
a `DuplexStream` with independent transmit and receive state:

```text
Tx: produced -> posted -> retired
Rx: requested -> arrived -> consumed -> credited
```

- `produced`: the worker has finished staging the next send slot.
- `posted`: the IB proxy warp has posted that send and recorded its completion
  ticket.
- `retired`: the IB proxy warp has observed enough CQ progress to reuse the
  physical send-staging slot.
- `requested`: the exclusive receive frontier requested by the worker. Ring
  publishes its complete receive plan only when the schedule exceeds one
  pipeline depth; shallower schedules advance it one slot at a time.
- `arrived`: the IB proxy warp has observed its `DATA_READY` value.
- `consumed`: the worker has finished reading or reducing the receive slot.
- `credited`: the IB proxy warp has posted the corresponding `SLOT_FREE`
  update.

All are monotonically increasing absolute slot ordinals. The physical slot and
generation are derived rather than published in a command:

```text
slot       = ordinal % pipelineDepth
generation = ordinal / pipelineDepth
```

The first operation binds its absolute base ordinal from the transport's
persistent progress cursor. This lets a new proxy invocation safely inherit
completion state left by earlier proxy, blocking, or registered operations on
the same transport channel.

There is no command ring, per-depth publication-descriptor queue, runtime queue
depth, or queue-full telemetry. Each Tx stream retains one published byte count
per physical slot. Backpressure comes from the real resources:

- a producer cannot overwrite a send slot until `retired` covers its prior use;
- a receiver consumes only after `arrived` covers its request; and
- the IB proxy warp posts no send until the peer's cumulative `SLOT_FREE` counter
  covers that slot.

## Fixed-slot control and exact TX length

The stream representation is possible because every publication advances one
complete logical protocol slot. The proxy requires:

- `protocol::Simple`;
- a fixed-size `CopyOp`;
- `maxSignalBytes == 0`; and
- `pipelineDepth` in `[1, kIbgdaWarpProxyMaxPipelineDepth]`.

For every send, the producer stores the valid protocol length in
`publishedBytes[slot]` before release-publishing `produced`. The IB proxy warp
uses that value as the RDMA data length. For Simple it is the actual protocol
length rounded to the protocol's 16-byte unit, and may be smaller than the
physical slot for a final partial chunk. Receivers still copy only their locally
known valid payload.

The control protocol does not become variable-sized: `DATA_READY`, `SLOT_FREE`,
staging offsets, ordinals, and cursor advancement all remain full-slot. A short
put and its full-slot `DATA_READY` update stay ordered on the same QP. This keeps
exact-length Ring interoperable with the blocking path and lets the proxy
reconstruct every location from the stream key and ordinal; only the data WQE
needs per-slot length metadata.

## Send lifecycle and lazy CQ retirement

For each send slot, workers:

1. bind the stream and wait until the slot's previous generation is retired;
2. copy valid bytes into transport-owned send staging;
3. store `publishedBytes[slot]` and publish the next `produced` ordinal with
   release ordering.

The IB proxy warp acquires `produced`, waits for the peer's `SLOT_FREE` frontier,
posts one put for that logical slot with fused full-slot `DATA_READY`, records
the completion ticket, and advances `posted`. The RDMA data length is always
`publishedBytes[slot]`.

CQ polling uses a per-stream high watermark `H`: one at pipeline depth one,
otherwise `pipelineDepth - 1`. After trying to post one send, the same progress
warp compares `posted` and `retired` and attempts one retirement when their
distance reaches `H`. Successful polling advances `retired`; the worker acquires
that frontier before overwriting staging. Starting one slot before the pipeline
fills gives retirement time to complete without polling short streams or
waiting for the producer to stall. At depth one, the worker cannot stage the
next generation until the IB proxy warp advances `retired`, so both the
completion record and the single `publishedBytes` entry remain protected.

If the producer reaches a physical-slot wrap before the corresponding
completion is retired, it waits on `retired` until the IB proxy warp observes that
completion. Slot reuse therefore remains completion-gated even though CQ
polling is not part of every publication's critical path.

The transport-owned completion records remain durable across kernel launches.
Normal proxy exit does not retire a tail below the high watermark. A later
owner retires any remaining completion before reusing its physical slot.

## Receive lifecycle

Ring publishes the exact final `requested` frontier before entering a receive
schedule that exceeds one pipeline depth, allowing the IB proxy warp to poll
ahead without a per-slot handoff. A shallower Ring schedule, like other
demand-driven callers, advances `requested` immediately before each receive.
The proxy advances `arrived` only after polling the transport's per-lane
`DATA_READY` state. Workers then copy or reduce the valid payload and publish
`consumed`; the proxy responds with one full-slot `SLOT_FREE` credit and advances
`credited`.

After stream binding, the worker leader combines request publication and the
wait for `arrived` before broadcasting the receive token once to the worker
group. This keeps both planned and demand-driven request/wait paths to one
multiwarp rendezvous per receive.

`DATA_READY` and `SLOT_FREE` remain the transport's cumulative wire-byte
counters. The proxy's shared-memory counters describe ownership and progress;
they do not replace the peer-visible protocol.

## Forwarding

Forwarding has no proxy-specific state. The collective composes two ordinary
streams:

```text
upstream Rx: requested -> arrived -> consumed -> credited
downstream Tx: produced -> posted -> retired
```

The worker waits for upstream arrival, prepares the downstream physical slot,
performs the fused receive/reduce/copy, then publishes upstream consumption and
downstream production. The IB proxy warp scans all streams without blocking on an
unready peer, so it can continue issuing credits and making progress elsewhere.

This is the boundary that keeps topology out of the proxy: a Ring forward is
only an Rx event followed by a Tx event.

## Drain and abort

Normal `run()` completion waits for the proxy-visible protocol frontiers:

```text
posted == produced
credited == requested
```

It does not require `retired == produced`; the unreused send tail may stay
outstanding as described above. After the worker function returns,
`producerDone` tells the IB proxy warp that no new streams or publications can
appear. The IB proxy warp exits only after the same normal drain condition, and the whole
block joins once at the end.

Ring does not add a Phase-2 drain. Its outgoing payloads have already been
copied into transport-owned staging, so the IB proxy warp may finish posting
them while workers execute Phase 3. The final `run()` join still provides the
kernel-exit guarantee.

On abort, termination replaces drain as the contract. The worker and proxy
waits observe the shared abort state and unwind. Locally published but not yet
peer-visible work may remain with `posted < produced` or
`credited < requested`; the IB proxy warp deliberately abandons it. Recovery requires
the normal host-side transport reconfiguration because peer counters and
completion state are no longer reusable.

If abort ends a stalled `wait_recv()`, it returns an invalid sequence sentinel.
The caller checks `recv_wait_succeeded()` before reading receive staging or
forwarding its contents. `publish_recv()` also ignores the sentinel, so an
aborted receive cannot advance `consumed` or issue a `SLOT_FREE` credit for data
that never arrived.

## Capacity and concurrency contract

`MaxStreams` is a compile-time shared-memory capacity (8 by default), not a
runtime queue depth. Ring specializes it to 2 for its previous-peer and
next-peer streams. Each proxy instance owns at most that many distinct
`(transport, channel)` keys. Exceeding the capacity traps with a diagnostic.

Stream binding and publication assume one cooperative producer group. Different
logical streams may be interleaved by that group, but independently executing
subgroups must not bind or publish concurrently through the same `Ops` object.
Supporting that model would require atomic stream allocation and per-subgroup
publication ownership; merely increasing `MaxStreams` is insufficient.

There is no cross-block coordination. Each CUDA block has its own `SharedState`
and IB proxy warp, while the logical channel selects that block's transport
channel and QPs.

Distinct stream keys must not alias the same physical send QP/CQ. Production
transport handles satisfy this because QP ownership is keyed by transport,
logical channel, direction, NIC, and QP lane. This is also why shallow copies of
a device transport that refer to the same QP storage must not be passed as
independent streams: the proxy would see different pointers while the hardware
would still have two owners.

## Ring adapter

Ring uses the existing `IbOps` policy seam: `send`, `recv`, and `forward` call
the proxy through the generic transport helpers. When
`MCCL_IBGDA_WARP_PROXY_ENABLE` is enabled, the host selects the proxy only for
staged, untraced, unidirectional, one-stripe, Simple IBGDA runs with fixed-slot
control and a supported pipeline depth.
Every Ring proxy send uses its published exact length; a partial final slot
sends only its 16-byte-aligned Simple protocol extent while control credits
still advance by one physical slot. Multi-stripe Ring still uses
`BlockingIbOps` because its independent subgroups do not satisfy the proxy's
single-producer contract.

## Transport boundary

The proxy reuses `P2pIbgdaTransportDevice` for lane selection, WQE reservation,
descriptor construction, doorbells, signal operations, and CQ polling. A data
put and its `DATA_READY` update remain ordered on one QP. `SLOT_FREE` remains a
standalone receive credit. QPs are still owned by `(channel, direction)` and may
round-robin across NIC/QP lanes exactly as in the blocking path.

The proxy does not use IBRC, registered-source sends, LL/LL128, variable-size
copy operations, counter-carrying puts, `put_cooperative`, or fine-grained Pipes
trace entry points.

## CQ ownership decision

We compared two otherwise matched versions of the exact-length stream proxy:

- **IB-warp poll:** the IB proxy warp posts sends and polls their CQs;
- **worker-poll:** workers poll the CQ only while acquiring a send slot for
  reuse.

The comparison used four GB300 ranks, one Ring block, one virtual stripe,
pipeline depth 8, 512 KiB physical slots, two active Data Direct NICs per rank,
and three alternating paired rounds. The 1 KiB-16 MiB ranges used 50 warmup and
200 measured iterations; the 32-512 MiB range used 10 warmup and 50 measured
iterations. All 72 rank logs reached the final barrier, all 18 runs reported
zero out-of-bounds values, and every runtime launch summary selected the warp
proxy.

The table reports the geometric mean of paired `worker/IB-warp` latency ratios;
positive values mean worker-poll was slower.

| Message range | Out of place | In place |
|---|---:|---:|
| 1 KiB-1 MiB | +10.66% | +10.54% |
| 2-16 MiB | +5.08% | +5.34% |
| 32-512 MiB | +7.98% | +8.34% |
| All 13 sizes | +8.30% | +8.41% |

IB-warp poll won all 39 out-of-place and all 39 in-place paired observations.
Even the closest point, 4 MiB, favored IB-warp poll by 2.28% out of place and
2.27% in place. Worker-poll puts CQ progress on the producer's slot-acquisition
path and stalls the cooperative worker group at slot reuse. The final design
therefore keeps CQ polling in the existing IB proxy warp and has neither a
dedicated CQ warp nor worker-side CQ polling.

## Full redesign comparison

We also compared the final depth-gated stream proxy with IB-warp-owned CQ
polling against its immediate master parent, which contains the exact-length
mailbox/worker-CQ proxy. Both artifacts used the same B300 build configuration.
The test used four GB300 ranks, one Ring block, one virtual stripe, a configured
maximum of 32 channels, pipeline depth 8, 512 KiB physical slots, two active
Data Direct NICs per rank, and three alternating paired rounds. Bidirectional
Ring was explicitly disabled so both binaries selected their warp-proxy
kernels. Paired deltas are geometric means of same-round
`stream/mailbox - 1` latency ratios; negative values mean the stream proxy was
faster.

| Message range | Out of place | In place |
|---|---:|---:|
| 1 KiB-1 MiB | -2.13% | -2.17% |
| 32-512 MiB | -12.22% | -13.17% |

At 32 MiB, median out-of-place latency fell from 3332.0 to 2933.2 us and
in-place latency fell from 3176.7 to 2768.6 us. The stream proxy was faster at
every measured size from 1 KiB through 1 MiB and at 32, 128, and 512 MiB.

Ring counts the exact receive publications for the block. When that count fits
within one pipeline depth, it advances `requested` on demand; when the count
exceeds the depth, it publishes the exact end frontier so the proxy can poll
ahead. This is a logical-publication threshold rather than a message-size
threshold and therefore follows changes in node count, shard geometry, and
pipeline depth. The worker leader also combines request publication and arrival
waiting into one broadcast for both modes. The performance crossover was
measured for the four-node, depth-eight geometry above; other geometries require
their own performance validation.

Component campaigns explain the gate. An all-demand proxy improved 1 KiB-1 MiB
latency by 7.20% out of place and 7.04% in place relative to always planning,
but regressed 32-512 MiB by 8.15% and 9.31%. A 2/4/8/16 MiB sweep placed the
crossover at one pipeline depth: demand won at 2 MiB, was no longer a clear win
at 4 MiB, and lost by 5-10% at 8 and 16 MiB. Depth-gated planning retains the
shallow benefit and the large-message lookahead.

All 48 canonical rank logs reached the final barrier, all 12 runs reported zero
out-of-bounds values, all 108 result error fields were zero, all 264 runtime
launch summaries selected the Ring warp proxy and reported `maxChannels=32`,
and every mapped RNIC counter showed positive traffic. One partial pair with a
bootstrap socket error during mailbox teardown was rejected and rerun
successfully.

### Small-message blocking fallback

We separately tested whether small messages should bypass the proxy. The same
rebased binary ran three alternating paired rounds with only
`MCCL_IBGDA_WARP_PROXY_ENABLE` changed. Bidirectional Ring remained disabled,
and runtime summaries confirmed the blocking and proxy paths independently.

| Message range | Out of place | In place |
|---|---:|---:|
| 1 KiB-1 MiB | -5.94% | -5.94% |

The deltas are geometric means of same-round `proxy/blocking - 1` latency
ratios, so the proxy was faster. At 1 KiB through 256 KiB it reduced latency by
6.3-7.0%; at 1 MiB it reduced latency by 1.5% out of place and 3.4% in place. A
blocking size fallback would therefore make the observed small-message latency
worse. The depth-gated receive policy above instead keeps the proxy and avoids
planning when all receive publications fit within one pipeline depth.
