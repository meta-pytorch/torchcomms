# IB Warp Proxy

`IbgdaWarpProxy` (`transport/ibgda/IbgdaWarpProxy.cuh`) is a block-resident service warp
that owns every peer-visible RDMA action for the IBGDA transport. This document describes
its design, lifecycle, API contract, and the way it maps onto the underlying verbs.

Scope is IBGDA. IBRC is a separate backend with a host-side progress thread and a
GPU-written command ring; see `transport/ibrc/`. Nothing here applies to it.

Read `Channels.md` first for the channel and QP ownership model — this document builds on
it and does not restate it.

## Why it exists

In the blocking IBGDA path, the collective's own worker group posts its RDMA. The group
leader builds the WQE, rings the doorbell, and then spins on `wait_signal` or
`wait_local_completion` while every other thread in the block waits at a barrier. For the
fused Ring AllReduce that is 639 threads parked behind one thread talking to a NIC.

The warp proxy restores the shape of a proxy without a host thread: one trailing service
warp owns the NIC interaction, and the worker warps hand it commands through a
shared-memory ring and carry on computing.

## Placement

The proxy is an *ops policy*, interchangeable with `BlockingIbOps`
(`transport/P2pIbTransportDeviceDecl.cuh`). Both expose `group()`, `sync()`, `drain()`,
`send()`, `recv()`, `forward()`, and a `WireProto` typedef, and both plug into the same
pipelined state machines in `transport/P2pIbTransportDeviceImpl.cuh` through the `IbOps`
template parameter of `send_impl` / `recv_impl` / `forward_impl`. A collective is written
once against that interface; swapping the policy swaps blocking-leader IB for warp-proxied
IB.

```text
  collective kernel
    └── IbgdaWarpProxy<WorkerThreads>::run(state, block, abort, workerFn)
          ├── worker group   → workerFn(ops), calls Ops::send/recv/forward
          └── service warp   → run_service() loop
                └── detail::send_impl / recv_impl / forward_impl<IbOps = Ops>
                      └── P2pIbgdaTransportDevice::put() / signal() / read_signal()
                            └── DOCA GPUNetIO device verbs
```

## Thread geometry

A block running the proxy is `WorkerThreads` worker threads plus exactly one trailing
32-thread service warp:

```text
  kBlockThreads = WorkerThreads + kWarpSize
  threads [0, WorkerThreads)              worker group
  threads [WorkerThreads, kBlockThreads)  service warp
```

Compile-time contract: `WorkerThreads > 0`, `WorkerThreads % kWarpSize == 0`, and
`WorkerThreads + kWarpSize <= 1024`. At runtime `validate_block()` traps unless
`blockDim == (kBlockThreads, 1, 1)` and the supplied `ThreadGroup` is `SyncScope::BLOCK`
with a matching `group_size`.

The worker group synchronises on **named barrier 1** (`kWorkerNamedBarrierId`); barrier 0
stays reserved for full-block synchronisation. That separation is what lets the workers
barrier among themselves while the service warp spins freely.

Instantiations today:

| Caller | `WorkerThreads` | block |
| --- | ---: | ---: |
| MCCL fused Ring AllReduce | 640 | 672 |
| `benchmarks/IbgdaSendRecv.cu` | 512 | 544 |
| `tests/MultipeerIbgdaTransportTest.cu` | 512 | 544 |

## Lifecycle

```text
run(storage, fullBlock, [config], abortDevice, workerFn)
  validate_block          trap on wrong geometry
  validate_config         trap unless queueDepth in [1, kQueueCapacity]
  initialize              leader zeroes counters, records queueDepth + telemetry pointer
  fullBlock.sync()        START barrier: state visible to both sides
  ├── workers: Ops ops(...); workerFn(ops); finish_workers()  → producerDone = 1
  └── service: run_service(...)                               → drains, then exits
  fullBlock.sync()        STOP barrier: both sides joined
```

There is no host-side start or stop. The proxy exists for the duration of one `run()` call
inside one kernel launch.

### Two completion shapes

**Normal completion.** `finish_workers()` publishes `producerDone`. The service warp keeps
iterating until `send.posted == send.tail` and `recv.credited == recv.tail`, then breaks.
Every staged send has been posted and every receive credit issued.

**Abort completion.** `run()` returns with `send.posted < send.tail` and/or
`recv.credited < recv.tail`. Pending commands are deliberately abandoned and their credits
never issued.

That is not an oversight. Publishing a credit or a fused `DATA_READY` for work this rank
gave up on would release a peer that is correctly blocked and stop it ever reaching its own
deadline, so one rank's abort would silently suppress fault detection on the rest. See
principle 4 in `comms/common/fault_tolerance/FAULT_TOLERANCE.md`. Queue drain is therefore
**not** a postcondition of `run()`; termination is. Recovery is a host `reconfigure()`, as
everywhere else in the abort contract.

### Ownership contract

A `run()` exclusively owns every `(transport, channel)` pair passed through `Ops` until it
returns, and transport objects must have shared or global lifetime. `workerFn` must
synchronise only via `Ops::group()` and must issue `Ops` calls collectively from that
single producer group. Block-wide barriers inside `workerFn` and concurrent sub-group
issuers are unsupported.

## Shared state

All proxy state lives in one `__shared__ SharedState`, so the hand-off never touches HBM.

```c++
struct SendQueue {                     struct RecvQueue {
  uint64_t tail;    // worker            uint64_t tail;      // worker
  uint64_t posted;  // service           uint64_t ready;     // service: DATA_READY seen
  SendSlotState slots[MaxDepth];         uint64_t copied;    // worker: payload consumed
  SendCommand commands[kQueueCapacity];  uint64_t credited;  // service: SLOT_FREE sent
};                                       RecvCommand commands[kQueueCapacity];
                                       };
```

Every counter is single-producer, single-consumer and monotonic, accessed through
`cuda::atomic_ref<..., cuda::thread_scope_block>` with acquire/release ordering. Only
`queueFullCount` is device-scope, because the host reads it back.

`SendCommand` carries its own `P2pIbgdaTransportDevice*`. That is what lets one proxy serve
several peers — a ring rank's upstream and downstream neighbour — from one queue.

`send.posted` does double duty. Besides flow control it is the happens-before edge that
makes the HBM `IbSendCompletionSlot` hand-off safe: the service warp writes
`sendCompletionSlots[slot]` inside `record_send_completion()` and *then* releases `posted`;
the worker acquires `posted` in `wait_prior_send_posted()` before
`detail::prepare_send_slot()` reads and clears the same lane mask. Removing that wait would
introduce a cross-warp data race on HBM, not merely a stall.

## API contract

The proxy does not reimplement send, recv, or forward. It supplies five call-outs that the
shared state machines invoke when `IbOps` is non-void:

| `Ops` method | Caller | Meaning |
| --- | --- | --- |
| `prepare_send_slot(...) -> bool` | worker leader | `wait_prior_send_posted` (this slot's last command must be posted) then `detail::prepare_send_slot` (local-completion retirement). Returns **true** when the slot could *not* be retired; the caller must not stage over it. |
| `submit_send(...)` | worker leader | Wait for queue space, then enqueue a `SendCommand`. Declines silently on abort rather than handing the service warp work it would turn into a peer-visible put. |
| `wait_recv(...) -> uint64_t` | workers | Enqueue a `RecvCommand`, then block until `recv.ready > sequence`. Returns `kInvalidSequence` on abort. |
| `publish_recv(...)` | worker leader | Advance `recv.copied`, licensing the service warp to emit `SLOT_FREE`. A `kInvalidSequence` is a register-compare no-op — the abort verdict was already decided group-uniformly in `wait_recv`, so re-asking would cost another barrier. |
| `drain()` | worker group | Block until `posted == tail` and `credited == tail`, or abort. |

Constraints enforced by `static_assert`:

- `protocol::Simple` only. `Ops::WireProto` is fixed, and the `*_impl` functions reject a
  non-Simple protocol whenever `IbOps` is non-void. There is no LL counterpart.
- Fixed-size `CopyOp`s only. Variable-size policies such as `AnsCompress` must use the
  blocking path.
- `Ops::send/recv/forward` additionally validate that
  `transport.channel_layout().pipelineDepth` lies in `[1, MaxPipelineDepth]`.

## Flow: send

```text
WORKER GROUP                             SERVICE WARP (leader lane)
─────────────────────────────            ─────────────────────────────────────
per chunk in send_impl:

1. ops.prepare_send_slot(slot, gen)
     wait until posted >= slots[slot].lastCommand + 1
     for each pending lane in slot.laneMask:
       transport.wait_local_completion()   ← CQ poll
     clear observed lanes, bump generation
     true (unretired) ⇒ leave the loop

2. CopyOp::send(src -> sendStaging)       cooperative, all worker threads

3. leader: __threadfence_system(); group.sync()

4. ops.submit_send(...)
     wait_queue_space<true>
     enqueue_send; tail.store(release)  ─────►  post_send_once():
                                                  cmd = commands[posted % cap]
                                                  gate: recv.credited >= requiredRecvCredit
                                                  gate: read_signal(slotFree) >= slotFreeExpected
                                                  transport.put(solo{channel}, ...,
                                                     remote.dataReady, protocolBytes,
                                                     signalPerLane = true)
                                                  record_send_completion(...)
   returns immediately                    ◄────  posted.store(+1, release)
```

The SLOT_FREE backpressure wait and the WQE build plus doorbell both leave the worker's
critical path. What stays on the worker is the completion-retirement check in step 1 and
the staging copy in step 2.

## Flow: recv

```text
WORKER GROUP                             SERVICE WARP
─────────────────────────────            ─────────────────────────────────────
seq = ops.wait_recv(protocolBytes)
   enqueue_recv; tail++             ─────►  publish_recv_readiness():
                                              while ready < tail:
                                                poll_recv_data_ready(cmd)   ← per-lane
                                                ready.store(+1, release)
   block until recv.ready > seq     ◄─────
CopyOp::recv(recvStaging -> dst)          cooperative
group.sync()
ops.publish_recv(seq)
   copied.store(seq + 1, release)   ─────►  post_recv_credits():
                                              while credited < copied:
                                                transport.signal(solo{channel},
                                                   remote.slotFree, bytes, Recv)
                                                credited.store(+1, release)
```

`DATA_READY` polling and the `SLOT_FREE` credit both move to the service warp. The workers
only run the staging-to-destination copy, which for AllReduce is the reduce itself.

## Flow: fused forward

```text
recvToken = ibOps->wait_recv(prev, recvBytes)                  // upstream ready
if (ibOps->prepare_send_slot(next, fwdSlot, fwdCycle)) break;  // downstream slot retired?
CopyOp::forward(dst, fwdStaging, recvStaging, ...)             // one fused pass
group.sync(); leader __threadfence_system(); group.sync()
ibOps->publish_recv(prev, recvBytes, recvToken)                // -> upstream SLOT_FREE
ibOps->submit_send(next, ..., requiredRecvCredit = recvToken + 1)
```

`requiredRecvCredit = recvToken + 1` is load-bearing. It encodes the ring's
deadlock-avoidance ordering inside the command: `post_send_once` refuses to post the
downstream put until `recv.credited > recvToken`, meaning this chunk's upstream receive
credit has already been posted. In the blocking path that ordering comes from statement
order (signal SLOT_FREE to the predecessor before waiting on the successor). With an
asynchronous poster the ordering has to be carried explicitly, and this field is how.

## Service loop

```c++
loop {
  leader only:
    aborted = FT_ABORT_CHECK(...)              // BEFORE any peer-visible work
    step(post_recv_credits)                    // each step runs only if !aborted,
    step(publish_recv_readiness)               // and re-reads the flag afterwards
    step(post_send_once)
    if (aborted) stop = 1
    else if (producerDone && posted == tail && credited == tail) stop = 1
  stop = service.broadcast(stop)
  if (stop) break
}
```

The abort check is hoisted **above** the three steps and re-read **between** them, and both
placements are needed for different reasons. With the check only at the bottom, the
iteration on which an abort first becomes visible has already emitted one more round of
credits and puts. Hoisting alone does not close it either, because
`publish_recv_readiness()` is itself abortable, so an abort first observed inside it would
still be followed by `post_send_once()` in the same iteration.

Exiting on abort does not strand the workers. Their credit and slot waits are
`FT_ABORT_BREAK`-guarded, so the same abort releases both sides — which is the property
that matters, since the two sides coordinate only through these release/acquire counters
and each is otherwise waiting for the other to move them.

`post_send_once` posts at most one command per iteration, interleaved with credits and
readiness, which bounds head-of-line blocking between the three duties.

## Mapping onto the verbs

The proxy issues nothing itself. It calls the ordinary `P2pIbgdaTransportDevice` methods,
so the WQE lifecycle is shared verbatim with the blocking path.

| Stage | Call | Runs on |
| --- | --- | --- |
| lane select | `select_put_lane_ordinal` (`cursor++ % numLanes`) | service warp |
| reserve | `reserve_wqes` → `doca_gpu_dev_verbs_reserve_wq_slots<EXCLUSIVE>` | service warp |
| prepare | `doca_gpu_dev_verbs_wqe_prepare_write` per data chunk, plus `..._prepare_atomic(ATOMIC_FA)` for the fused `DATA_READY` | service warp |
| mark ready | `mark_wqes_ready_mode` — a bare store under `EXCLUSIVE` | service warp |
| doorbell | `submit_wqes` → `doca_gpu_dev_verbs_submit<EXCLUSIVE, SYNC_SCOPE_GPU, AUTO>` | service warp |
| SLOT_FREE | `signal()` → `signal_fenced()` on the control lane, IB `FENCE` bit, `ATOMIC_FA` | service warp |
| completion ticket | `record_send_completion` → `slot.values[lane]`, `slot.laneMask \|= 1 << lane` | service warp |
| CQ poll / retire | `wait_local_completion` / `is_local_completion_ready` → `doca_gpu_dev_verbs_poll_one_cq_at` | **worker leader** |
| DATA_READY poll | `read_signal` via `poll_recv_data_ready` | service warp |
| SLOT_FREE poll | `read_signal(localSlot.slotFree)` in `post_send_once` | service warp |

The signalling protocol is unchanged from the blocking path: cumulative wire-byte counters
for `DATA_READY` (sender fetch-add, piggybacked on the put, per-lane slot) and `SLOT_FREE`
(receiver fetch-add, one slot per channel), plus a local completion ticket per
`(slot, lane)`.

A fused put is a single reservation of `numChunks + 1` WQEs on one QP: N `RDMA_WRITE`s
followed by the `ATOMIC_FA` carrying `DATA_READY`, marked ready together and submitted with
one doorbell. The signal cannot overtake its data because they share a QP and the atomic is
last.

`put_signal_counter_single_impl` — put plus signal plus counter, which uses the companion
QP and `submit_multi_qps` — is never reached from the proxy, because the proxy always
passes an empty `counterBuf`.

Cost note: `doca_gpu_dev_verbs_submit_db` rings the doorbell twice (an early ring to push
WQEs to the NIC, then the DBR update, then a second ring that covers the recovery path). So
each put and each `SLOT_FREE` credit costs two GPU-scope release fences and two MMIO
doorbell writes.

## Channels and QPs

Three multiplicities, handled at different levels.

**Channels are blocks.** The channel id is `ThreadGroup::group_id`. Ring AllReduce builds it
as `logicalDataGroup(blockGroup, blockIdx.x, numBlocks)`, so channel equals `blockIdx.x`.
`make_worker_group` and `make_service_group` both inherit `fullBlock.group_id`, and
`submit_send` / `wait_recv` stamp `command.channel = workers.group_id`. N channels means N
blocks means N independent proxies, each with its own `SharedState`, service warp, and
queues. A single proxy instance drives exactly one channel; `SendCommand::channel` exists so
the service warp can rebuild the right `ThreadGroup` and `IbRemoteChannel` without carrying
a `ThreadGroup` in the command, not to multiplex channels. There is no cross-block
coordination.

**Peers are multiplexed within a proxy** through `SendCommand::transport`.

**QP lanes round-robin within a channel.** `numLanes = numNics * qpsPerConnection`, with
NIC-first ordinals. Selection is a plain non-atomic `IbQpState::cursor++ % numLanes` held in
the channel's state and advanced once per data put. Under the proxy that increment runs on
the service warp — still exactly one thread per `(channel, direction)`, so the non-atomic
cursor stays sound.

Consequences the proxy inherits from `Channels.md` rather than creates:

- A QP belongs to exactly one `(channel, direction)` and has exactly one poster, which is
  what licenses `DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE`. The proxy
  strengthens this: the poster is a dedicated warp rather than whichever worker leader
  arrived first.
- `put(..., signalPerLane = true)` offsets the remote signal buffer by
  `sendRecvSignalSlotOffset(lane_ordinal)` so each lane fetch-adds its own single-writer
  slot. The receiver mirrors the sender's cursor in `recvDataReadyLaneCursor` and waits on
  the lane that carried that chunk, which removes the cross-lane out-of-order hazard.
- `SLOT_FREE` always uses the control lane (ordinal 0) and never advances the cursor, so the
  two sides' lane mirrors stay in lock-step.
- Completion retirement is per `(slot, lane)`. `IbSendCompletionSlot` holds
  `{generation, laneMask, values[lane]}`, and `prepare_send_slot` clears only lanes whose
  completion it actually observed. The masks are never cleared on abort, because clearing
  them would claim the NIC had finished reading staging when it had not.

## Queue sizing

| Knob | Value |
| --- | --- |
| `kIbgdaWarpProxyQueueCapacity` | 16, compile-time, ring size for both send and recv |
| `Config::queueDepth` | runtime, defaults to the capacity, validated to `[1, capacity]` |
| `kIbgdaWarpProxyMaxPipelineDepth` | 16, capacity of `send.slots[]`, must be at least `channelLayout.pipelineDepth` |
| `Config::queueFullCount` | optional device pointer, incremented per full-queue observation |

Capacity 16 is a shared-memory and occupancy budget, not a flow-control choice. The rings
are `SendCommand[16]` plus `RecvCommand[16]` plus slot state, all in `__shared__`, in a
block that already runs 672 threads at one block per SM. It is `constexpr` so the array is
statically sized and the modulo folds to a mask.

`queueDepth` is the runtime throttle: `wait_queue_space` blocks the worker leader while
`tail - head >= queueDepth`, where `head` is `posted` for sends and `credited` for
receives. Lowering it parks the workers on credit waits instead of letting them run ahead;
the unit tests use `queueDepth = 1` deliberately, both to force observable backpressure and
to make "abort mid-flight" mean something.

`queueDepth` should exceed `channelLayout.pipelineDepth`. The real ceiling on in-flight
sends is the number of staging slots, because `prepare_send_slot` blocks on slot reuse; a
command ring at or below the pipeline depth would make the ring, rather than the staging
window, the binding constraint. Both are 16 today, and `shouldUseIbgdaWarpProxy` refuses
the proxy when `pipelineDepth > kIbgdaWarpProxyMaxPipelineDepth`, since `send.slots[]`
cannot index past it.

## What the warp proxy does not do

**Still on the worker warps.** Local-completion retirement, including the CQ poll
(`Ops::prepare_send_slot` → `detail::prepare_send_slot` → `poll_one_cq_at`); every staging
copy; the `__threadfence_system()` before `submit_send`; geometry, cursor, and
progress-slot bookkeeping; and any NVL phases of the collective.

**Falls back to `BlockingIbOps`.** Multi-stripe IB phases; the registered zero-copy source
path, which has no `IbOps` seam; trace-enabled kernels, which use blocking IB ops by
construction; LL and LL128; variable-size `CopyOp`s; and the resumable `progress_*_once`
and acquire/release receive APIs.

**Not implemented.** AMD/ROCm — the whole implementation is behind
`#if defined(__CUDACC__) && !defined(__HIP_PLATFORM_AMD__)`.

**Never invoked by the proxy.** `flush()`, `fence()`, `wait_counter()`, counter-carrying
puts, and `put_cooperative()`, and therefore the companion-QP path.

**Always on the host.** `MultipeerIbgdaTransport` has no progress thread of any kind. The
host still owns NIC discovery and open, PD and MR registration, the bootstrap exchange, QP
creation and state transitions, the atomic fetch-add sink buffer, `IbChannelLayout`
construction and the device-side channel and completion-slot arrays, lazy peer
materialization, kernel launch, and abort escalation plus `reconfigure()`.

## Enablement

```text
useWarpProxy = virtualStripes == 1
            && hasIbPhase
            && !registeredSourceActive
            && MCCL_IBGDA_WARP_PROXY_ENABLE      // cvar, bool, default false
            && pipelineDepth in [1, kIbgdaWarpProxyMaxPipelineDepth]
            && !traceEnabled
            && !__HIP_PLATFORM_AMD__
```

`threadsPerBlock` grows by `kWarpProxyThreads` when the proxy is selected, and
`selectAllReduceRingKernel` dispatches to `mcclKernelAllReduceRing<Type><UseWarpProxy>`.
Both instantiations are emitted per datatype.

Two build gates protect the codegen, which is why the worker callback is a
`__forceinline__` functor rather than a lambda. `AllReduceBuildContractTest.py` pins the
`RingWarpProxyWorker` shape and the `RingWarpProxy::run(...)` call site;
`CodegenGateCheck.py` asserts that no `IbgdaWarpProxy::Ops` callback survives as a separate
device function, because a high-register non-inlined callback cannot be called from a
launch-bounded kernel.

## Invariants

1. One producer group, one consumer warp, per channel. Monotonic SPSC counters in shared
   memory; no locks and no atomics beyond block scope.
2. The service warp is the only poster on the channel's QPs, which preserves the
   `EXCLUSIVE` sharing mode and the non-atomic lane cursor.
3. Peer-visible actions happen only on the service warp, and only when the abort has been
   checked immediately beforehand. Workers may enqueue; only the service publishes.
4. A staging slot is never written while the NIC may still be reading it:
   `wait_prior_send_posted` first, then completion retirement, and on abort the lane masks
   are left alone rather than cleared.
5. `requiredRecvCredit` carries the ring's SLOT_FREE-before-forward ordering that statement
   order used to provide.
6. Abort means terminate, not drain. `posted < tail` or `credited < tail` on exit is a
   deliberate outcome; recovery is a host `reconfigure()`.

## Future work: CQ polling on the service warp

Local-completion retirement is the one RDMA operation still performed by the worker warps.
Moving it to the service warp is deferred, but the design and the reasoning are recorded
here so they do not have to be rediscovered.

**Why it is worth doing.** Today `record_send_completion` (service warp) and
`detail::prepare_send_slot` (worker leader) both read-modify-write the same HBM
`IbSendCompletionSlot::laneMask` from two different warps, made safe only by the
`send.posted` release/acquire edge described under *Shared state*. Making the service warp
the sole writer deletes that cross-warp read-modify-write outright. Reaping would also
become eager rather than demand-driven, so the worker's check would almost always find the
slot already retired, and a 640-thread barrier plus an HBM read would leave the worker's
critical path.

**How to poll cheaply.** `doca_gpu_dev_verbs_poll_one_cq_at` returns success immediately
when `cons_index < cq->cqe_ci`, and send-queue CQEs complete in order. One poll per lane at
the highest outstanding ticket therefore retires every older ticket on that lane, which
collapses today's up-to-`pipelineDepth * numLanes` polls to at most `numLanes`. An error on
an earlier WQE puts the QP into error state and flushes the later ones with error CQEs, so
polling at the newest ticket still detects the fault.

**How to publish the result.** In shared memory, as a per-slot retired generation or a
monotonic retirement counter, via `cuda::atomic_ref<..., cuda::thread_scope_block>` —
consistent with every other handshake in `SharedState`. The HBM `IbSendCompletionSlot`
stays the durable record, because it persists across kernel launches and is shared with the
blocking, progress, and registered paths.

**Follow-on unlock.** `is_local_completion_ready` currently polls with
`RESOURCE_SHARING_MODE_GPU`, paying a device-scope `atomic_max` on `cq->cqe_ci` for every
successful poll. Once the service warp is the sole poller of that CQ, `EXCLUSIVE` becomes
available and that atomic degenerates to a plain store.

**Risks to weigh when picking it up.**

- Polling unconditionally every iteration adds fixed cost to the serialization point. A
  per-lane count of outstanding tickets is needed so the poll can be skipped when there is
  nothing in flight.
- The `NETWORK_ERROR` abort latch moves to the service warp, so the worker's retirement
  wait must be `FT_ABORT_BREAK`-guarded and must broadcast a group-uniform verdict, exactly
  as `wait_prior_send_posted` does today.
- Reaping must be ordered before posting in the service loop. Otherwise a service warp
  spinning inside a full-send-queue `reserve_wq_slots` — which itself blocks on a CQ poll —
  can starve a worker waiting on an unrelated lane's retirement.
