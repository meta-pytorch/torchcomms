# Fault Tolerance Abort Contract

This document describes the shared abort contract used by MCCL, CTRAN, and
Prims device code.

## Principles

Read these before adding or changing a wait loop, a device timeout, or a
collective's abort wiring. The sections below are the detail; these are the
rules the detail exists to serve.

1. **Liveness, not error propagation.** A wait terminates itself once abort is
   visible. Returning a status is optional and must never be the thing that
   prevents a hang. A collective on an aborted communicator completes quickly;
   it does not promise a meaningful result.
2. **Prefer `break` over an early `return`.** Falling through keeps every
   thread on the path to the same barriers. A subset returning early strands
   its peers at `group.sync()`, which is undefined behavior. If you must return
   early, make the decision group-uniform first.
3. **The abort exit belongs in the macro, not in the signature.** Use
   `FT_ABORT_BREAK` / `FT_ABORT_RETURN` (`AbortMacros.cuh`); they terminate the
   loop themselves. Do not convert a `void` wait to `bool` just to report that
   it aborted.
4. **Never signal a peer for work you abandoned.** `FT_ABORT_BREAK` ends the
   *spin* it is written in; it says nothing to the pipeline loop around it. A
   loop that keeps going after its wait gave up will still issue the put, the
   fused `DATA_READY`, and the `SLOT_FREE` credit for every remaining chunk. So
   put a group-uniform `groupAborted()` break between the wait and the first
   peer-visible side effect.

   This is not about garbage output — that is contract-legal. It is that a false
   signal **releases a peer that is correctly blocked and stops it ever reaching
   its own deadline**, so the fault looks to that peer like a successful
   collective. One rank's abort then silently suppresses fault detection on the
   rest, which is the opposite of what this design is for. Measured at
   `IB_ONLY_1x8` before this was fixed: only 5 of 7 survivors recorded
   `TIMED_OUT` from their own deadline; the other two were spuriously completed
   by the ranks that had. After: 7 of 7.
5. **Group uniformity idiom: leader polls, then broadcasts.** Use the existing
   leader + `broadcast<uint32_t>` shape rather than adding a new reduction
   primitive for one call site.
6. **Never derive a loop bound or an index from peer-written data.** Bounds come
   from local geometry or parameters. This is the assumption that lets a wait
   give up without unwinding the caller.
7. **Do not store started deadline state in a transport object.** Transport
   handles outlive operations and are shared across blocks; copy the handle per
   block and call `startTimeout()` on the copy.
8. **Keep abort polling off hot paths.** Gate the check behind a spin count so
   it only engages once actually stalled, and prefer one poller per group over
   one per thread when the loop is per-thread.
9. **Onboard every new or in-flight wait as it lands**, including
   work-in-progress paths. A spin loop that reaches main without an abort check
   is a hang waiting to happen.
10. **Per-operation timeouts come only from the collective API.** The
    communicator deadline stays late-bound in shared state so `setTimeout()` is
    observed by already-created device handles.
11. **Validate a timeout's sign, not its size.** Choosing a sane magnitude is
    the caller's job; `std::chrono` types already make unit mistakes unlikely.
    A negative value must be rejected, because at the `AbortDevice` layer it
    silently means "no override".
12. **Fault tolerance is an MCCL-communicator feature.** Communicators created
    through the NCCLX/CTRAN factory get a disabled `Abort`; see *Scope*.
13. **Terminate the kernel cleanly; never trap to do it.** A trap takes the CUDA
    context down and loses every other stream on the device, which is a worse
    outcome than the hang it replaces. Abort must unwind to a normal kernel
    exit. `FT_DEVICE_TRAP()` stays reserved for the death tests that assert trap
    semantics deliberately.
14. **Contain the abort path in the transport.** The transport leaves its
    per-channel state releasable on abort -- an unwinding operation drives its
    progress slot to the terminal stage (`abandon_progress_state()`), so a
    kernel already queued on the same channel re-initializes without tripping
    `assert_progress_slot_idle()`, and any further progress call short-circuits
    to `Done`. A collective therefore drains and exits on its existing loop
    conditions. This is a liveness guarantee, not a promise that the channel
    still carries meaningful data: after an abort the flow-control counters are
    skewed against the peer's, and recovery is still `reconfigure()`. Do not add
    group-uniform abort gates, entry guards, or slot bookkeeping to a
    collective: if a collective needs one, the transport is leaving state behind
    and that is where the fix belongs.

## Host `Abort`

`Abort` is a communicator-scoped controller. MCCL creates the host object and
passes the same `std::shared_ptr<Abort>` to CTRAN, so host-side MCCL and CTRAN
observe and update the same state.

An enabled `Abort` owns one `AbortState`. CUDA-capable builds allocate that
state as mapped pinned host memory so CPU and CUDA device code can access the
same abort reason. Disabled abort objects do not allocate state and all query
and mutation APIs behave as no-ops or non-aborted results.

The abort reason is first-writer-wins:

- `AbortReason::ABORTED` records an explicit user or transport abort.
- `AbortReason::TIMED_OUT` records an expired timeout.
- `AbortReason::BOOTSTRAP_POLL` records a failure detected by Bootstrap socket-health polling.
- `AbortReason::NETWORK_ERROR` records a transport or peer-network failure.
- `AbortReason::INTERNAL_ERROR` records an internal communication failure.
- `AbortReason::NONE` means no terminal reason has been recorded.

Host and device writers only transition from `NONE` to one terminal reason.
Later writers cannot overwrite the recorded reason.

`AbortInfo` pairs that winning reason with optional host diagnostic context.
`AbortInfo::reasonString()` computes the stable lowercase reason label directly
from the enum, so callers always have displayable text without duplicating it in
the stored object. `Abort::getAbortInfo()` may materialize an expired host
timeout before returning. A device-originated abort, or a host read that races
the winning host writer before context publication, returns the winning reason
with an empty context.

## Device `AbortDevice`

`AbortDevice` is a small, non-owning CUDA device view of the host-owned
`AbortState`. It is safe to pass by value to kernels and device helper methods.
It must not outlive the owning host `Abort` object.

Create device handles through `Abort::getDeviceHandle()`. The returned handle
captures the mapped device pointer and device clock conversion for the current
CUDA device. Kernels that consume the handle must run on that same device, or
the caller must create a new handle after switching devices.

Disabled handles are valid kernel arguments. They have a null state pointer and
all APIs behave as no-op or non-aborted.

`AbortBehavior` selects what device wait code should do after observing an
abort:

- `AbortBehavior::SKIP` is the default. Device waits should return a failure
  status so the caller can unwind without consuming incomplete transport data.
- `AbortBehavior::TRAP` preserves legacy Prims behavior. Common fault-tolerance
  code returns `AbortCheckResult::TRAP`; Prims helpers perform the actual
  `__trap()` so this package stays transport-agnostic.

Callers that only need the old boolean predicate may continue using
`AbortDevice::isAborted()` or `checkExpired()`. New Prims wait loops should use
`AbortDevice::check()` and handle `SKIP` explicitly.

## Device Timeout Semantics

`AbortDevice::startTimeout()` starts a timeout on the copied device handle. The
deadline is local to that copy; it is not stored in shared `AbortState`.

For Prims collectives and transports, timeout-bearing abort handles should be
treated as per-operation and per-block state:

1. The kernel receives an unstarted `AbortDevice`.
2. Each block copies the handle locally.
3. The block starts the local deadline once near kernel entry.
4. Wait loops pass the local handle into transport or synchronization waits.

If any block's local deadline expires, `AbortDevice::isAborted()` records
`AbortReason::TIMED_OUT` in shared state. Other blocks and host code then see
the same terminal abort reason on their next poll.

Do not store started timeout state in persistent transport objects. Transport
device handles may live across calls and may be reused by many blocks. Storing a
started deadline there would make timeout state shared across unrelated blocks,
kernels, or operations.

## Why the abort check is amortized

This is the single most important performance property of the whole design, and
it is easy to undo by accident, so it is worth stating with numbers.

`AbortState` lives in **mapped pinned host memory** so that host and device see
one abort reason. That is what makes the contract work, and it is also what
makes a naive check ruinous: every read of the shared reason from the device is
an uncached PCIe round trip. Spin loops call the check on *every iteration*, and
in the LL small-message path that is up to 32 lanes x 2 loads per warp per
iteration.

So `checkExpired()` does not read shared state on most calls. It gates the read
behind the free device clock:

```cpp
const uint64_t now = detail::deviceClock();
const bool deadlineDue = deadlineCycles_ != 0 && now >= deadlineCycles_;
if (!deadlineDue && now < nextPollCycles_) {
  return false;          // register compare only -- no memory touched
}
nextPollCycles_ = now + pollIntervalCycles_;
```

`pollIntervalCycles_` is `cyclesPerMs_ / kAbortPollsPerMs` with
`kAbortPollsPerMs = 1`, i.e. **one shared read per millisecond per handle**,
independent of how fast the loop spins. Once a terminal reason has been seen,
`sawTerminalReason_` answers from a register forever.

That constant used to be 100, and gating the read was only half the job: the
gate makes the cost independent of loop speed, but the *rate* still sets a
fixed fraction of kernel runtime, namely `kAbortPollsPerMs x 1.1us / 1000us`.
At 100 that is 11% of every collective, and it measured as exactly that: 4-rank
IB_ONLY on GB300, `MCCL_ABORT_MODE=skip` against a same-session `none`, went
from **+11.4us to +3.1us on tree and +10.8us to +5.6us on ring** when this
constant moved to 1. Amortizing a 1.1us read to "only" once per 10us is not
amortizing it.

What that costs: abort-*observation* latency goes from ~10us to ~1ms. Deadline
expiry is unaffected — `checkExpired()` tests `deadlineDue` ahead of the
throttle, so a timeout still fires on time. This governs only how quickly one
rank notices *another* rank's abort, and the only thing delayed is how fast an
already-failed collective unwinds.

`startTimeout()` seeds `nextPollCycles_` rather than leaving it at zero, so the
first `checkExpired()` on an armed handle is throttled like every other. The
cost is that an abort raised before the kernel started is observed up to one
poll interval later, which is inside the bound this constant already advertises
and is unreachable in practice because the host checks `Abort::isAborted()`
before it launches.

### What it is worth

Measured on 8x H100 (`devgpu012.mwg1`) with
`comms/common/fault_tolerance/benchmarks:abort_bench`; full tables and
methodology in that directory's `Perf.md`.

| Row | Ungated | Gated | |
|---|---:|---:|---|
| `AbortDeviceIsAbortedLoadLoop` | 225.93ms / 100K polls | 4.87ms / 100K polls | 46x |
| `AbortDeviceIsAbortedWithDeadlineLoadLoop` | 226.28ms / 100K polls | 4.88ms / 100K polls | 46x |

That is ~2.26us per ungated poll against ~49ns gated. The
`CudaAtomicDeviceLoadLoop` row corroborates the mechanism independently: a bare
mapped-pinned load is 1.11us, so the ungated cost is simply "the shared read,
every time".

Two consequences worth internalizing:

- **Arming a deadline is free.** The with-deadline row is within noise of the
  no-deadline row (4.88ms vs 4.87ms) because the deadline is one more register
  compare on the same gated path. There is no performance argument for leaving a
  collective unarmed.
- **The cost of an abort check is a property of the poll interval, not of the
  loop.** A tighter spin loop does not make aborts more expensive. This is why
  Principle 7 says to gate the check rather than to call it less often.

### How to not undo it

- Do not add work to `checkExpired()` before the gate. Anything above the
  `now < nextPollCycles_` early return runs on every spin iteration of every
  wait in the codebase.
- Do not add a debug assertion to this path "just in case" -- it was considered
  for the arm-site invariant and rejected for exactly this reason; a static
  audit plus per-collective stall tests cover that without touching the gate.
- Prefer one poller per group over one per thread when the loop is per-thread.
  The gate is per-handle-copy, so N thread-local copies mean N times the shared
  reads.
- If you change `kAbortPollsPerMs`, re-run `abort_bench`, re-run the GB300
  sweep, and update this section, `Perf.md`, and the constant's own comment.
  It trades abort-detection latency against a fixed percentage of every
  collective's runtime, and the cost is linear in the value.

## MPT And Prims Integration

`MultiPeerTransport` is the device-handle propagation point for Prims
collectives. It is created for the communicator, receives the same host `Abort`
shared by MCCL and CTRAN, and includes a device abort handle in each
`MultiPeerDeviceHandle` it returns.

`MultiPeerDeviceHandle` exposes the abort handle as data. It should not wrap
additional convenience methods such as `isAborted()`. Kernel code should copy
the exposed `AbortDevice`, start a per-block timeout when needed, and pass that
local copy to waits.

Transport handles must not embed started abort timeout state. NVL, IBGDA, and
IBRC waits consume the operation-local `AbortDevice` passed by the collective or
kernel.

### Proxy-facing waits: bounded by the clock, never by polling

IBRC is the exception to "pass the operation-local handle in", and deliberately
so. Its producer-side waits — `reserve()` on a full command ring and
`drain_queue()` — are reached from `put()` / `signal()`, which carry no
deadline. Threading one down would put an abort parameter on every producer for
a value none of them consume.

**These loops do not poll the abort at all.** They bound themselves on
`deviceClock()` and, on expiry, *write* the shared state rather than reading it:

```cpp
while (/* ring full */) {
  if (gpu_clock64() - start >= kIbrcDefaultDeviceTimeoutCycles) {
    abort_.setAbort(AbortReason::IBRC_PROXY_TIMEOUT, "...");
    return kIbrcInvalidReadySeq;
  }
}
```

Three properties follow, and each replaces a defect this design used to have.

**1. The bound is unconditional.** It used to be gated on
`!abort.isEnabled()`, so *enabling* fault tolerance removed the only bound a
proxy-facing wait had. Combined with `P2pIbTransportDevice::flush()` dropping
the caller's deadline on the IBRC branch, a kernel that ended in `flush()` had
no watchdog, no deadline, and only an explicit host abort as an exit. The
watchdog is the contract between this kernel and the host proxy — a submitted
descriptor is consumed within a bounded time — and that obligation does not
change because FT is on.

**2. The bound is the fixed watchdog, not the collective's deadline.** These two
are different obligations with different owners, and the wait honours its own.
`kIbrcDefaultDeviceTimeoutCycles` bounds one rank's SM against its own host
proxy; the communicator timeout bounds the collective across ranks. So
`reserve()`, `drain_queue()` and `flush()` take no deadline argument and read no
timeout — the budget is a compile-time constant compared against `deviceClock()`,
which keeps the stall path free of mapped-host traffic and the producer
signatures free of an abort parameter. A caller that sets a shorter per-operation
deadline does not shorten the proxy watchdog, and a longer one does not extend
it.

**3. Poll state cannot be shared, by type.** The member is an `AbortFlag`, not
an `AbortDevice`. `AbortDevice` carries mutable throttle state
(`nextPollCycles_`, an absolute per-SM `clock64()`), and this object lives in
device memory that every block sees — so several blocks would write one field
non-atomically, and a value stamped by a leading SM could suppress a lagging
SM's polls far past the interval, hiding the abort it was parked on.
`AbortFlag` holds nothing mutable and exposes no poll, so that is unwritable
here. What a shared handle *may* do — report whether FT is on, and record a
terminal reason via a system-scope CAS — is all it offers.

Do not "fix" any of this by arming a copy in the wait. `startTimeout()` is
relative to the moment of the call, so every entry into `reserve()` would get a
fresh full budget and N stalled reserves would cost N × timeout.

**The reason is its own.** A stalled proxy latches
`AbortReason::IBRC_PROXY_TIMEOUT`, not `TIMED_OUT`. They are different faults
with different owners — the host proxy thread versus the collective's own
deadline — and the host log should say which. `isTimedOut()` stays exact and
does not report a proxy stall; `isAborted()` covers both.

**The residual trade, stated plainly.** Because these loops never read shared
state and their budget is the fixed watchdog, a block parked in one does not
observe *another* wait's abort promptly, and a collective deadline shorter than
the watchdog does not pull it out early — it leaves on the watchdog instead.
Liveness is what the contract requires and it is bounded either way, but unwind
is not instantaneous across a kernel whose producer is parked, and the worst-case
unwind for such a kernel is the watchdog rather than the configured timeout. The
single one-shot exception is the check before
`reserve()` claims a sequence number, which does read shared state so that a
producer on an already-aborted communicator leaves the wire untouched; it is
once per descriptor and never in a loop.


## MCCL Collective Integration

MCCL and CTRAN already share the same host `Abort`. MCCL collective code should
not add a second host abort pointer to `CommContext`.

MCCL creates enabled abort controllers with `AbortBehavior::SKIP` by default.
`McclCommCreateOpts::trapOnDeviceAbort` opts into `AbortBehavior::TRAP` for
callers that need the legacy device trap behavior. `MCCL_ABORT_MODE`
(`skip`/`trap`/`none`) overrides both at runtime; `none` disables abort support
for the communicator entirely.

The device-side abort deadline comes from `McclCommCreateOpts::abortTimeout`
when set, otherwise from a positive `MCCL_ABORT_TIMEOUT_MS`. It is deliberately
**not** derived from `InitOpts::timeout`: that value bounds bootstrap, and
reusing it as a per-collective watchdog would let a short init budget abort
healthy long collectives, or a long one silently remove the 30 s floor.
`commSplit()` carries the resolved deadline to the child communicator, matching
`reconfigure()`.

For Prims-backed collectives:

1. Host orchestration gets a `MultiPeerDeviceHandle` from MPT.
2. Kernel arguments carry either the full handle or the exposed
   `AbortDevice`.
3. Kernel entry creates a local per-block copy and calls `startTimeout()`.
4. Transport and synchronization waits poll that local abort handle.

## What Replaced the Prims `Timeout`

`AbortDevice` is the only spelling. The standalone Prims `Timeout` and its
transitional alias are gone: there is no per-launch GPU-cycle timeout object and
no `makeTimeout()` helper. Timeout duration comes from the communicator-owned
host `Abort` default timeout and is read by `AbortDevice::startTimeout()`.

The behavior differences from the old `Timeout` are intentional:

- A default-constructed `AbortDevice` is disabled and behaves like the previous
  no-timeout default.
- Handles borrowed from MPT observe explicit host aborts and timeout-triggered
  aborts through shared state.
- Timeout expiry records `AbortReason::TIMED_OUT` once in shared abort state,
  making the result visible to host code and other device consumers.

### Per-operation timeouts

A collective may override the deadline for a single operation by passing
`timeout` on its options struct. The launcher copies the communicator device
handle, calls `AbortDevice::setOpTimeoutMs()` on that copy, and stores it in the
kernel arguments, so the override travels by value and costs no shared-state
read.

When no per-operation timeout is supplied the override stays unset and the
device reads the communicator timeout from shared state on every
`startTimeout()`. That keeps it late-bound, so `IComm::setTimeout()` is observed
even by transports that cache one device handle for the communicator's lifetime.

`MCCL_ABORT_TIMEOUT_MS` seeds only the communicator timeout. It is never turned
into a per-operation override: doing so would snapshot it at launch and defeat
that late-binding.

## Integration Notes

What fault tolerance guarantees here, and what you must do to get it.

**The guarantee is liveness, not error propagation.** A collective on an aborted
communicator **completes quickly** rather than hanging. It does not promise a
meaningful result, and it does not promise that every layer forwards an error
code. Device waits stop waiting once an abort is visible; the collective then
runs to completion over whatever is in its buffers.

This is deliberate. Requiring every caller of every wait to consume and forward
an abort result is a large, easily-broken obligation: one missed call site is
either a hang or a silent wrong result. Making the *waits* self-terminating puts
the obligation in one place per loop instead.

It also matches the documented `IComm::abort` contract: results from work that
completes after an abort "should be treated as the reason for abort. In the
unlikely case of a `commSuccess`, the comm result data should still be ignored."

### Transport Foundation — adding or changing a wait

1. **Never block indefinitely.** Every unbounded loop must consult the
   `AbortDevice` and stop waiting once it reports abort. This is the whole
   correctness surface: a loop without a check is a hang.
2. **Prefer `break` over early `return`.** Falling through keeps every thread on
   the same path to the same barriers. An early `return` from a subset of a
   group leaves the rest at `group.sync()` / `__syncwarp()` naming threads that
   have exited, which is undefined behavior.
3. **If you must return early, make the decision group-uniform first.**
   `ThreadGroup` (`comms/prims/core/ThreadGroup.cuh`) offers exactly two
   reductions, and both barrier, so every thread in the group must reach them:
   - Leader-owned decision — the leader computes, then publishes with
     `group.broadcast<uint32_t>(stop)`. This is the preferred shape (principle
     4); it is also the cheaper one when only the leader polls.
   - Per-thread predicate — `group.all(pred)` is a group-wide AND, so
     "any thread saw it" is `!group.all(!pred)`. There is no `any()`.

   Do not add a new reduction primitive for a single call site.
4. **Never derive a loop bound or an index from peer-written data.** Bounds must
   come from local geometry or parameters. If garbage could extend a loop or
   move an index, "proceed on data that never arrived" stops being safe — this
   assumption is what lets a wait give up without unwinding the caller.
5. **Do not store started deadline state in a transport object.** Transport
   handles outlive individual operations and are shared across blocks. Copy the
   handle per block and call `startTimeout()` on the copy.
6. **Returning a status is welcome, not required.** Several waits return `bool`
   and the IB and NVL progress paths return `Aborted`; that lets callers stop
   sooner and more precisely. Callers are not *obliged* to consume it, so never
   rely on propagation for liveness.
7. **Leave the channel state releasable.** A wait that unwinds must not strand
   the resource it was waiting on. In the IB progress path this is
   `abandon_progress_state()` (`P2pIbTransportProgressImpl.cuh`): every abort
   exit drives the progress slot to `Done` before returning `Aborted`, so the
   next `init_send_progress()` / `init_recv_progress()` on that channel passes
   `assert_progress_slot_idle()` instead of trapping, and a driver loop that
   keeps calling progress simply drains. The reserved byte range is abandoned
   rather than returned to the channel cursor: a peer RDMA write may still land
   in it, and the flow-control counters are skewed after an abort regardless —
   the point is that the queued kernel exits instead of trapping, not that the
   channel is fit to reuse. If a new wait acquires state of its own, releasing
   it on abort is part of adding the wait — pushing that onto the collective
   violates the containment principle.

   The NVL progress path holds the same guarantee by the same shape:
   `abandon_progress_state()` in `P2pNvlTransportDevice.cuh` drives the slot to
   `Idle` — NVL's terminal stage — so the aborting call returns `Aborted` and a
   later progress call short-circuits to `Done`, and the next
   `init_*_progress()` passes `assert_progress_slot_idle()`. The channel cursor
   stays advanced there too, because the peer may still write into the reserved
   range over NVLink. `NvlSendRecvProgressStatus::Aborted` mirrors the IB enum,
   so a transport-agnostic driver loop keeps one abort branch across both
   transports: treat `Done` and `Aborted` alike as "stop polling this
   operation", and take `Aborted` as the signal to consult the host `Abort` for
   the reason rather than to record a completed transfer.

### Collective Enablement — onboarding a collective

1. Take the communicator handle from the device handle you already fetch:
   `mpt->get_device_handle(peers).abort`, and store it in your launch params /
   kernel arguments.
2. In the kernel, copy it per block and call `startTimeout()` once near entry.
3. Pass that local copy into every transport wait you call.
4. Optionally set a per-operation deadline with `setOpTimeoutMs()` on the copy —
   **only** from a timeout the caller supplied on the collective API. Never seed
   it from the communicator default; see *Per-operation timeouts*.
5. You do **not** need to check the return value of any wait to be hang-safe.
6. You do **not** need an abort gate, an entry guard, or slot bookkeeping of your
   own. The transport releases its state on abort (see *Principles*), so a loop that
   already retires on `Done` retires on an aborted channel too.

Steps 1–3 are the entire integration. Forgetting them means the collective runs
with a disabled handle and silently has no fault tolerance.

To catch that, call `debugCheckAbortWired()` on the host where you build your
launch parameters. It logs once when the communicator has fault tolerance
enabled but the handle you are about to pass is disabled — the signature of a
missing step 1 — and is compiled out in optimized builds.

> This document describes the end state of the abort migration and lands ahead
> of parts of it. `debugCheckAbortWired()` arrives with the Prims migration in
> D115656601; until then the diagnostic above is guidance, not an available
> call.

The check must be host-side: on the device a disabled handle is just
`state_ == nullptr`, which is identical for "fault tolerance is off for this
communicator" and "the collective forgot to pass the handle". Only the host can
see both the communicator's `Abort` and the handle being launched, so only the
host can tell those apart.

### One budget per kernel

`startTimeout()` computes `deadlineCycles_ = deviceClock() + timeoutMs *
cyclesPerMs_` **at the moment it is called**. There is no host-settable absolute
deadline, so a deadline is only ever as wide as the region between the `start()`
that armed it and the wait that observes it.

The rule that follows, and the one the table below audits:

> **Exactly one `start()` per `__global__` kernel, at entry. Everything below it
> takes the handle by reference.**

A second `start()` anywhere downstream re-arms from the current clock, so the
work after it gets a fresh full budget on top of whatever the work before it
already spent. Two arm sites in one kernel means the kernel can take twice the
deadline the caller asked for, and the effect is worst exactly when it matters
least — when things are already running slowly.

The accepted consequence: a collective that spans **two kernel launches gets two
budgets**. That is deliberate. Closing it would need the host to stamp an
absolute deadline into the handle, which means carrying a device-clock reference
sample to convert `steady_clock` into the device cycle domain. Not worth it
while one collective is one launch.

### Per-collective FT status

Onboarding is incremental. A collective that has not been onboarded carries a
**disabled** handle, and `AbortDevice::checkExpired()` returns `false`
immediately for those — so it is inert rather than wrong. What must never happen
is an *enabled* handle that nothing armed: the waits would then observe explicit
aborts but no deadline would ever fire.

| Collective | Kernel(s) | Arm site | Status |
| --- | --- | --- | --- |
| AllReduce Direct | `AllReduceIbDirect.cu` | `args.timeout.start()` at entry, forwarded to the 4-arg `runAllReduceFused` | onboarded |
| AllReduce Ring | `AllReduceIbRingImpl.cuh` | `runRing` arms at entry; passed to `runAllReduceFused` and `phase2IbRing` | onboarded |
| AllReduce Tree | `AllReduceIbTreeImpl.cuh` | kernel entry; passed to `runAllReduceFused` and `TreePhase2` | onboarded |
| SendRecv | `SendRecvLauncher.cu` | `abort.start()` at entry, threaded into every send/recv | onboarded |
| ReduceScatter Ring | `RingReduceScatterKernel.cuh` | kernel entry | onboarded |
| ReduceScatter Direct / DirectIbV2 | `prims/collectives/ReduceScatterDirect*.cu` | kernel entry | onboarded |
| ReduceScatter DirectIb (`ctdirect_ib`) | `ctran/algos/ReduceScatterDirectIb.cc` | **none — `params.abort` is left default-constructed** | **not onboarded** |
| AllGather Direct | `prims/collectives/AllGatherDirect.cu` | kernel entry, once in each of the three kernels | onboarded |
| AllGather Ring | `prims/collectives/RingAllgather.cu` | kernel entry | onboarded |
| AllToAllv (+ Ll128) | `prims/collectives/AllToAllv*.cu` | kernel entry | onboarded |

Two things this audit turned up that are worth keeping written down:

- `runAllReduceFused` has a **three-argument overload that arms a handle of its
  own**. It exists for callers that have no handle to pass. A caller that has
  already armed one must use the four-argument overload — using the short one
  is the easiest way to end up with two budgets in a kernel, and it is how Tree
  did before this was fixed.
- `prims::collectives::all_gather` takes its `AbortDevice` **by value and arms
  it**.
  That is correct for its current callers, which are all kernel entries handing
  in an unarmed handle, but it means a collective that ever passes its own armed
  handle would silently get it re-armed. Prefer taking it by reference if that
  call site appears.

### What the caller sees

- The collective completes; **output buffers are undefined** after an abort.
- Check `IComm::isAborted()` and `getAbortInfo()`.
- Where the host can determine it cheaply, the work handle also reports a
  non-success result — but a `commSuccess` after an abort is contract-legal and
  its data must still be ignored.

### Scope

Fault tolerance is a **MCCL-communicator** feature. Communicators created
through the NCCLX/CTRAN factory get a disabled `Abort`, so they have no device
deadline and no abort observation.

`ctdirect_ib` ReduceScatter is unbounded on **more than those comms**, and the
earlier wording here understated it. `McclComm::reduceScatter()` intercepts only
`ctring_ib`, so an ordinary **FT-enabled** MCCL communicator that selects
`ctdirect_ib` — via `NCCL_REDUCESCATTER_ALGO` or `ncclReduceScatterQuantize` —
falls through to `ctranReduceScatter()` and reaches
`ReduceScatterDirectIb.cc`, which deliberately leaves `params.abort`
default-constructed. The kernel therefore arms a disabled handle and has neither
the communicator's abort state nor its deadline, so peer loss is unbounded on a
communicator whose owner asked for fault tolerance. Thanks to @benrcarver for
catching that the status table and this section contradicted each other.

Until that route is either plumbed or rejected under FT, the table above marks
it **not onboarded**. This is a regression against the pre-migration code, which
applied `MCCL_ABORT_TIMEOUT_MS` unconditionally on that path. The debug diagnostic above
is scoped to FT-enabled communicators so it stays silent here and under
`MCCL_ABORT_MODE=none`.

## Usage Examples

Host explicit abort:

```cpp
auto abort = comm->getAbort();
abort->setAbort(
    comms::fault_tolerance::AbortReason::ABORTED,
    "user requested abort");
```

Host default timeout:

```cpp
auto abort = comm->getAbort();
abort->setDefaultTimeout(std::chrono::milliseconds{5000});
```

Kernel wait:

```cpp
__global__ void kernel(KernArgs args) {
  auto abort = args.handle.abort;
  abort.startTimeout();

  while (!ready()) {
    switch (abort.check()) {
      case comms::fault_tolerance::AbortCheckResult::CONTINUE:
        break;
      case comms::fault_tolerance::AbortCheckResult::SKIP:
        return;
      case comms::fault_tolerance::AbortCheckResult::TRAP:
        printf("CUDA ABORT ERROR: wait aborted\n");
        __trap();
    }
  }
}
```
Device explicit abort:

```cpp
abort.setAbort(
    comms::fault_tolerance::AbortReason::ABORTED,
    "device callsite");
```

The host copies the context into host-only storage. The device never stores the
string in mapped shared state; a device diagnostic may consume the
device-accessible string only at the winning callsite.

Device timeout:

```cpp
abort.startTimeout();
while (!ready()) {
  if (abort.check() != comms::fault_tolerance::AbortCheckResult::CONTINUE) {
    return;
  }
}
```
