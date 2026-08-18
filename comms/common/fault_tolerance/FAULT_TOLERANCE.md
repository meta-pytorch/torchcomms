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
4. **Group uniformity idiom: leader polls, then broadcasts.** Use the existing
   leader + `broadcast<uint32_t>` shape rather than adding a new reduction
   primitive for one call site.
5. **Never derive a loop bound or an index from peer-written data.** Bounds come
   from local geometry or parameters. This is the assumption that lets a wait
   give up without unwinding the caller.
6. **Do not store started deadline state in a transport object.** Transport
   handles outlive operations and are shared across blocks; copy the handle per
   block and call `startTimeout()` on the copy.
7. **Keep abort polling off hot paths.** Gate the check behind a spin count so
   it only engages once actually stalled, and prefer one poller per group over
   one per thread when the loop is per-thread.
8. **Onboard every new or in-flight wait as it lands**, including
   work-in-progress paths. A spin loop that reaches main without an abort check
   is a hang waiting to happen.
9. **Per-operation timeouts come only from the collective API.** The
   communicator deadline stays late-bound in shared state so `setTimeout()` is
   observed by already-created device handles.
10. **Validate a timeout's sign, not its size.** Choosing a sane magnitude is
    the caller's job; `std::chrono` types already make unit mistakes unlikely.
    A negative value must be rejected, because at the `AbortDevice` layer it
    silently means "no override".
11. **Fault tolerance is an MCCL-communicator feature.** Communicators created
    through the NCCLX/CTRAN factory get a disabled `Abort`; see *Scope*.

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
- `AbortReason::NONE` means no terminal reason has been recorded.

Host and device writers only transition from `NONE` to one terminal reason.
Later writers cannot overwrite the recorded reason.

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

## Prims `Timeout` Compatibility

Prims `Timeout` is a source-compatible alias to `AbortDevice` during the
migration:

```cpp
using Timeout = comms::fault_tolerance::AbortDevice;
```

This preserves existing kernel signatures while removing the old standalone
`Timeout` implementation. There is no per-launch GPU-cycle timeout object and
no `makeTimeout()` helper. Timeout duration comes from the communicator-owned
host `Abort` default timeout and is read by `AbortDevice::startTimeout()`.

The behavior change is intentional:

- Default-constructed `Timeout`/`AbortDevice` is disabled and behaves like the
  previous no-timeout default.
- Handles borrowed from MPT observe explicit host aborts and timeout-triggered
  aborts through shared state.
- Timeout expiry records `AbortReason::TIMED_OUT` once in shared abort state,
  making the result visible to host code and other device consumers.

New code should name the concrete type `AbortDevice`. `Timeout` spelling is
only for migration compatibility and should not be used in new Prims APIs.

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
   and the IB progress path returns `Aborted`; that lets callers stop sooner and
   more precisely. Callers are not *obliged* to consume it, so never rely on
   propagation for liveness.

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

### What the caller sees

- The collective completes; **output buffers are undefined** after an abort.
- Check `IComm::isAborted()`, `getAbortReason()` and `getAbortReasonStr()`.
- Where the host can determine it cheaply, the work handle also reports a
  non-success result — but a `commSuccess` after an abort is contract-legal and
  its data must still be ignored.

### Scope

Fault tolerance is a **MCCL-communicator** feature. Communicators created
through the NCCLX/CTRAN factory get a disabled `Abort`, so they have no device
deadline and no abort observation; the `ctdirect_ib` ReduceScatter path is
unbounded on those comms. This is an accepted limitation, not an oversight, and
it is a regression against the pre-migration code, which applied
`MCCL_ABORT_TIMEOUT_MS` unconditionally on that path. The debug diagnostic above
is scoped to FT-enabled communicators so it stays silent here and under
`MCCL_ABORT_MODE=none`.

## Usage Examples

Host explicit abort:

```cpp
auto abort = comm->getAbort();
abort->setAbort(comms::fault_tolerance::AbortReason::ABORTED);
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
abort.setAbort(comms::fault_tolerance::AbortReason::ABORTED);
```

Device timeout:

```cpp
abort.startTimeout();
while (!ready()) {
  if (abort.check() != comms::fault_tolerance::AbortCheckResult::CONTINUE) {
    return;
  }
}
```
