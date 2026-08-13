# Fault Tolerance Abort Contract

This document describes the shared abort contract used by MCCL, CTRAN, and
Prims device code.

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
callers that need the legacy device trap behavior. MCCL initializes the shared
default timeout from the communicator init timeout when provided; otherwise a
positive `MCCL_ABORT_TIMEOUT_MS` supplies the default device-side abort timeout.

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
