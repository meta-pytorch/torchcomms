# CollTrace: Baseline Modifications

## Background

CollTrace records, per communicator, the metadata and timing of every collective
launched through `ncclLaunchKernel`: what it was, how big, which algorithm and
protocol, when the kernel started and finished. It backs the comm dump, the
collective watchdog, the lifecycle event feed, and `NCCL_COLLTRACE=algostat`.

The engine (`comms/utils/colltrace/`) has no dependency on `comms/ncclx`;
`meta/colltrace/CollTraceWrapper.{h,cc}` is the shared NCCLX adapter that turns
an `ncclKernelPlan` into a record and implements `newCollTraceInit` /
`newCollTraceDestroy`. Baseline holds only what cannot be reached from outside.

## Versions Affected

v2.30, v2.32. The launch hook is in `src/enqueue.cc` in v2.30 and in
`src/enqueue/enqueue.cc` in v2.32; every other touch point has the same path.
Both `def_build.bzl` files compile `meta/colltrace/*.cc` with the
`comms/utils/colltrace` deps; only v2.30's `src/Makefile` builds any `meta/`.

## Baseline Files Modified

### 1. `src/include/comm.h` — per-communicator state

In the NCCLX state block of `struct ncclComm`, with includes of
`comms/utils/colltrace/{AlgoStats,CollTraceInterface}.h` and `commSpecs.h`:

```cpp
  struct CommLogData logMetaData;
  std::shared_ptr<meta::comms::colltrace::ICollTrace> newCollTrace;
  std::shared_ptr<meta::comms::colltrace::AlgoStats> algoStats;
```

**Why in baseline**: `prepareNcclKernelColltrace` reads all three through
`plan->comm` on every launch, and `logMetaData` is also the identity ProxyTrace,
CommsMonitor and the comm dump read. In v2.32 these members, the two forced
changes below, the `nccl.h.in` C++ tail (item 5) and the build wiring come with
the shared `ncclComm` state block port, not with the CollTrace hooks.

Forced by adding non-trivial C++ members:

- The `static_assert(offsetof(struct ncclComm, startMagic | endMagic) ...)`
  checks after the struct are commented out (`-Winvalid-offsetof` on a type that
  is no longer standard-layout). In v2.32 they follow upstream's new
  `ncclNvls*Enabled` helpers, which stay, and carry a `[META]` comment.
- `src/device/common.cu` (an nvcc unit) must not include `comm.h`, which now
  pulls folly-backed headers nvcc cannot parse; nothing there uses `ncclComm`.
  v2.30 replaces the include with `checks.h` + `sym_kernels.h`; v2.32 drops it.

### 2. `src/init.cc` — filling `logMetaData`

v2.32: a static `ncclxFillLogMetaData(comm, commId)` in `init.cc`, called in
`ncclCommInitRankFunc` right after `commAlloc` in both the split and the
init/grow branch, before `bootstrapSplit` / `bootstrapInit`. v2.30: inline,
after `comm->cudaArch = cudaArch;` and before `initTransportsRank`. The fill:

```cpp
  comm->logMetaData.commId = commIdHash;
  comm->logMetaData.commHash = comm->commHash;
  comm->logMetaData.commDesc = NCCLX_CONFIG_FIELD(comm->config, commDesc);
  comm->logMetaData.rank = comm->rank;
  comm->logMetaData.nRanks = comm->nRanks;
```

**Why in baseline**: `commIdHash` is a local of this function (`commHash` for
init and grow, `0` for split/shrink children). `commDesc` must come through
`NCCLX_CONFIG_FIELD` because upstream's `parseCommConfig` never copies it onto
`comm->config`. The v2.32 placement matters because `proxyTraceInit` runs inside
bootstrap (`ncclProxyInit`) and registers the comm with NetworkPerfMonitor under
this identity. v2.30 fills it after bootstrap, so there an enabled
NetworkPerfMonitor stores an all-zero identity.

### 3. `src/init.cc` — lifecycle

**Init**: `NCCLCHECKGOTO(meta::comms::ncclx::newCollTraceInit(comm), res, fail)`
in `ncclCommInitRankFunc`, immediately before
`if (comm->useCtran_) createCtranComm(comm)`, which hands `comm->newCollTrace`
to CTRAN. `CommsMonitor::registerComm` follows both. v2.32 runs the three after
`ncclProgressCounterMonitorInit` and before the atomic `initState` store; v2.30
runs them after `initState` is set, before `// Trace this call for replay tool`. Both
this file and `enqueue.cc` include `meta/colltrace/CollTraceWrapper.h`
(`// @manual` in v2.32 only).

**Destroy**: in `commDestroySync`, `newCollTraceDestroy(comm)` right after
`CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);` and before
`destroyCtranComm` (which drops CTRAN's copy of the handle), hence before the
stream sync, the intra-node barrier (v2.32 only) and `ncclProxyStop()`. v2.32
logs a failure and continues so the rest of teardown still runs; v2.30 uses
`NCCLCHECKGOTO(..., ret, fail)`.

**Why in baseline**: init needs a fully initialised comm: it reads
`logMetaData`, and the watchdog plugin captures the raw `comm` and calls
`ncclCommGetAsyncError` from the CollTrace worker. Destroy is the only teardown
hook: `commFree` releases the `ncclCalloc`'d comm with a raw `free()`, so no
destructor ever runs on `ncclComm`.

**Destroy may not drop the last reference.** CommsMonitor snapshots
`newCollTrace` in `registerComm` and `deregisterComm` only marks it `DEAD`, so
with `NCCL_COMMSMONITOR_ENABLE` (on by default) `~CollTrace` and the worker join
run at process exit, or when a new comm reuses the address (on the registering
thread, after the registry lock is released). Do not assume the worker has stopped once
`newCollTraceDestroy` returns. Known gaps:

- **Address reuse (seen on v2.32)**: CommsMonitor keys entries on the comm
  pointer and never erases them, so `registerCommImpl` must replace a stale
  entry when a new comm lands on a destroyed comm's address; otherwise dumps
  return the dead comm's snapshot. See `commsmonitor_commdump_baseline.md`.
- (both trees) The watchdog's captured `comm` can be freed while a traced
  collective is in flight; the worker can then call `ncclCommGetAsyncError` on
  freed memory.
- (both trees) `ncclCommRevoke` sets `finalizeCalled` without stopping
  CollTrace, and `commReclaim` skips `commDestroySync` for finalized comms, so a
  revoked comm never reaches `newCollTraceDestroy`.

**v2.32: concurrent destroy.** `commReclaim` runs `commDestroySync` per
intra-process comm on its own thread (v2.30: serially). Safe: each call touches
only its own comm, and the lifecycle-feed registry is updated under a lock.

### 4. `enqueue.cc` — the launch path

In `ncclLaunchKernel`, right after the `void* extra[] = {...}` initializer and
before `int driverVersion;`, plus a trigger pair and `disarm()` around each of
the two launch calls:

```cpp
  auto colltraceHandle = ncclx::colltrace::prepareNcclKernelColltrace(plan, launchStream, comm->compCap);
  meta::comms::colltrace::CollTraceEnqueueGuard colltraceEnqueueGuard{colltraceHandle};
  ...
    colltraceHandle->trigger(meta::comms::colltrace::CollTraceHandleTriggerState::BeforeEnqueueKernel);
    CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
    colltraceHandle->trigger(meta::comms::colltrace::CollTraceHandleTriggerState::AfterEnqueueKernel);
    colltraceEnqueueGuard.disarm();
```

`prepareNcclKernelColltrace` feeds `comm->algoStats`, records the plan (a
`GraphCudaWaitEvent` if `plan->persistent`, else a `CudaWaitEvent`) and arms
`plan->kernelArgs->colltraceHdr`. It returns a dummy handle if `NCCL_COLLTRACE`
is empty, `comm->newCollTrace` is null (e.g. `NCCL_COLLTRACE=algostat` alone),
recording fails, or a graph-captured plan has no coll/p2p task or is symmetric.
Under capture, timing comes only from the kernel, which is armed on sm_90+;
older GPUs record graph-captured collectives without timing.

- **The declaration position is load-bearing.** The first `goto do_return` is
  the `NCCLCHECKGOTO(ncclCudaDriverVersion(...))` right after
  `int driverVersion;`, and jumping over these initializations is ill-formed.
  v2.32's new locals (`userKernelEvent`, `relayStream`, ...) precede `extra[]`.
- **The two pairs are a CUDART/driver split, not graph vs eager**:
  `cuLaunchKernelEx` when `CUDART_VERSION >= 11080 && driverVersion >= 11080`,
  legacy `cuLaunchKernel` otherwise. Exactly one pair fires per launch.
- **The guard covers every early exit.** A `goto do_return` before `disarm()`
  (including v2.32's relay-stream acquire) cancels the record, which would
  otherwise stay pending and make the next collective report an overlap.
- **v2.32 launch-completion event.** `ncclCollConfig_t::launchCompletionEvent`
  adds (a) a fallback `cudaEventRecord` before the launch (CUDA < 12.3) and (b)
  on the `cuLaunchKernelEx` path, a relay record after it. `BeforeEnqueueKernel`
  goes after (a); `AfterEnqueueKernel` and `disarm()` go before (b). A failure
  in (a) cancels the record; a failure in (b) keeps it, as the kernel is queued.

**Why in baseline**: the one place with both the finished plan and the launch
stream, and the last point to edit the kernel-args blob before the driver
copies it. There is no plugin seam.

**Coverage gap** (both trees): `doLaunches` sends CE plans to `ncclLaunchCeColl`
and RMA plans to `ncclLaunchRma`, bypassing `ncclLaunchKernel`; no record.

**Symmetric kernels** are never armed for in-kernel timestamps, and:

- v2.30's scheduler frees each task, so the plan arrives empty. Eager: a
  placeholder record (`"Unknown"` / `"EmptyKernelTask"`, rate-limited error,
  skipped by AlgoStats). Graph capture: a dummy handle.
- v2.32's scheduler queues tasks on `plan->collTaskQueue`. Eager: real metadata
  and AlgoStats. Graph capture: a dummy handle rather than a never-timed graph
  record, because the shared `getMetadataFromNcclKernelPlan` returns no metadata
  for `persistent && isSymColl` (`PersistentSymmetricPlanReturnsNoMetadata`).

### 5. `src/nccl.h.in` — public API

`#define NCCL_COLLTRACE_CUDA_GRAPH_COMPATIBLE` goes after `#include <stdint.h>`,
before `/* Opaque handle to communicator */` (among NCCLX feature macros in
v2.30; alone under a `[META]` comment in v2.32). It is **not decorative**:
launchers check the installed `include/nccl.h` for that plain substring and only
then set `NCCL_COLLTRACE`, `NCCL_PROXYTRACE`, `NCCL_MAPPERTRACE_ENABLE` and
related settings. The directive must appear exactly once, with one space between
`#define` and the name; the `[META]` comment deliberately does not spell it out.
Renaming or respacing it disables fault localization silently, and a second
copy would keep the check passing after the macro is removed.

In the `#ifdef __cplusplus` tail after `} // end extern "C"` (C++ signatures),
after `ncclCommDump` / `ncclCommDumpAll` (directly after them in v2.32):

- `namespace ncclx::colltrace`: `kInvalidReplayId`, `LifecycleEventType`,
  `LifecycleEvent`, `getCollTraceCommId`, `getLatestCollTraceCollectiveId`,
  `drainUnreadLifecycleEvents`.
- `NCCL_HAS_DUMP_ALGO_STAT` and `dumpAlgoStat`. Keep the
  `#if !defined / #elif == 0 / #undef` idiom: `comms/mccl/adapter` builds with
  `-DNCCL_HAS_DUMP_ALGO_STAT=0` to opt out.

On the OSS make path, `src/version.script`'s `*nccl*` exports the mangled
`ncclx::colltrace` symbols; `libnccl.map` (CMake) cannot match mangled names.

## `opCount` and the contract with ProxyTrace

CollTrace stamps `plan->comm->opCount` in `ncclLaunchKernel`; ProxyTrace stamps
`comm->opCount` into `traceArgs` (`proxyTraceInfoCopy` in
`ncclAddProxyOpIfNeeded`) as `ncclLaunchPrepare` builds proxy ops. For a dump to
correlate them, the counter must be final before proxy ops are built.

| tree | increment site | granularity |
| --- | --- | --- |
| upstream 2.30 | `ncclProxyStart` (`proxy.cc`) | per plan, **after** its proxy ops are built |
| NCCLX v2.30 | `ncclLaunchPrepare` plan loop; upstream's removed | per plan, **before** its proxy ops are built |
| upstream = NCCLX 2.32 | `ncclEnqueueCheck`, after `taskAppend` | per enqueue API call |

**v2.32 carries no `opCount` hunk**: the increment precedes `ncclEnqueueCheck`'s
own `ncclGroupEndInternal()`, which runs `ncclLaunchPrepare` and
`ncclLaunchKernel`. Expected differences from v2.30:

- It counts enqueue API calls (collectives, `ncclSend` / `ncclRecv`,
  `ncclPutSignal` / `ncclSignal` / `ncclWaitSignal`), not plans. Plans of one
  multi-plan group share the final value: their CollTrace records share an
  `opCount`, and ProxyTrace merges them into one `commHash:opCount` entry.
- Observed with `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` on 4 GPUs (NET/IB), one
  4 MiB `ncclSend` + `ncclRecv` per group: v2.32's `PT_pastColls` SendRecv
  entries have `opCount` 2, 4, 6, ... and `nProxyOps` 2; v2.30 records 1, 2,
  3, ... with `nProxyOps` 8 (4 channels). Tests encoding v2.30's arithmetic
  need re-baselining.

## Tests on v2.32

- Enabled: `colltrace_wrapper_ut`, `new_colltrace_ut`, `dump_new_colltrace_ut`,
  `baseline_cudagraph_colltrace_dist` (in-kernel graph timestamps end to end),
  `new_colltrace_dist_local` and `dump_algo_stat_ctran_test` (with CTRAN), and
  `comms_monitor_dist_nolocal` (its two dump cases now drop the always-present
  `"GlobalInfo"` entry; they failed identically on v2.30).
- Also enabled: `comms_monitor_ut` and `comm_dump_test`, once CommsMonitor
  replaces stale entries on address reuse.
- Excluded: `proxytrace_dist_fastinit` and `mappertrace_dist_nolocal` (need two
  or more nodes); the `colltrace_watchdog` tests and `new_colltrace_dist_nolocal`
  (need `ncclx::setGlobalHint` and the ncclx RMA window API, not yet on v2.32).

## What to re-check at the next rebase

- `ncclLaunchKernel`: first `goto do_return` still after `extra[]`? New early
  exits or post-launch steps? Before / After / `disarm()` stay directly around
  the launch call. Does a new plan kind bypass it in `doLaunches`?
- Symmetric scheduler: if it stops queuing on `plan->collTaskQueue`, revisit the
  eager-path metadata and the `persistent && isSymColl` guard.
- `ncclCommInitRankFunc`: the fill stays right after `commAlloc`, before
  bootstrap; `newCollTraceInit` stays before `createCtranComm`.
- `commFree` / `commReclaim`: still a raw `free()`? New destroy concurrency?
- `opCount`: if the increment moves, it must still precede proxy-op building in
  `ncclLaunchPrepare`; otherwise restore a v2.30-style hunk.
- `nccl.h.in`: the macro line survives any upstream reformat, exactly once.
- State block port: the `static_assert`s and the `device/common.cu` include.

## Related

- `inkernel_colltrace_baseline.md`: the device-side emit the arming step feeds.
- `proxytrace_baseline.md`: the other half of the `opCount` contract.
- `commsmonitor_commdump_baseline.md`: CommsMonitor's registry and reference.
