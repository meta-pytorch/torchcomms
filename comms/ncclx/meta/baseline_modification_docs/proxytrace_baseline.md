# ProxyTrace: Baseline Modifications

## Background

ProxyTrace records, for every send/recv proxy sub-op on the network transport,
the latest step it reached in each state of the proxy state machine —
`POSTED`, `REM_FIFO_WAIT`, `RECEIVED`, `TRANSMITTED`, `DONE` — with a timestamp,
plus the bytes moved. When a job hangs, that is what lets a dump say *which*
channel, to *which* peer, stalled at *which* step. It is enabled by
`NCCL_PROXYTRACE` and feeds `ProxyTrace::dump()`, the comm dump, and
`CommsMonitor`. `ProxyMock` is the companion fault injector:
`NCCL_PROXYMOCK_NET_SEND_FAILURE` makes sends matching a trace identity skip
their `isend`, which is how the hang-detection paths are tested.

The implementation is entirely in shared `meta/colltrace/` —
`ProxyTrace.{h,cc}`, `ProxyTraceFunc.{h,cc}`, `ProxyMock.{h,cc}`. Everything
below is baseline for one of two reasons: the state machine lives in `static`
functions in `transport/net.cc` with no callback seam, or the trace payload has
to ride inside the baseline proxy structs themselves.

## Versions Affected

v2.30, v2.32

## How the payload travels

1. While the plan is built, `proxyTraceAddBasicInfo` stamps `nChannels` and
   `coll` on the `ncclProxyOp`. `ncclAddProxyOpIfNeeded` clones it into the
   plan's `proxyOpQueue`, and `proxyTraceInfoCopy` stamps `commHash`,
   `opCount`, `rank` and `remoteRank` (from `op->root`) on the clone.
2. At launch, `ncclProxySaveOp` eventually `memcpy`s the whole op into
   `ncclProxyOpsPool`, a **shared-memory** segment. `ProxyTraceArgs` is
   scalar-only and trivially copyable, so it survives.
3. On the **proxy** thread, `ncclProxyOpToArgs` copies it op -> sub, and the
   `net.cc` progress functions read `sub->traceArgs`. `startSend` /
   `startRecv` assign the `proxyOpId` that later hooks and `ProxyMock` use.

`ncclProxyOpsPool` embeds `ops[MAX_OPS_PER_PEER * NCCL_MAX_LOCAL_RANKS]`, so
`traceArgs` (48 bytes on x86-64) is paid per slot. `MAX_OPS_PER_PEER` is 2048 in
both versions, but upstream 2.32 doubled `NCCL_MAX_LOCAL_RANKS` from 72 to 144:
the pool grows by about 6.75 MiB on v2_30 and about 13.5 MiB on v2_32.

## Baseline Files Modified

### 1. `src/include/proxy.h` — the payload

`#include "meta/colltrace/ProxyTrace.h"` goes after the `<atomic>` / `<mutex>` /
`<condition_variable>` block, not with the nccl headers: `ProxyTrace.h` uses
`std::string` and `std::chrono` without including them itself.

Three members, each appended at the tail of its struct under a
`// NCCLX - ProxyTrace` comment: `struct ProxyTraceArgs traceArgs;` in
`ncclProxyOp` (after `enqNext`) and in `ncclProxySubArgs` (after
`recvRequestsSubCount`), and `std::shared_ptr<ProxyTrace> trace{nullptr};` in
`ncclProxyState` (after `expectedResponses`).

`shared_ptr` because `meta/comms-monitor/CommsMonitor.cc` copies
`comm->proxyState->trace` into a handle of its own. A non-trivial member works
only because upstream allocates the state with `new ncclProxyState{}` and frees
it with `delete`; if upstream switches to `calloc` / `free`, this must change.

**Note for future ports**: v2_30's `ncclProxyState` also has NCCLX-only
`uint64_t commHash` and `ncclComm* owner` (other hooks; ProxyTrace never reads
them). v2_32 does not. If they are ported, check that a three-way merge at the
struct tail kept all three NCCLX members.

### 2. `src/proxy.cc` — lifecycle and the op -> sub copy

Includes `meta/colltrace/ProxyTraceFunc.h`. Three hooks:

- `ncclProxyOpToArgs`: `PROXY_TRACE_OP_TO_SUBARGS(sub, op);` right after the
  `subIndex >= NCCL_PROXY_MAX_SUBS` bounds check (v2_30 has it before upstream's
  commented-out `memset`, v2_32 after; either is fine).
- `ncclProxyInit`:
  `NCCLCHECK(ncclx::colltrace::proxyTraceInit(comm->proxyState, comm));` after
  `comm->proxyState->netAttr = NCCL_NET_ATTR_INIT;`, before the
  `ncclIpcSocketInit` UDS setup.
- `ncclProxyDestroy`:
  `NCCLCHECK(ncclx::colltrace::proxyTraceDestroy(sharedProxyState));` inside
  `if (sharedProxyState)`, after upstream's `refCount != 0` early return, before
  the `free(...)` calls.

**Deviation from v2_30**, which calls destroy unconditionally at the top of the
function (and keeps 2.30's `assert(refCount == 0)` instead of the early return).
`proxyTraceDestroy` dereferences `state` unchecked, so it needs the guard. On
the `refCount != 0` path the state and its proxy thread survive, so resetting
the trace there could race a `PROXY_TRACE_CALL`. On the normal path the call is
belt-and-braces, since `delete sharedProxyState` drops the `shared_ptr` anyway.

**Identity at init**: `ncclProxyInit` runs from `bootstrapInit` /
`bootstrapSplit`, and `proxyTraceInit` registers the comm with
`NetworkPerfMonitor` under `comm->logMetaData`. v2.32 fills `logMetaData` right
after `commAlloc`, before bootstrap, so the registration carries the real
identity; v2.30 fills it after bootstrap and registers commHash 0.
`NetworkPerfMonitor` guards its comm-info map with a lock, since per-device init
jobs (`ncclCommInitAll`) register concurrently.
Compile-time prerequisites: `ncclComm::logMetaData`, `ncclConfig_t::commDesc`.

### 3. `src/transport/net.cc` — the state machine

Includes `meta/colltrace/ProxyMock.h` and `ProxyTrace.h` after
`register_inline.h`. Fourteen `PROXY_TRACE_*` sites, seven per direction, plus
the `ProxyMock` wrapper. Each step hook sits right after the `ncclProfiler*`
call for that transition, which follows the step-counter update.

| function | hook | attached to |
| --- | --- | --- |
| `sendProxyProgress` | `startSend` | end of the `args->state == ncclProxyOpReady` block, after the per-sub init loop, before `args->state = ncclProxyOpProgress` |
| | `recordSendProgress(..., sub->posted, POSTED)` | after `sub->posted += args->sliceSteps` and `ncclProfilerProxyStepSendGPUWait`, before `args->idle = 0; continue;` |
| | `recordSendProgress(..., sub->transmitted + args->sliceSteps, REM_FIFO_WAIT)` | inside `if (ready)`, after `ncclProfilerProxyStepSendPeerWait_v4`, before the `isend`; the step *about to* be sent |
| | `ProxyMockNetSendFailure::mock(sub, sub->transmitted, sub->requests + buffSlot)` | wraps `proxyState->ncclNet->isend(...)`; `isend` runs only when `mock` returns `false` |
| | `recordSendProgress(..., sub->transmitted, TRANSMITTED)`, then `PROXY_TRACE_EXECUTE(sub->traceArgs.transSize += size)` | inside `if (sub->requests[buffSlot] != NULL)`, after `sub->transmitted += args->sliceSteps` and `ncclProfilerProxyStepSendWait` |
| | `recordSendProgress(..., sub->done, DONE, size)` | `ncclNet->test` completion block, after `sub->done += args->sliceSteps` and `ncclProfilerStopProxyStepEvent`, before the `sendHead` publish |
| | `completeSend` | inside `if (args->done == args->nsubs)`, immediately after `args->state = ncclProxyOpNone` |
| `recvProxyProgress` | `startRecv` | end of the `ncclProxyOpReady` block, after the loop that sets `sub->base` / `groupSize`, before `args->state = ncclProxyOpProgress` |
| | `PROXY_TRACE_EXECUTE(sub->traceArgs.transSize += sizes[i])`, then `recordRecvProgress(..., sub->posted, POSTED)` | per-sub loop under `if (*requestPtr)` after `irecv`: after `sub->posted += args->sliceSteps` and `ncclProfilerProxyStepRecvWait` |
| | `recordRecvProgress(..., sub->received, RECEIVED)` | `ncclNet->test` completion loop, after `sub->received += args->sliceSteps` and `ncclProfilerProxyStepRecvFlushWait` |
| | `recordRecvProgress(..., sub->transmitted, TRANSMITTED)` | flush-completion loop, after `sub->transmitted += args->sliceSteps` and `ncclProfilerProxyStepRecvGPUWait`, before the `*recvTail` publish |
| | `recordRecvProgress(..., sub->done, DONE)` | `sendHead` drain loop, after `sub->done += args->sliceSteps` and `ncclProfilerStopProxyStepEvent` |
| | `completeRecv` | inside `if (args->done == args->nsubs)`, immediately after `args->state = ncclProxyOpNone` |

Things to preserve when re-applying:

- **Completion hooks follow the state reset.** Upstream orders the per-sub
  `ncclProfilerStopProxyOpEvent` loop differently in the two functions, so
  `completeSend` fires after that loop and `completeRecv` before it.
- **Start before record, complete exactly once.** Hooks are `NCCLCHECK`ed, and
  `record*` / `complete*` return `ncclInternalError` without the entry `start*`
  created (`complete*` removes it). Getting this wrong fails the proxy thread
  whenever `NCCL_PROXYTRACE` is set.
- **Only the send-side `DONE` hook passes `size`** (for the `NetworkPerfMonitor`
  RDMA completion event).
- **Recv `transSize` adds the requested `sizes[i]` at post time**, hence
  `ProxyTraceColl::totalRecvSize` is documented as approximate. Do not "fix" it
  by moving it to the `RECEIVED` hook.

**Why in baseline**: `sendProxyProgress` and `recvProxyProgress` are `static`,
the state transitions exist nowhere else, and there is no callback seam.

**Do not** add hooks to `coll_net.cc`'s own static `sendProxyProgress` /
`recvProxyProgress`; neither version traces those. v2_32's `net.cc` is
clang-formatted, so a v2_30 patch won't apply by line, but every anchor in the
table is the same in both versions.

### 4. `src/enqueue/enqueue.cc` (v2_30: `src/enqueue.cc`) — identity stamping

Includes `meta/colltrace/ProxyTraceFunc.h` next to the `CollTraceWrapper.h`
include. In `ncclAddProxyOpIfNeeded`, right after
`*q = *op; // C++ struct assignment` and before `ncclIntruQueueEnqueue`:
`ncclx::colltrace::proxyTraceInfoCopy(*q, comm);`. Plus three
`proxyTraceAddBasicInfo` sites:

| function | after | call |
| --- | --- | --- |
| `scheduleCollTasksToPlan`, `if (task->isCollnet)` branch | `calcCollChunking(..., &proxyOp)`, before the per-channel loop (v2_30: after `devWork->channelLo = 0;`) | `proxyTraceAddBasicInfo(proxyOp, nChannels, task->func)` |
| `scheduleCollTasksToPlan`, non-CollNet per-channel loop | `proxyOp->ringAlgo = NULL;` | `proxyTraceAddBasicInfo(*proxyOp, nMaxChannels[kind], task->func)` |
| `addP2pToPlan`, per-direction op setup loop | `op->eActivationMask = ...;` | `proxyTraceAddBasicInfo(*op, nChannels[dir], static_cast<ncclFunc_t>(op->coll))` |

Each must run before the `ncclAddProxyOpIfNeeded` whose `*q = *op` copy carries
`nChannels` and `coll` forward.

**Losing these four calls fails silently.** ProxyTrace still records every op,
but with zeroed (collective) or default (p2p: `opCount` `UINT64_MAX`, ranks -1)
identity, so dumps can't be attributed and `ProxyMock` matches the wrong ops.
`proxy_mock_ut` still passes; check by reading the file after every rebase.

**Why in baseline**: `commHash`, `opCount`, `rank` and `root` are only all
available while the op is being built inside the planner.

Things worth knowing about these sites:

- The collnet call passes `task->func`, **not** v2_30's
  `proxyOp.task.coll->func`: `calcCollChunking()` memsets `proxyOp` and
  `task.coll` is assigned only in the loop below, so v2_30 dereferences null
  (SIGSEGV on every CollNet collective). v2_30 still carries it.
- Same as v2_30: the unused direction of a one-sided p2p is stamped
  `ncclFuncBroadcast` (`op->coll == 0`) but has 0 channels and is never queued;
  `remoteRank` is the root, not the ring neighbour, for collectives; and
  `scheduler/allgatherv_sched.cc` ops get `proxyTraceInfoCopy` only (`coll`
  Broadcast, `nChannels` 0).
- v2_30's second, redundant `proxyTraceAddBasicInfo` after
  `ncclAddWorkBatchToPlan` (identical values) is not ported.

## Deliberate deviations from v2_30

1. The collnet `proxyTraceAddBasicInfo` passes `task->func`.
2. The redundant second per-channel `proxyTraceAddBasicInfo` is dropped.
3. `proxyTraceDestroy` moves inside the guard, after the `refCount` check.

## Other differences from v2_30

**Profiler proxy ops.** Upstream removed the profiler-proxy-op mechanism after
2.30 (`addProfilerProxyOpIfNeeded()`, `SaveProxyProfiler()`,
`ncclPatternProfiler`, `transport/profiler.cc`); nothing replaces v2_30's
companion calls next to `ncclAddProxyOpIfNeeded`. ProxyTrace counts are
unaffected: v2_30 never traced `profilerProxyProgress`.

**opCount granularity.** `opCount` is read from `comm->opCount` at plan-build
time. v2_32 relies on upstream's increment in `ncclEnqueueCheck` (per API call)
instead of v2_30's `ncclLaunchPrepare` hunk (per plan); see
`colltrace_baseline.md`. Running `meta/tests:comm_dump_test` `DumpAfterSendRecv`
alone with `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` on 4 local GPUs (NET/IB), one
4 MiB `ncclSend` + `ncclRecv` per group, `PT_pastColls` records `SendRecv`
entries with:

| version | opCount sequence | `nProxyOps` per entry |
| --- | --- | --- |
| v2_30 | 1, 2, 3, ... | 8 (4 channels) |
| v2_32 | 2, 4, 6, ... | 2 |

This run exercises the `net.cc`, `proxy.cc` and `enqueue.cc` hooks end to end;
no test enabled on v2_32 does. Anything asserting on ProxyTrace opCount or
`nProxyOps` must account for the difference.

## Build Integration

Both versions compile the shared sources through the `meta/colltrace/*.cc` glob
in `def_build.bzl`, with `comms/ctran/utils:checks`, `comms/utils:str_utils` and
`comms/utils/colltrace:network_perf_monitor` as deps. v2_30's OSS `src/Makefile`
globs `meta/colltrace/*.cc`; v2_32's does not list any `meta/` sources yet.

## Tests

- `meta/colltrace/tests:proxy_mock_ut` is re-enabled on v2_32. It uses
  hand-built `ncclProxySubArgs`, so it exercises no baseline hook.
- `meta/tests:comm_dump_test` is enabled on v2_32. Its `DumpAfterSendRecv`
  checks the `PT_*` keys but runs on P2P/SHM, so it does not exercise the
  `net.cc` hooks unless run with `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`.
- `meta/colltrace/tests:proxytrace_dist_fastinit` stays excluded: every case
  skips unless `comm->nNodes >= 2`, and the target runs one node. Its expected
  opCounts assume v2_30's per-plan counting and are unverified on v2_32.

## What to re-check at the next rebase

- `MAX_OPS_PER_PEER` and `NCCL_MAX_LOCAL_RANKS` (shared-memory pool growth).
- `ncclProxyState` is still `new` / `delete`d, its NCCLX tail members survived
  the merge, and `ncclProxyDestroy` still has the `refCount != 0` return.
- Every `net.cc` anchor and the completion-hook placement.
- The `enqueue.cc` `proxyTraceInfoCopy` and all three `proxyTraceAddBasicInfo`
  calls are present (nothing fails if they are not).
- Where `comm->opCount` is incremented, and that CollTrace sees the same value.
- `logMetaData` is still filled before `ncclProxyInit`.

## Related

- `colltrace_baseline.md`, in particular the `opCount` section: ProxyTrace and
  CollTrace must observe the same `comm->opCount`.
- `commsmonitor_commdump_baseline.md`: `CommsMonitor` holds a `shared_ptr` to
  the trace, and the comm dump serializes it.
