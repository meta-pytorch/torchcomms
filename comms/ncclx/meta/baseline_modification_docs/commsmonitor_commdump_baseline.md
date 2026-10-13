# CommsMonitor / CommDump: Baseline Modifications

## Background

- **CommsMonitor** (`meta/comms-monitor/CommsMonitor.{h,cc}`) is a process-wide
  registry of communicators. At init it snapshots each comm's identity
  (`logMetaData` plus topology) and `shared_ptr`s to its CollTrace, ProxyTrace,
  MapperTrace, MemoryTrace and AlgoStats into a `folly::Synchronized` map keyed
  on the `ncclComm_t` pointer.
- **CommDump** (`meta/commDump.{h,cc}`) turns one communicator, or via the
  registry all of them, into a flat `{key: value}` map. It implements
  `ncclCommDump` / `ncclCommDumpAll`, which out-of-tree consumers such as the
  NCCLX process group and torchcomms call to collect communicator state.

With `NCCL_COMMSMONITOR_ENABLE` (default true) both read the registry snapshot
(an unregistered comm dumps nothing). With it off, `ncclCommDump` reads the live
comm and `ncclCommDumpAll` returns `ncclInternalError`. Both are shared code;
the baseline only reports comm birth and death and declares the entry points.

## Versions Affected

v2.30, v2.32

## Baseline Files Modified

### 1. `src/init.cc` — registry lifecycle

Includes `meta/comms-monitor/CommsMonitor.h` (with `// @manual` on v2_32 only).
One register, three deregisters.

**Register**, `ncclx::comms_monitor::CommsMonitor::registerComm(comm);` in
`ncclCommInitRankFunc`, after `newCollTraceInit(comm)` and the
`if (comm->useCtran_) createCtranComm(comm)` block (which hands `newCollTrace`
to CTRAN). v2_32 runs all three before publishing `initState`; v2_30 runs them
after it, before `// Trace this call for replay tool`. Every creating path (`ncclCommInitRank*`, `ncclCommInitAll`,
`ncclCommSplit`, `ncclCommShrink`, `ncclCommGrow`) runs this function; a
`NCCL_SPLIT_NOCOLOR` rank exits before a comm exists.

Ordering is load-bearing: `registerComm` *snapshots* the comm, so everything it
reads must already be populated (same on both versions):

| Snapshot field | Populated by |
| --- | --- |
| `logMetaData` | the NCCLX fill in `ncclCommInitRankFunc` (v2_32: right after `commAlloc`, before bootstrap; v2_30: after bootstrap) |
| `proxyState->trace` | `proxyTraceInit` in `ncclProxyInit`, called from `bootstrapInit` and `bootstrapSplit`; a split child that shares resources takes the parent's `proxyState` in `initTransportsRank` |
| topology (`localRank`, `rankToNode`, `localRanks`, `nNodes`, `clique.size`) | `initTransportsRank` |
| `newCollTrace`, `algoStats` | `newCollTraceInit` (`algoStats` only when `NCCL_COLLTRACE` includes `algostat` or `ALL`) |
| `ctranComm_` (for MapperTrace) | `createCtranComm`, only when CTRAN is enabled for the comm |

Without CTRAN the snapshot's MapperTrace is null, which
`NcclCommMonitorInfo::fromNcclComm` and `getMapperTrace` both handle.

**Deregister**, `ncclx::comms_monitor::CommsMonitor::deregisterComm(comm);` in
the three user-facing teardown APIs, none of which calls another (Finalize
enqueues `commDestroySync`, Destroy and Abort enqueue `commReclaim`). The calls
deliberately do *not* go in those async workers, which may run on another
thread after a non-blocking API call has returned (v2_30 always uses
`ncclAsyncLaunch`; v2_32 uses `ncclMgmtTaskEnqueue` or `ncclAsyncLaunch` per
`ncclParamEnqueueRearchEnable()`): the registry wants the DEAD transition at the
API boundary. `deregisterCommImpl` is idempotent, so Finalize-then-Destroy is
safe. Placement is *not* uniform, and is the same on both versions:

| Site | Position | Anchor |
| --- | --- | --- |
| `ncclCommFinalize` | before the null guard | after the local declarations, before `NCCLCHECK(ncclGroupStartInternal()); if (comm == NULL) goto exit;` |
| `ncclCommDestroy` | after the null guard | after the `if (comm == NULL) { ...; return ncclSuccess; }` early return, before `int rank = comm->rank, nranks = ...` |
| `ncclCommAbort` | after the null guard | after the "Abort START" log, before `NCCLCHECK(ncclGroupStartInternal())` and `setCommAbortFlags(comm, 1)` |

The Finalize placement is kept for parity despite *Known issues* 2.
`ncclCommRevoke` has no deregister on either version (*Known issues* 5). On
v2_30, Scuba event helpers add lines around these anchors, and "Abort START"
is `commAbortLog` plus `abortEvent.lapAndRecord` rather than `INFO`. v2_30's
`ncclCommAbort` also has the force-abort block (`NCCL_COMM_ABORT_SCOPE`) ahead
of its null guard; scopes `none` (return) and `job` (`exit(1)`) leave before the
deregister, so the comm stays ALIVE. v2_32 has the same block, and the
deregister sits after it there too.

**Why in baseline**: the registry must observe every communicator, and only
`init.cc` knows when one becomes fully formed or starts tearing down. There is
no upstream hook, and the snapshot has to happen at a point where the whole
NCCLX state block is populated.

### 2. `src/nccl.h.in` — public API

Declared after `} // end extern "C"`, with C++ linkage and a defaulted argument
that `commDump.cc` matches:

```c
#define NCCL_COMM_DUMP
#define NCCL_COMM_DUMP_ALL

ncclResult_t  ncclCommDump(ncclComm_t comm, std::unordered_map<std::string, std::string>& map);

ncclResult_t ncclCommDumpAll(std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& map,
    const std::unordered_map<std::string, std::string>& hints = {});
```

The doc comments above them (including the `comm_dump::requestFields` /
`comm_dump::flush` hints) are identical on both versions. Placement differs:

- **v2_32**: the C++ tail already exists, opened by the
  `// [META] CollTrace lifecycle feed and algorithm-statistics dump` comment and
  its std includes. The block goes after those includes and before
  `namespace ncclx::colltrace {`, and adds no includes.
- **v2_30**: the block opens the C++ tail with its own `<optional>`, `<string>`,
  `<unordered_map>` and `<vector>` includes.

Both versions also define `NCCL_COMM_DUMP` a second time in the feature-macro
list near the top of the file, outside any `__cplusplus` guard (v2_32: between
`NCCL_COMM_SPLIT_GROUP_RANKS_SUPPORTED` and
`NCCL_COLLTRACE_CUDA_GRAPH_COMPATIBLE`).

**Why in baseline**: out-of-tree consumers only ever see the installed
`include/nccl.h`; the feature macros are the NCCLX capability-probe idiom.

### 3. `def_build.bzl` — build wiring

Sources, alphabetically around `"meta/commHash.cc"`:

```python
    "meta/commDump.cc",
    "meta/comms-monitor/*.cc",
```

Dep, at the end of the deps list with a one-line comment:

```python
    # meta/comms-monitor/CommsMonitor.h (folly::Synchronized registry).
    "fbcode//folly:synchronized",
```

The deps the sources use directly (`comms/utils/memtrace:memory_trace`,
`folly:map_util`, `folly/json:dynamic`) are already named explicitly in the
v2_32 target, for allocation tracing, the CTRAN integration and the tuner. The
row adds `folly:synchronized`, which v2_30 also names, for `CommsMonitor.h`'s
`folly::Synchronized` registry. The sources also need
`comms/utils:str_utils` and `comms/utils/colltrace:network_perf_monitor`, which
v2_32's ProxyTrace row already lists. The private-header globs already exist.

**Conda build**: both versions' `src/Makefile` compile the two sources from
`${SHAREDDIR}/meta/`, v2_30 through globs and v2_32 through its `meta/`
allowlist, which mirrors `def_build.bzl`.

## Interaction with CollTrace and ProxyTrace

Registry entries hold `shared_ptr`s to the trace objects and are never erased
(*Known issues* 1). So with CommsMonitor enabled, neither
`newCollTraceDestroy`'s `comm->newCollTrace.reset()` nor `proxyTraceDestroy`'s
`proxyState->trace.reset()` drops the last reference: `~CollTrace`, and with it
the poll-thread join, runs only when the registry is destroyed or when a new
comm registers at the same address (on the registering thread, after the
registry lock is released). This is
pre-existing v2_30 behaviour; on v2_32 the row rewords the comment above
`newCollTraceDestroy` in `commDestroySync` to say so. A hazard follows from
code reading (not observed): the CollTrace watchdog's `funcIfError` lambda
(`CollTraceWrapper.cc`) captures the raw `ncclComm*` and calls
`ncclCommGetAsyncError` on it, so a retained CollTrace processing an in-flight
event after `commFree` would read freed memory.

## v2_32 dump output versus v2_30

- **`MT_*` keys** appear only for comms with CTRAN enabled (`ctranComm_` set).
- **`memory` counters** come from the MemoryTrace allocation/free hooks and
  communicator attribution (`recordAlloc` / `recordFree`, `memLogMetaData`
  setters), which are ported separately from this row and are in the v2_32
  build, so enabled memtrace populates the snapshot's tracker.
- **`PT_*` p2p records differ.** For the same 4-GPU send/recv test over NET
  (one `ncclSend` + one `ncclRecv` per group), v2_32's `PT_pastColls` has
  `opCount` 2, 4, 6, ... (one per API call) with `nProxyOps` 2; v2_30 has
  1, 2, 3, ... with `nProxyOps` 8 (4 channels). Do not assume consecutive IDs.

## Known issues — present in v2_30 (one fixed by the follow-up, as marked)

Recorded so they are not mistaken for port regressions. All live in shared code
or in hooks placed identically on both versions.

1. **Registry entries are never erased.** `deregisterCommImpl` only sets
   `status = DEAD`; the in-code comment says this is intentional (dead comms
   stay dumpable) and the comms-monitor tests assert it. Consequences:
   - `commsMap_` grows for the process lifetime, pinning every trace object,
     including a running CollTrace poll thread per destroyed comm.
   - (Fixed for both versions.) Entries are keyed on the comm pointer, so a
     new `ncclComm` can reuse a destroyed comm's address (it does
     deterministically in the v2_32 test suites). `registerCommImpl` therefore
     replaces the stale entry: with `emplace` the new comm was never registered
     and `ncclCommDump` / `commDumpAll` returned the dead comm's snapshot. The
     stale entry is swapped out under the registry lock and destroyed after it
     is released, since it may hold the last reference to the dead comm's
     CollTrace. The dead comm's own entry is lost when its address is reused.
   - `commDumpAllImpl` has no `status == ALIVE` filter, so `ncclCommDumpAll`
     includes dead communicators.
2. **`deregisterComm` runs before the null guard in `ncclCommFinalize`.** A
   NULL comm (monitor enabled) hits `ERR(ncclInternalError, "Deregistering
   comm %p that is not registered")`: an error log, a Scuba error record by
   default, and an overwritten `ncclGetLastError()`. The same fires on Abort or
   Destroy after `ncclCommInitRankFunc` failed before `registerComm`, so a later
   `ncclGetLastError()` hides the init failure's root cause.
3. **Dead code.** The `CommLogData*` overload of `dumpCommInfo` null-checks the
   address of a by-value member, and `getNumOfCommMonitoring`'s "lock timed
   out" path is unreachable with the untimed `rlock()`.
4. **`waitForCollTraceDrain` polls expensively.** The test-only helper calls
   `dumpNewCollTrace` every 50 ms with an empty `DumpFieldSet`, serialising
   every `CT_*` field just to compare two strings against `"[]"`.
5. **`ncclCommRevoke` never deregisters**: a revoked comm stays ALIVE in
   `ncclCommDumpAll` until Destroy or Abort. Whether intended is unresolved.

## Test parity

This row removes `"2.32"` from `excluded_versions` of the rules that could not
build without its symbols. On v2_32:

| Rule | Needs | Result |
| --- | --- | --- |
| `meta/comms-monitor/tests:comms_monitor_dist_nolocal` | `CommsMonitor`, `commDumpAll` | 9/9 |
| `meta/colltrace/tests:new_colltrace_ut` | `dumpNewCollTrace` | passes |
| `meta/colltrace/tests:dump_new_colltrace_ut` | `dumpNewCollTrace` | passes |
| `meta/colltrace/tests:baseline_cudagraph_colltrace_dist` | `waitForCollTraceDrain`, `dumpNewCollTrace` | 2/2; in-kernel graph timestamps verified end to end |

`comms_monitor_ut` (which now clears the registry per case and covers the
stale-entry replacement) and `meta/tests:comm_dump_test` are enabled on v2_32.

It also fixes `testOneCommDump` and `testMultipleCommDump` in shared
`CommsMonitorDist.cc`, which failed identically on v2_30: with no hints the
result always has a `"GlobalInfo"` entry beside the commHash keys, but the cases
expected exactly 1 and 4 entries. Both now `erase("GlobalInfo")` first.

`new_colltrace_dist_local`, `new_colltrace_dist_nolocal` and
`dump_algo_stat_ctran_test` are enabled as well; they also need CTRAN, and
`new_colltrace_dist_nolocal` the v2_32 RMA window API (`ncclWinAllocate`,
`ncclWinSharedQuery`).

Still excluded on v2_32 although they use this row's symbols:

- `meta/tests:alltoall_test` / `alltoallv_test` are Ctran tests;
  `alltoallv_test` also needs `ncclAllToAllv`, not yet declared on v2_32.

## Re-check at the next rebase

- `registerComm` still follows everything in the snapshot table, in particular
  if upstream moves `ncclProxyInit` out of `bootstrapInit` / `bootstrapSplit` or
  reorders `ncclCommInitRankFunc`. `createCtranComm` stays between
  `newCollTraceInit` and `registerComm`, and the Abort deregister stays after
  the force-abort block.
- No new teardown API. Today only Finalize, Destroy and Abort tear down: Split,
  Shrink and Grow only create, Revoke does not free the comm, and
  `ncclCommSuspend` / `ncclCommResume` carry no hooks.
- The `nccl.h.in` declarations stay outside `extern "C"`; the `def_build.bzl`
  entries and the `src/Makefile` allowlist entries stay in sync.

## Related

- `colltrace_baseline.md` — CollTrace, whose handle CommsMonitor snapshots.
- `proxytrace_baseline.md` — ProxyTrace, likewise.
