# In-kernel colltrace emit for baseline collectives: Baseline Modifications

## Background

The colltrace graph watchdog (and graph-mode colltrace generally) times
graph-captured collectives from per-collective `kStart`/`kEnd` timestamps that
the collective **kernel** publishes into a shared HRDW ring at replay. Only
ctran GPE kernels emitted them (`ctran::device::ColltraceEventScope`), so a
graph-captured baseline (`orig` path) collective registered a `collId` on the
host but the ring stayed empty and the watchdog had nothing to observe.

The host side is already backend-agnostic: `getHandleFromNcclKernelPlan` →
`CollTrace::recordCollective` (the call ctran also makes) assigns the graph
`collId`, `CollTrace::pollGraphEvents` reads the ring whichever kernel wrote it,
and `ICollTraceHandle::getColltraceDeviceHandle()` returns a ring-backed
`ColltraceDeviceHandle` on the graph path and an unarmed `{}` otherwise. Only
the **device-side write** and arming the per-launch handle were missing; no
`CollTrace.cc` handle or poll changes were needed. The writer is the shared
`meta::comms::colltrace::ColltraceDeviceEventScope`
(`comms/utils/colltrace/ColltraceDeviceEventScope.cuh`), which ctran's
`ColltraceEventScope` subclasses.

## Versions Affected

v2.30, v2.32

## Baseline Files Modified

Two baseline files per version, each a tagged, minimal hook; the arming rides on
the existing CollTrace launch hook (item 3). All logic lives in the
version-agnostic `meta/colltrace/CollTraceWrapper.cc`. The hunks are the same in
v2.30 and v2.32 apart from clang-format wrapping.

1. `src/include/device.h` — the gate and its include right after
   `#include "bitops.h"`, and one field at the end of the fixed part of
   `struct ncclDevKernelArgs`, after `void* workBuf;`:
   ```cpp
   #if !defined(NCCLX_NO_INKERNEL_COLLTRACE)
   #define NCCLX_INKERNEL_COLLTRACE 1
   #endif
   #ifdef NCCLX_INKERNEL_COLLTRACE
   #include "comms/utils/colltrace/ColltraceDeviceHandle.h"
   #endif
   ...
   struct alignas(16) ncclDevKernelArgs {
     ...
     void* workBuf;
   #ifdef NCCLX_INKERNEL_COLLTRACE
     meta::comms::colltrace::ColltraceDeviceHandle colltraceHdr;
   #endif
     // struct ncclDevWorkBatch batches[];  // trailing inline region
   };
   ```
   The handle is the **last fixed field**, so the trailing inline work-batch
   region stays `sizeof(ncclDevKernelArgs)`-relative on host and device. It is
   trivially copyable, so the word-by-word arg→shmem copy carries it.

2. `src/device/common.h` — the `ColltraceDeviceEventScope.cuh` include (same
   gate) after `#include "network/unpack/unpack_defs.h"`, and one RAII scope in
   `ncclKernelMain` between `__syncthreads(); // publish ncclShmem` (the
   prologue's last barrier) and the `while (ncclShmem.aborted == 0)` loop:
   ```cpp
   #ifdef NCCLX_INKERNEL_COLLTRACE
   meta::comms::colltrace::ColltraceDeviceEventScope colltraceScope(
       ncclShmem.args.colltraceHdr);
   #endif
   ```
   The ctor emits `kStart` and the dtor `kEnd` at kernel exit; block 0, thread 0
   is the single writer. Every baseline collective funnels through
   `ncclKernelMain`, so one scope covers them all; an unarmed handle is a no-op.

3. Arming is **not** a separate hook. `ncclLaunchKernel` already calls
   `ncclx::colltrace::prepareNcclKernelColltrace` right after the `extra[]`
   initializer (`src/enqueue.cc` in v2.30, `src/enqueue/enqueue.cc` in v2.32;
   see `colltrace_baseline.md`); its file-local `armNcclInKernelColltrace`
   writes the ring/`collId` into `plan->kernelArgs->colltraceHdr` before the
   launch, so the `collId` assigned at capture is baked into the graph node and
   re-emitted on every replay. It skips `isSymColl` plans (different arg
   layout) and null `kernelArgs` / handle; otherwise it resets `colltraceHdr`
   to `{}` and arms it (`emitStart` and `emitEnd`) only on sm_90+ (the ring's
   128b atomic) when `getColltraceDeviceHandle().valid()`, i.e. `NCCL_COLLTRACE`
   is set and the plan is graph-captured.

## Shared-code guard: graph-captured symmetric plans

Symmetric kernels are never armed, so a graph colltrace record for one would
never complete and would hang the poll thread on drain. v2.30 frees symmetric
tasks, so the existing "persistent plan with no coll/p2p task → no metadata"
check covered them; upstream 2.32 keeps them on `plan->collTaskQueue`.
`getMetadataFromNcclKernelPlan` (`meta/colltrace/CollTraceWrapper.cc`) therefore
returns no metadata for `plan.persistent && plan.isSymColl`, so the plan gets a
`DummyCollTraceHandle`. Eager plans are unaffected. Unit test:
`CollTraceWrapperUT.PersistentSymmetricPlanReturnsNoMetadata`.

## Platform gating: CUDA only

`NCCLX_INKERNEL_COLLTRACE` is left undefined when `NCCLX_NO_INKERNEL_COLLTRACE`
is set, which `device_object`'s `propagated_pp_flags` does on
`ovr_config//gpu:amd` (`build_device_object` in
`comms/ncclx/nccl_build_config.bzl`, shared by every version dir). A second
`select` on the same condition drops `colltrace_device_handle` from its
`exported_deps`. Flag and dep come from one target and condition, so no consumer
can disagree about whether `colltraceHdr` exists (a mismatch would silently
change `ncclDevKernelArgs`'s layout between TUs). Two reasons for the AMD case:

- **It cannot work on AMD.** The ring publishes a slot with `atom.exch.b128`
  inline PTX, which has no HIP equivalent; `HrdwRingBufferWriter::write`
  compiles to a `printf` + `abort()` stub under `__HIPCC__`.
- **It breaks the AMD build if left in.** `colltrace_device_handle` reaches
  `//comms/utils:hrdw_ring_buffer`, a `gpu_cpp_library` that hipify rewrites to
  `<hip/hip_runtime.h>`. ncclx is never hipified (it still compiles with
  `nvcc`), so any consumer of `collectives.h` gets CUDA's and HIP's vector types
  in one TU and fails with ~20 `typedef redefinition` errors. Embedding the
  handle by value is what put it on ncclx's exported header path.

There is deliberately **no CUDA-version carve-out**. An earlier gate excluded
CUDA 13.0-13.2, where nvcc mis-offsets device fields after a
`[[no_unique_address]]` member and faulted any launch carrying the args. That is
fixed at the source: `HRDWRingBufferDeviceHandle` declares its blocking-only
members without the attribute (`sizeof(BlockingHandleProbe) == 48` assert in
`comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h`), and `ColltraceDeviceHandle`
pins its scalars ahead of the ring (`static_assert`). Do not reintroduce it.

## Why in baseline

- The `collId` is per-collective and must be baked into each graph node's arg
  bytes at capture, so it must live in the per-launch `ncclDevKernelArgs` (a
  comm-level slot would be overwritten by the next captured collective).
- The emit must run inside the collective kernel, and `ncclKernelMain` is the
  one generic entry every baseline collective shares.
- The arm must happen where `plan->kernelArgs` is baked into the graph node; the
  existing `prepareNcclKernelColltrace` call is already there.

## v2.30 vs v2.32

Upstream left `ncclDevKernelArgs` and the `ncclKernelMain` prologue unchanged
(bar reflow), so the hunks were a pure re-anchor. Besides the symmetric guard:

- **`kEnd` timing.** In v2.30 thread 0 of block 0 stamps `kEnd` right after
  `profiler(FINI)`, as soon as it leaves the loop. 2.32 adds progress-counter
  `__syncthreads()` calls (only with `NCCL_RAS_ENABLE=1` and
  `NCCL_PROGRESS_COUNTERS` non-zero), one after the loop, so then `kEnd` waits
  for all of block 0. It is a block-0 timestamp, not grid-wide, in both.
- **New `__global__` entries outside `ncclKernelMain`**, both non-collectives:
  the P2P diagnostic kernels in `diagnostics/device/p2p.cu` (only with
  `NCCL_RUN_DIAGNOSTICS`) and the `ncclProgressCounterCaptureGpuTime` clock
  calibration kernel in `device/common.cu`. Outside it in both versions, and so
  untimed: the symmetric kernels, `oneRankReduce` (`ncclLaunchOneRank`) and the
  Meta `ncclKernel_AllReduceSparseBlock_Unpack`.
- **OSS make build.** The device build needs `INCFLAGS += -I$(BASE_DIR)` (as in
  v2.30 `src/device/Makefile`) for the `comms/utils/colltrace/...` includes; the
  v2.32 Makefiles carry no Meta wiring yet, so a 2.32 make build must add it.
- **Tests.** `baseline_cudagraph_colltrace_dist` passes on 2.32, verifying
  in-kernel graph timestamps end to end. The colltrace watchdog tests stay
  excluded until `ncclx::setGlobalHint` (Hints/Ctran) is ported.

## Re-check at the next rebase

- `colltraceHdr` is still the **last fixed field**. The trailing region is
  `sizeof`-relative on host (`kernelArgs + 1` in `finishPlan` / `uploadWork`)
  and device (`(args + 1)[batchIx]`, `(char*)args + batch.offsetBase`).
- Arg space. The 56-byte handle grows `sizeof(ncclDevKernelArgs)` from 32 to
  96; the 4 KB `ncclMaxKernelArgsSize` still fits 64 channels
  (`ncclDevMaxChannelsForArgsBytes(4096)` = min(64, (4096 − 96) / 16)) and the
  in-args work budget shrinks by 64 bytes. There is no `static_assert`, so
  recompute if the struct or `ncclDevWorkBatch` grows.
- New `__global__` entries that bypass `ncclKernelMain`, and what runs between
  the work loop and kernel exit (where `kEnd` is stamped).
- Whether symmetric tasks still sit on the plan, and whether symmetric kernels
  gain a `colltraceHdr` (then the guard could be relaxed).
- The arming stays inside `ncclLaunchKernel`, not
  `ncclLaunchKernelBefore_NoUncapturedCuda`: `doLaunches` calls that for CE and
  RMA plans too, whose `ceCollArgs` / `rmaArgs` share a union with `kernelArgs`,
  and `armNcclInKernelColltrace` has no `isCeColl` / `isRma` check.
