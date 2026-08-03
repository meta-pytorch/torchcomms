# In-kernel colltrace emit for baseline collectives: Baseline Modifications

## Background

The colltrace graph watchdog (and graph-mode colltrace tracing generally) times
graph-captured collectives by reading per-collective `kStart`/`kEnd` timestamps
that the collective **kernel** publishes into a shared HRDW ring at replay. Only
ctran GPE kernels emitted these timestamps (via
`ctran::device::ColltraceEventScope`); baseline NCCL collectives (the `orig`
algorithm path) did not, so a graph-captured baseline collective registered a
`collId` on the host (`recordGraphCollectiveImpl`) but the ring stayed empty and
the watchdog had nothing to observe.

The host-side colltrace machinery is already backend-agnostic:

- Registration + `collId` assignment for a graph-captured collective happens in
  `collTraceBaselineGetHandle` → `getHandleFromNcclKernelPlan` →
  `CollTrace::recordGraphCollectiveImpl` — shared with ctran.
- `CollTrace::pollGraphEvents` reads `kStart`/`kEnd` from the ring keyed on
  `collId`, agnostic to which kernel wrote them.
- `ICollTraceHandle::getColltraceDeviceHandle()` already returns the armed,
  ring-backed `ColltraceDeviceHandle` (ring + `collId`) on the graph path and an
  unarmed `{}` otherwise.

So the only missing piece for baseline was the **device-side write** plus arming
the per-launch handle onto the kernel args. The device writer itself is shared:
the generic `meta::comms::colltrace::ColltraceEventScope` lives in
`comms/utils/colltrace/ColltraceEventScope.cuh` (ctran's
`ctran::device::ColltraceEventScope` is a thin subclass adding a `KernelFlagDev`
overload).

## Versions Affected

v2.29, v2.30

## Baseline Files Modified

Three baseline files per version, each a tagged, minimal hook. All logic lives in
the Meta helper `ncclx::colltrace::armNcclInKernelColltrace`
(`meta/colltrace/CollTraceWrapper.{h,cc}`), which is version-agnostic.

1. `src/include/device.h` — one field on the per-launch arg struct plus its
   include:
   ```cpp
   // [META] In-kernel colltrace gate, defined here and used by every
   // colltraceHdr reference so the struct has one layout across all TUs.
   #if !defined(NCCLX_NO_INKERNEL_COLLTRACE) && \
       !(CUDART_VERSION >= 13000 && CUDART_VERSION < 13030)
   #define NCCLX_INKERNEL_COLLTRACE 1
   #endif

   #ifdef NCCLX_INKERNEL_COLLTRACE
   #include "comms/utils/colltrace/ColltraceDeviceHandle.h"
   #endif
   ...
   struct alignas(16) ncclDevKernelArgs {
     ...
     void* workBuf;
     // [META] ... armed host-side in armNcclInKernelColltrace() ...
   #ifdef NCCLX_INKERNEL_COLLTRACE
     meta::comms::colltrace::ColltraceDeviceHandle colltraceHdr;
   #endif
     // struct ncclDevWorkBatch batches[];  // trailing inline region
   };
   ```
   The handle is the **last fixed field**, so the trailing inline work-batch /
   work region (located at `kernelArgs+1` and via `batch.offsetBase`, both
   `sizeof(ncclDevKernelArgs)`-relative on host and device) stays consistent.
   `ColltraceDeviceHandle` is a trivially-copyable aggregate, so the word-by-word
   arg→shmem copy in `ncclKernelMain` carries it correctly.

2. `src/device/common.h` — one RAII scope in the single generic kernel
   `ncclKernelMain`, right after the prologue publishes `ncclShmem`:
   ```cpp
   // [META] in-kernel colltrace ...
   #ifdef NCCLX_INKERNEL_COLLTRACE
   meta::comms::colltrace::ColltraceDeviceEventScope colltraceScope(
       ncclShmem.args.colltraceHdr);
   #endif
   ```
   ctor emits `kStart`; dtor emits `kEnd` at kernel exit. One scope covers
   **every** baseline collective, since they all funnel through `ncclKernelMain`.
   An unarmed handle makes it a no-op (single-writer election is block0/thread0).

3. `src/enqueue.cc` — one tagged call in `ncclLaunchKernel`, next to the existing
   `collTraceBaselineGetHandle` injection:
   ```cpp
   ncclx::colltrace::armNcclInKernelColltrace(
       plan, colltraceHandle.get(), comm->compCap);
   ```
   This writes the ring/`collId` into `plan->kernelArgs->colltraceHdr` just
   before `cuLaunchKernelEx`, so the `collId` assigned at capture is baked into
   the graph node and re-emitted on every replay. The helper is a no-op unless a
   graph is being captured (`getColltraceDeviceHandle().valid()`) on an sm_90+
   device (the ring's 128b atomic requirement); symmetric-memory kernels use a
   different arg layout and are skipped.

## Platform gating: CUDA only

`device.h` defines a single gate macro, `NCCLX_INKERNEL_COLLTRACE`, and all
three touch points test it with `#ifdef`. It is left undefined when
`NCCLX_NO_INKERNEL_COLLTRACE` is set, which `device_object`'s
`propagated_pp_flags` does on `ovr_config//gpu:amd`
(`comms/ncclx/nccl_build_config.bzl`). The same `select` also drops
`colltrace_device_handle` from that target's `exported_deps`, so on AMD the
header is neither included nor on the include path.

It is also left undefined on CUDA 13.0-13.2, where an nvcc
`[[no_unique_address]]` device-layout bug (fixed in 13.3) misaligns the
`args->__shared__` copy of `ncclDevKernelArgs`; any kernel launch carrying those
args then faults with `cudaErrorMisalignedAddress` (seen as "Failed to allocate
barrier buffer: misaligned address" in the torchcomms integration tests). The
guard lives in the same commit that adds the field, so no point in the stack
builds a `colltraceHdr` that the toolchain will mislay.

Two reasons for the AMD case:

- **It cannot work on AMD.** The ring publishes a slot with `atom.exch.b128`
  inline PTX, which has no HIP equivalent —
  `HrdwRingBufferWriter::write` compiles to a trap under `__HIPCC__`, and
  `comms/utils/hrdw_ring_buffer/BUCK` already states AMD "will fail at
  compile/link time when actually exercised".
- **It breaks the AMD build if left in.** `colltrace_device_handle` reaches
  `//comms/utils:hrdw_ring_buffer`, a `gpu_cpp_library`, so on AMD hipify
  rewrites that target's `<cuda_runtime.h>` into `<hip/hip_runtime.h>`. ncclx is
  never hipified — under `mode/*-amd-gpu` it still compiles with `nvcc` — so its
  `nccl.h` keeps the real CUDA runtime. Any consumer including `collectives.h`
  then gets CUDA's and HIP's vector types in one TU and fails with ~20 `typedef
  redefinition` errors (first seen in `comms/utils/tests:comm_specs_test`).
  Before this change ncclx's public headers never reached a hipified header: the
  handle was only ever a `shared_ptr<ICollTraceHandle>` used inside `.cc` files,
  and a pointer to an abstract type needs no layout. Embedding the handle
  by value is what put it on the exported header path.

The flag is propagated from the same target and the same `select` as the dep so
no consumer can disagree about whether `colltraceHdr` exists — a mismatch would
silently change `ncclDevKernelArgs`'s layout between TUs.

## Why in baseline

The three touch points are structurally forced:

- The `collId` is per-collective and must be baked into each graph node's arg
  bytes at capture, so it must live in the per-launch `ncclDevKernelArgs` (a
  comm-level slot would be overwritten by the next captured collective) — hence
  the `device.h` field.
- The emit must run inside the collective kernel, and `ncclKernelMain` is the one
  generic entry every baseline collective shares — hence the `common.h` line.
- The arm must happen at the launch site where `plan->kernelArgs` is baked into
  the graph node — hence the `enqueue.cc` call, placed beside the existing
  colltrace injection.

All heavy logic is in the Meta helper; the baseline edits are a field, a scope,
and a call, each `[META]`-tagged. No `CollTrace.cc` / `CollTraceWrapper` handle
or poll changes were needed — baseline reuses the ctran path end to end.
