# ncclReduceScatterQuantize for v2_29

## Overview

Port `ncclReduceScatterQuantize` from ncclx v2_27 to v2_29, using the v2_29 `ncclInfoExt` mechanism for algorithm/protocol selection instead of v2_27's ad-hoc cost table filtering.

## Background

In v2_27, the collective was added by:
- Adding `randomSeed` and `transportType` fields directly to the `ncclInfo` struct
- Filtering `updateCollCostTable` to force PAT algorithm + SIMPLE protocol
- Adding `quantizeRandomSeedPtr` to `ncclTaskColl` and `ncclDevWorkColl`
- Adding a float-specialized `RunWorkColl` template in `reduce_scatter.h`

The v2_27 code has TODO comments: *"Switch to Min's infoExt approach"* (enqueue.cc lines 1823, 1838).

## v2_29 Approach: InfoExt

v2_29 introduces `ncclInfoExt` (`meta/algoconf/InfoExt.h`), a per-collective override that bypasses the cost table entirely. The canonical example is PAT AVG in `PatAvgHelper.h`.

### Key Design Decisions

1. **Quantize fields in InfoExt**: `quantizeRandomSeedPtr` and `transportType` are added to `ncclInfoExt` (with defaults so existing callers are unaffected).
2. **No cost table modifications**: InfoExt bypasses `updateCollCostTable` via `infoExtOverride()` (already wired in enqueue.cc lines 451-456).
3. **Device-side branching**: `quantizeRandomSeedPtr` is still needed in `ncclDevWorkColl` for the kernel to branch between quantized and non-quantized paths.
4. **Two separate functions in quantCollectives.cc**: The API entry point uses `#if`/`#else` to compile either the InfoExt-based implementation (v2_29+) or the legacy implementation (v2_27). This is cleaner than version-guarding inside a single function.

### Data Flow

```
API call (quantCollectives.cc)
  -> ncclReduceScatterQuantizeInfoExt() [v2_29+]
     -> validates args
     -> creates ncclInfo with info.ext = setupQuantizeInfoExt(...)
     -> ncclEnqueueCheck(&info)
        -> copies info.ext to task.ext           (enqueue.cc:2672)
        -> ncclPrepareTasks:
           -> infoExtOverride() sets algo=PAT, proto=SIMPLE  (enqueue.cc:451-453)
        -> ncclTasksRegAndEnqueue:
           -> copies task.ext->quantizeRandomSeedPtr to devWork.quantizeRandomSeedPtr
        -> Device kernel: reduce_scatter.h
           -> float specialization checks work->quantizeRandomSeedPtr != nullptr
           -> branches to PrimitivesQuantized (quantized) or Primitives (regular)
```

## Files Modified

| File | Action | Description |
|------|--------|-------------|
| `meta/algoconf/InfoExt.h` | Modify | Add `quantizeRandomSeedPtr` and `transportType` fields |
| `v2_29/src/nccl.h.in` | Modify | Add `NCCL_REDUCE_SCATTER_QUANTIZE_SUPPORTED` |
| `v2_29/src/include/device.h` | Modify | Add `quantizeRandomSeedPtr` to `ncclDevWorkColl` |
| `v2_29/src/enqueue.cc` | Modify | Copy seed ptr from task.ext to devWork |
| `meta/collectives/quantCollectives.cc` | Modify | Two functions: InfoExt (v2_29+) vs legacy (v2_27), with `#if`/`#else` guard |
| `meta/collectives/QuantizeHelper.h` | **New** | Helper to construct quantize InfoExt (analogous to `PatAvgHelper.h`) |
| `v2_29/src/device/reduce_scatter.h` | Modify | Add float-specialized `RunWorkColl` with quantized branch |
| `meta/collectives/kernels/prims_quantize.cuh` | Modify | Version guard for shmem `redOpArgs` layout change |

## Existing Code Reused

- `computePatAvgChannelsAndWarps()` from `meta/collectives/PatAvgHelper.h`
- `setupPatAvgInfoExt()` pattern from `PatAvgHelper.h`
- `infoExtOverride()` from `meta/algoconf/InfoExtOverride.h` (no changes needed)
- `PrimitivesQuantized` from `meta/collectives/kernels/prims_quantize.cuh`
- `reduceCopySR` from `meta/collectives/kernels/reduce_copy_sr_v2.cuh`

## shmem Layout Difference (v2_27 vs v2_29)

- v2_27: `ncclShmem.redOpArgs[0]` (top-level array in `ncclShmemData`)
- v2_29: `ncclShmem.groups[group].redOpArgs` (scalar in per-group `ncclShmemGroup`)

This affects `prims_quantize.cuh` line 103, which needs a version guard.
