# PAT AVG Design

For the PAT algorithm details (phases, data flow, buffer addressing), see [ReduceScatterPat.md](ReduceScatterPat.md).

## Original AVG Limitation

The original NCCL AVG implementation (`FuncSumPostDiv`) has a critical limitation:

**Only supports unsigned integer types** (uint8, uint32, uint64)

This is because `FuncSumPostDiv` uses integer division which:
- Truncates results for signed integers (incorrect for negative values)
- Cannot represent fractional results for floating-point types
- Is fundamentally incompatible with float, half, bfloat16, and fp8 types

## Overview

FuncPatAvg provides native average (division) support for the PAT (Partition Aggregation Tree) algorithm. Unlike `FuncSumPostDiv` which only supports unsigned integers, `FuncPatAvg` supports all data types including float, half, bfloat16, and fp8.

## Key Design: Two-Phase Approach

Reduction is pure sum; division is applied as postOp on final write only.

## 1. Apply_Reduce / Apply_PostOp Traits

The kernel uses trait classes to customize behavior per reduction function:

```cpp
// Kernel calls generic helpers (common_kernel.h):
acc = applyReduce(redFn, acc, tmp);  // -> Apply_Reduce<Fn>::reduce()
acc = applyPostOp(redFn, acc);       // -> Apply_PostOp<Fn>::postOp()

// FuncPatAvg specializations (meta/device/FuncPatAvg.cuh):
Apply_Reduce<FuncPatAvg<T>>  -> delegates to FuncSum (pure addition)
Apply_PostOp<FuncPatAvg<T>>  -> fn.divide(x) (divide by nRanks)
```

## 2. Host-Side Dispatch

```cpp
// enqueue.cc:1873 - after algorithm selection
ncclx::maybeEnablePatAvg(info, comm->nRanks);

// PatAvgAlgoHelper.h
void maybeEnablePatAvg(ncclTaskColl* info, int nRanks) {
  if (info->algorithm == NCCL_ALGO_PAT &&              // PAT selected
      info->func == ncclFuncReduceScatter &&           // ReduceScatter
      (info->opDev.op == ncclDevSumPostDiv ||          // AVG operation
       info->opDev.op == ncclDevPreMulSum) &&
      isPatAvgEnabled()) {                             // NCCL_REDUCESCATTER_PAT_AVG_ENABLE=1
    info->opDev.op = ncclDevPatAvg;                    // Switch to PatAvg
    info->opDev.scalarArg = nRanks;                    // Pass nRanks for division
  }
}
```

## 3. Kernel-Side Division at Final Write Step

Large messages are split into multiple chunks, each processed independently through the PAT algorithm. The key challenge is: **when should division be applied?**

- Division must happen exactly once per output element
- Each chunk goes through multiple phases (0-4)
- Multiple local writes occur during processing (Phase 1 intermediate writes)
- Only the final write for each chunk should trigger division

### Solution: isFinalWrite Flag

Added explicit `isFinalWrite` flag to `ncclPatStep` struct:

```cpp
struct ncclPatStep {
  // ... other fields ...
  bool isFinalWrite;  // True if final write for a chunk
};
```

The flag is set only in Phase 4, which is the final write phase for each chunk. See phase explanation in [ReduceScatterPat.md](ReduceScatterPat.md).

```cpp
// PatRSAlgorithm::getNextOp(), Phase 4:
} else if (phase == 4) {
  ps->recvDim = 0;
  ps->sendDim = -1;
  ps->isFinalWrite = true;  // Division applied here
  offset += chunkCount;     // Move to next chunk
}
```

PostOp application uses this flag directly:

```cpp
// prims_simple.h patReduce():
const int applyPostOp = ps->isFinalWrite;
```

### Write Types During PAT Execution

| Write Type | Phase | sendDim | isFinalWrite | PostOp Applied |
|------------|-------|---------|--------------|----------------|
| Send to peer | 0-3 | >= 0 | false | No |
| Intermediate local write | 1 | -1 | false | No (partial sum) |
| Final chunk write | 4 | -1 | true | Yes (divide by nRanks) |

Phase 4 is the ONLY phase where all contributions have been accumulated, making it the correct place to apply division.

## Key Files

| File | Purpose |
|------|---------|
| `meta/device/FuncPatAvg.cuh` | FuncPatAvg definition, Apply_Reduce/Apply_PostOp traits |
| `meta/collectives/PatAvgAlgoHelper.h` | Host-side dispatch logic |
| `src/include/collectives.h` | PatRSAlgorithm with isFinalWrite flag in ncclPatStep |
| `src/device/prims_simple.h` | patReduce() applies postOp based on isFinalWrite |
| `src/device/reduce_kernel.h` | Base trait definitions, applyReduce/applyPostOp helpers |
| `src/device/common_kernel.h` | reduceCopyPacks kernel that calls the traits |
| `src/enqueue.cc` | Calls maybeEnablePatAvg after algorithm selection |

## Enabling PAT AVG

Set environment variables:
```bash
export NCCL_ALGO="reducescatter:pat"
export NCCL_REDUCESCATTER_PAT_AVG_ENABLE=1
```

The `isPatAvgEnabled()` function reads the `NCCL_REDUCESCATTER_PAT_AVG_ENABLE` CVAR
to determine if native PAT AVG support should be used. This is a clean, dedicated
control that works properly with the standard NCCL_ALGO algorithm parser.
