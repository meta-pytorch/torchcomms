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

## 2. Host-Side Dispatch (Per-Communicator Control via ncclInfoExt)

When per-communicator PAT AVG is enabled (`comm->usePatAvg_ = true`), the override
is set up at collective entry time using ncclInfoExt, before algorithm selection:

```cpp
// collectives.cc - ncclReduceScatter entry point
if (comm->usePatAvg_ && op == ncclAvg) {
  ncclx::setupPatAvgInfoExt(comm, nBytes, &info.ext);
}

// PatAvgHelper.h
void setupPatAvgInfoExt(ncclComm* comm, size_t nBytes, ncclInfoExt* ext) {
  ext->algorithm = NCCL_ALGO_PAT;
  ext->protocol = NCCL_PROTO_SIMPLE;
  ext->opDev.op = ncclDevPatAvg;
  ext->opDev.scalarArg = comm->nRanks;  // Pass nRanks for division
  ext->opDevSet = true;
  computePatAvgChannelsAndWarps(comm, nBytes, &ext->nMaxChannels, &ext->nWarps);
}
```

This approach has key advantages over the old CVAR-based method:
1. **Per-communicator control**: Each comm can independently enable/disable PAT AVG
2. **Early conversion**: Algorithm and opDev are set before algorithm selection
3. **Bypasses tuning**: When ext is complete, algorithm selection loop is skipped

NOTE: User-defined PreMulSum ops (created via ncclRedOpCreatePreMulSum) are NOT
converted to PatAvg because the check requires `op == ncclAvg` (built-in only).

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

## Enabling PAT AVG

### Per-Communicator Control (Recommended)

Set `comm->usePatAvg_ = true` during communicator creation:

```cpp
// At comm creation time (init.cc)
comm->usePatAvg_ = ncclx::commUsePatAvg();  // From CVAR or hint
```

Or via global hint before comm creation:

```cpp
ncclx::setGlobalHint("ncclx.comm.algo_reducescatter", "pat:avg");
ncclComm_t comm = createNcclComm(...);  // usePatAvg_ = true
ncclx::resetGlobalHint("ncclx.comm.algo_reducescatter");
```
