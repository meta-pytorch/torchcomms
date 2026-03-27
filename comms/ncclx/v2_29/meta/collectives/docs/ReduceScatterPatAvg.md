# PAT AVG Design

For PAT algorithm details (phases, data flow, buffer addressing), see [ReduceScatterPat.md](ReduceScatterPat.md).

## Why a Separate Device Op Instead of Extending FuncSumPostDiv?

FuncSumPostDiv uses `__umulhi` integer reciprocal multiplication for its `divide()` function, which has no floating-point equivalent — supporting floats would require a complete rewrite, not an extension. Additionally, extending FuncSumPostDiv to float types would generate ~200 additional kernel instantiations across all collectives, algorithms, and protocols. FuncPatSumPostDiv is restricted to ReduceScatter+PAT only (~18 kernels), keeping binary size minimal.

## SumPostDiv vs. PreMulSum

- **SumPostDiv (FuncPatSumPostDiv)**: Sum all contributions first, divide by N once on final write. Single rounding step gives better precision. Requires sufficient exponent range to avoid overflow during intermediate accumulation.
- **PreMulSum**: Multiply each contribution by 1/N before summing. Lower overflow risk since values are pre-scaled. Introduces rounding error at every rank's contribution.

Current PAT AVG implementation:
- FuncPatSumPostDiv: supports bf16, f32, f64, and integer types (signed and unsigned)
- TODO: FuncPatPreMulSum for fp16 and fp8 types that lack exponent range for safe intermediate accumulation
- Unsupported types (fp16, fp8) currently fall back to standard non-PAT algorithm selection

## Two-Phase Reduction

Apply_Reduce dispatches to FuncSum for pure addition. Apply_PostOp applies division by nRanks. The reduction accumulates an exact sum across all ranks, and the single final division preserves maximum precision.

## Host-Side Dispatch

When `comm->usePatAvg_` is true and op is ncclAvg, `setupPatAvgInfoExt()` configures ncclInfoExt to force PAT algorithm with ncclDevPatSumPostDiv. The scalarArg encodes the divisor (nRanks) and a signed-type flag using the same `(divisor << 1 | isSigned)` encoding as FuncSumPostDiv.

## Kernel-Side Division at Final Write Step

The `isFinalWrite` flag in `ncclPatStep` controls when postOp (division) is applied. Only Phase 4 sets `isFinalWrite=true`, ensuring division happens exactly once per output element after all contributions have been accumulated.

| Write Type | Phase | sendDim | isFinalWrite | PostOp Applied |
|------------|-------|---------|--------------|----------------|
| Send to peer | 0-3 | >= 0 | false | No |
| Intermediate local write | 1 | -1 | false | No (partial sum) |
| Final chunk write | 4 | -1 | true | Yes (divide by nRanks) |

## Enabling PAT AVG

PAT AVG can be enabled via two methods:

### Method 1: CVAR (Recommended for benchmarks/testing)

Set the environment variable before running:

```bash
NCCL_REDUCESCATTER_PAT_AVG_ENABLE=1 ./your_benchmark
```

Or in code before communicator creation:

```cpp
setenv("NCCL_REDUCESCATTER_PAT_AVG_ENABLE", "1", 1);
ncclComm_t comm = createNcclComm(...);  // usePatAvg_ = true
```

### Method 2: Global Hint (Recommended for per-communicator control)

Set the global hint before communicator creation:

```cpp
#include "meta/hints/GlobalHints.h"

// Enable PAT AVG for subsequent communicators
ncclx::setGlobalHint(
    std::string(ncclx::HintKeys::kCommAlgoReduceScatter), "avg:patavg");

ncclComm_t comm = createNcclComm(...);  // usePatAvg_ = true

// Reset hint if needed
ncclx::resetGlobalHint(
    std::string(ncclx::HintKeys::kCommAlgoReduceScatter));
```

The hint value format is `<redop>:<algo>`:
- `avg:patavg` - Enable PAT algorithm for AVG reduction operation

### Precedence

CVAR takes precedence over global hint:
1. If `NCCL_REDUCESCATTER_PAT_AVG_ENABLE=1`, PAT AVG is enabled regardless of hint
2. If CVAR is not set or false, global hint `avg:patavg` enables PAT AVG
3. If neither is set, PAT AVG is disabled

### Implementation Details

At communicator creation time (`init.cc`), `usePatAvg_` is set:

```cpp
comm->usePatAvg_ = ncclx::commUsePatAvg();  // Checks CVAR and hint
```

The helper function in `BaselineConfig.h`:

```cpp
inline const bool commUsePatAvg() {
  if (NCCL_REDUCESCATTER_PAT_AVG_ENABLE) {
    return true;
  }
  const auto algoHint = getGlobalHint(HintKeys::kCommAlgoReduceScatter);
  return algoHint.has_value() && algoHint.value() == "avg:patavg";
}
```
