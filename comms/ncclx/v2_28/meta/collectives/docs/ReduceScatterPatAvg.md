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
