# Fix TOCTOU Race in ReduceScatterQuantize Type Config

## Bug

ReduceScatterQuantize intermittently produces very large values (~1.25e38) or NaN from tiny inputs (~1e-6). The corruption is non-deterministic: replaying the same input with the same seed sometimes gives correct results.

## Root Cause

A time-of-check-time-of-use (TOCTOU) race condition in `prims_quantize.cuh` between the **source selection** and the **type configuration** for the reduce-copy dispatch.

### Background

ReduceScatterQuantize uses mixed-precision transport: user I/O is FP32, but data is transported as BF16 over the network. At each PAT reduce step, the kernel must decide:

1. **Source selection** (line 349): Is the local source (`srcs[1]`) fresh user input (FP32) or re-accumulated data from the transport buffer (BF16)?
2. **Type configuration** (`determineQuantizedTypeConfig`): What type should `reduceCopySR` cast `srcs[1]` as — `(float*)` or `(nv_bfloat16*)`?

These two decisions must agree. If the source selection sets `srcs[1]` to FP32 user input but the type config says BF16, `reduceCopySR` will interpret FP32 bit patterns as pairs of BF16 values, producing garbage.

### The Race

The source selection runs **before** the first `patBarrier()` and uses the per-thread `step` variable (loaded from `peer->step` at the start of `patReduce`):

```
if (peer->accSize < sendOffset + nelem + (step + stepOffset) * connStepSize) {
    srcs[1] = userInput;   // FP32
} else {
    srcs[1] = dsts[0];     // BF16 transport buffer
}
```

The type configuration ran **after** the `patBarrier()` and re-read `sendPeer->step` and `sendPeer->accSize` directly from shared memory:

```
bool isNewData = sendPeer->accSize < sendOffset + nelem +
    (sendPeer->step + stepOffset) * connStepSize;
config.src1IsAccumType = isNewData;
```

With `parallelFactor > 1`, multiple thread groups process consecutive PAT steps concurrently. Between the source selection and the type configuration call, another group's threads can execute their post-step updates:

- **RolePostSend** (line 406): `peer->step = step += StepPerSlice` — modifies `sendPeer->step`
- **RoleWaitSend** (line 418): `atomicMax(&peer->accSize, ...)` — modifies `sendPeer->accSize`

This can cause the type config to make a **different** new-vs-reaccumulation decision than the source selection, resulting in:

- `srcs[1]` points to FP32 data, but `reduceCopySR` casts it as `(nv_bfloat16*)` — FP32 interpreted as BF16 pairs → very large values
- `srcs[1]` points to BF16 data, but `reduceCopySR` casts it as `(float*)` — BF16 pairs interpreted as FP32 → garbage

### Why This Only Affects RSQ, Not RS

The non-quantized PAT ReduceScatter uses single-type `reduceCopy<..., T, ...>` where all data is always treated as type T (float). Even if the accSize decision is wrong, data is interpreted as the correct type. The worst case is a wrong value, never a wrong type.

RSQ's `quantizedPatReduceCopy` uses a 6-way type dispatch that casts sources and destinations to either `InputType` (float) or `TransportType` (BF16) based on the type config. This introduces a new failure mode — type confusion — that the non-quantized path does not have.

### Triggering Conditions

- `parallelFactor > 1`: requires the `chunkSize *= 2` doubling (unique to RSQ in enqueue.cc) to produce a large enough `aggFactor` via `PatRSAlgorithm`
- Consecutive PAT steps sharing the same `sendDim` (e.g., phase 0 where all steps use `sendDim=0`)
- GPU warp scheduling that causes one group to complete its reduce-copy and post-step updates before another group reaches its type config

## Fix

Instead of re-reading `sendPeer->step` and `sendPeer->accSize` from shared memory after the barrier, derive the type config from the pointer values that the source selection already set:

```cpp
if (ps->sendDim >= 0) {
    // If srcs[1] == dsts[0], source selection chose re-accumulation (BF16).
    // If srcs[1] != dsts[0], source selection chose new data (FP32).
    typeConfig.src1IsAccumType =
        (ncclShmem.groups[group].srcs[1] != ncclShmem.groups[group].dsts[0]);
} else {
    typeConfig.src1IsAccumType = true;  // output path: always FP32
}
```

This is inherently consistent with the source selection because it checks the *result* of that decision (the actual pointer stored in `srcs[1]`) rather than re-evaluating the *condition* (accSize threshold) with potentially stale inputs.

## Files Modified

| File | Change |
|------|--------|
| `meta/collectives/kernels/prims_quantize.cuh` | Replace `determineQuantizedTypeConfig()` call with pointer-equality-based type config derivation |
