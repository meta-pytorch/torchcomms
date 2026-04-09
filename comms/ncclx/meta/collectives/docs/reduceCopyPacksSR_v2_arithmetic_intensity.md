# Arithmetic Intensity Analysis: `reduceCopyPacksSR` (V2)

**File:** `comms/ncclx/v2_27/meta/collectives/kernels/reduce_copy_sr_v2.cuh`
**Function:** `reduceCopyPacksSR` (lines 433–483)

## Function Structure

The inner loop processes `Unroll * EltPerPack` elements per thread per iteration:

```
while (iter.hasWork()) {
    1. loadFirstSource       — load source data into float accumulator
    2. ReduceSources::apply  — load + reduce additional sources (FP add)
    3. PhiloxWarpExchange    — generate random bits (INT ALU + warp shuffles)
    4. storeFirstDestinationSR — stochastic round float->bf16 + store
}
```

The dominant hot path (bulk of data) is the first call from `reduceCopySR` (line 519),
with `EltPerPack = 4` (16-byte aligned vector loads/stores), `AccType = float`,
`DstType = __nv_bfloat16`.

---

## Philox-4x32 with 7 Rounds: Instruction Count

Source: `comms/utils/kernels/rng/philox_rng.cuh`

The implementation uses 7 rounds (minimum to pass BigCrush), not the standard 10.

Per round (lines 40–54 of `philox_rng.cuh`):

```
_c0 = c0;  _c2 = c2;                              // register rename, 0 instr
c0 = umulhi32(PHILOX_ROUND_B, _c2) ^ c1 ^ k0;     // 1 UMULHI + 1 LOP3 = 2
c2 = umulhi32(PHILOX_ROUND_A, _c0) ^ c3 ^ k1;     // 1 UMULHI + 1 LOP3 = 2
c1 = PHILOX_ROUND_B * _c2;                         // 1 IMUL
c3 = PHILOX_ROUND_A * _c0;                         // 1 IMUL
k0 += PHILOX_KEY_A;                                // 1 IADD
k1 += PHILOX_KEY_B;                                // 1 IADD
```

- **8 instructions/round x 7 rounds = 56 instructions**
- Setup (seed/offset split, lines 67–74): ~4 instructions
- **Total per `philox_randint4x` call: ~60 INT instructions**

For comparison, the standard 10-round variant would cost ~84 instructions.

---

## Exchange Overhead Per Call

Each exchange variant redistributes 1 Philox call's output across G lanes,
serving **G x EPP = 8 elements**:

### `exchangeEPP4` (G=2, lines 175–196)

- 8 CMOV + 2 SHFL = **10 instructions** -> **1.25/elem**

### `exchangeEPP2` (G=4, lines 198–228)

- 4x `philox_select` for sends/own (3 CMOV each) = 12 CMOV
- 3 SHFL
- 4x `philox_select` for stores = 12 CMOV
- Total: **27 instructions** -> **3.4/elem**

### `exchangeEPP1` (G=8, lines 230–339)

- channel calc: 1 SHR
- own `philox_select`: 3 CMOV
- 7 recv lines (each: 2 index calc + 3 `philox_select` + 1 SHFL): 42
- 8 `delta_select8` (7 CMOV each) + 7 XOR for indices: 63
- Total: **109 instructions** -> **13.6/elem**

### Simple Path (no exchange, `generateSimple`)

Per unroll step: 5 (address calc) + 60 (Philox) + 2 (channel) + 3 (`philox_select`)
+ [4 extra for EPP=4 `rand_b`]

| EPP | Instr/step | Elements/step | **Instr/elem** |
|:---:|:----------:|:-------------:|:--------------:|
| 4   | 74         | 4             | **18.5**       |
| 2   | 70         | 2             | **35.0**       |
| 1   | 70         | 1             | **70.0**       |

---

## Exchange Activation

The exchange path activates when `Unroll >= G` (and warp base is 8-aligned).
Within the exchange path or the simple path, varying Unroll does **not** change
per-element cost — it only determines which path is taken.

| EPP | G | Exchange requires |
|:---:|:-:|:-----------------:|
| 4   | 2 | Unroll >= 2       |
| 2   | 4 | Unroll >= 4       |
| 1   | 8 | Unroll >= 8       |

---

## Per-Element Cost Breakdown

### 1 Source (float -> bf16), 6 bytes/element

Memory: 4B read (float32) + 2B write (bf16) = **6 bytes/element**

| EPP | Path | Unroll req. | RNG+Addr | Exchange | SR (SW) | SR (BW HW) | Misc | **Total (SW)** | **Total (BW)** | **AI (SW)** | **AI (BW)** |
|:---:|:----:|:-----------:|:--------:|:--------:|:-------:|:----------:|:----:|:--------------:|:--------------:|:-----------:|:-----------:|
| 4   | Exch | >= 2        | 8.1      | 1.3      | 4       | 0.5        | 2    | **15.4**       | **11.9**       | **2.6**     | **2.0**     |
| 4   | Simple | 1         | 18.5     | —        | 4       | 0.5        | 2    | **24.5**       | **21.0**       | **4.1**     | **3.5**     |
| 2   | Exch | >= 4        | 8.1      | 3.4      | 4       | 0.5        | 2    | **17.5**       | **14.0**       | **2.9**     | **2.3**     |
| 2   | Simple | 1–3       | 35.0     | —        | 4       | 0.5        | 2    | **41.0**       | **37.5**       | **6.8**     | **6.3**     |
| 1   | Exch | >= 8        | 8.1      | 13.6     | 7       | 3          | 3    | **31.7**       | **27.7**       | **5.3**     | **4.6**     |
| 1   | Simple | 1–7       | 70.0     | —        | 7       | 3          | 3    | **80.0**       | **76.0**       | **13.3**    | **12.7**    |

*AI = INT ops / bytes. SR estimates are approximate.*

### Additional Costs for 2nd Source (per element)

| Component            | 2x fp32       | fp32 + bf16   |
|----------------------|:-------------:|:-------------:|
| Load (memory op)     | 4 bytes       | 2 bytes       |
| bf16->float cast     | —             | ~1 INT op     |
| FP reduction add     | 1 FP op       | 1 FP op       |
| Extra ptr advance    | ~0.5 INT op   | ~0.5 INT op   |
| **Delta INT ops**    | **+0.5**      | **+1.5**      |
| **Delta FP ops**     | **+1** (concurrent) | **+1** (concurrent) |

The FP add executes on the FP32 datapath concurrently with INT, so it does not
increase the INT bottleneck. Only INT ops matter for the ridge-point comparison.

### 2x fp32 -> bf16 (10 bytes/element)

Memory: 4B + 4B read + 2B write = **10 bytes/element**

| EPP | Path   | Unroll | INT (SW) | INT (BW) | AI (SW) | AI (BW) | Bound (SW) | Bound (BW) |
|:---:|:------:|:------:|:--------:|:--------:|:-------:|:-------:|:----------:|:----------:|
| 4   | Exch   | >= 2   | 15.9     | 12.4     | **1.6** | **1.2** | Memory     | Memory     |
| 4   | Simple | 1      | 25.0     | 21.5     | **2.5** | **2.2** | Memory     | Memory     |
| 2   | Exch   | >= 4   | 18.0     | 14.5     | **1.8** | **1.5** | Memory     | Memory     |
| 2   | Simple | 1–3    | 41.5     | 38.0     | **4.2** | **3.8** | Compute    | Compute    |
| 1   | Exch   | >= 8   | 32.2     | 28.2     | **3.2** | **2.8** | Compute    | ~Balanced  |
| 1   | Simple | 1–7    | 80.5     | 76.5     | **8.1** | **7.7** | Compute    | Compute    |

### fp32 + bf16 -> bf16 (8 bytes/element)

Memory: 4B + 2B read + 2B write = **8 bytes/element**

| EPP | Path   | Unroll | INT (SW) | INT (BW) | AI (SW) | AI (BW) | Bound (SW) | Bound (BW) |
|:---:|:------:|:------:|:--------:|:--------:|:-------:|:-------:|:----------:|:----------:|
| 4   | Exch   | >= 2   | 16.9     | 13.4     | **2.1** | **1.7** | Memory     | Memory     |
| 4   | Simple | 1      | 26.0     | 22.5     | **3.3** | **2.8** | Compute    | ~Balanced  |
| 2   | Exch   | >= 4   | 19.0     | 15.5     | **2.4** | **1.9** | Memory     | Memory     |
| 2   | Simple | 1–3    | 42.5     | 39.0     | **5.3** | **4.9** | Compute    | Compute    |
| 1   | Exch   | >= 8   | 33.2     | 29.2     | **4.2** | **3.7** | Compute    | Compute    |
| 1   | Simple | 1–7    | 81.5     | 77.5     | **10.2**| **9.7** | Compute    | Compute    |

---

## GB200 (Blackwell B200) Roofline Comparison

Since this kernel is INT-dominated (Philox RNG + stochastic rounding), the relevant
throughput is the INT32 pipeline:

| Spec                       | Value       |
|----------------------------|-------------|
| FP32 peak (non-tensor)     | ~90 TFLOPS  |
| INT32 peak (1 of 2 datapaths) | ~22 TOPS |
| HBM3e bandwidth            | ~8 TB/s     |
| **INT32 Ridge Point**      | **~2.8 ops/byte** |

---

## Summary: Bulk Path (EPP=4, Exchange)

This is where >99% of data is processed.

| Source config  | Bytes/elem | INT (SW) | INT (BW) | AI (SW) | AI (BW) | vs Ridge 2.8 |
|----------------|:----------:|:--------:|:--------:|:-------:|:-------:|:------------:|
| 1x fp32        | 6          | 15.4     | 11.9     | 2.6     | 2.0     | Memory       |
| 2x fp32        | 10         | 15.9     | 12.4     | **1.6** | **1.2** | Memory       |
| fp32 + bf16    | 8          | 16.9     | 13.4     | **2.1** | **1.7** | Memory       |

### Key Observations

1. **All bulk-path configurations are memory-bound on GB200.** AI ranges from
   1.2 to 2.6 ops/byte, all below the INT32 ridge point of 2.8 ops/byte.

2. **2x fp32 is the most memory-bound** (AI = 1.2–1.6). Doubling read bandwidth
   with almost no extra compute tanks the arithmetic intensity.

3. **fp32 + bf16 sits in between.** The bf16 source saves 2 bytes/element of
   read bandwidth vs a second fp32 source, at the cost of ~1 extra INT op for
   the cast. Net effect: higher AI than 2x fp32, but still well below the
   ridge point.

4. **The extra FP add for reduction runs concurrently** on the FP datapath and
   is fully hidden behind the INT/memory bottleneck.

5. **On Blackwell with HW stochastic rounding** (`__CUDA_ARCH__ >= 1000`), AI
   drops further (~0.5–3.5 fewer INT ops/elem), widening the gap below the
   ridge point and making the kernel even more memory-bound.

6. **Tail paths (EPP=4 simple, EPP=1) are compute-bound** but process negligible
   data volume. The EPP=1 simple path is heavily compute-bound (AI = 8–13)
   due to 1 Philox call per element with no amortization.

7. **The V2 warp-exchange optimization** halves Philox calls on the bulk path
   (1 call per 8 elements vs 1 per 4 in V1 for EPP=4). This saves ~40% of
   RNG compute, but since the bulk path is memory-bound anyway, the benefit
   is reduced power consumption and freed compute headroom rather than
   improved throughput.

### Actual Call Sites from `reduceCopySR`

| Phase | Call           | EPP | Unroll | Exchange? | Role                         |
|:-----:|----------------|:---:|:------:|:---------:|------------------------------|
| 1     | Bulk aligned   | 4   | U      | Yes (U>=2)| **Processes >99% of data**   |
| 2     | Aligned tail   | 4   | 1      | No        | <= 1 hunk of leftovers       |
| 3     | Unaligned bulk | 1   | 2U     | Yes if U>=4 | Only if ptrs were unaligned |
| 4     | Final tail     | 1   | 1      | No        | <= 7 elements                |
