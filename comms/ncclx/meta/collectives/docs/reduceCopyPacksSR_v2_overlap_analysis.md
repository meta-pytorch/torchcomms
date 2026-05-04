# Why `reduceCopyPacksSR` Underperforms `reduceCopyMixed`: Overlap Failure Analysis

**Context:** The benchmark (`SimpleCopySRV2Bench.cu`) shows `reduceCopyPacksSR`
(with stochastic rounding) achieving lower bandwidth than `reduceCopyMixed`/`reduceCopyPacks`
(RTN, no SR), despite the arithmetic intensity analysis predicting the bulk path
(EPP=4, exchange) should be memory-bound on GB200.

This document analyzes why the Philox RNG compute and memory accesses fail to
overlap, and provides a profiling procedure to identify the root cause.

---

## Relevant Code Structure

### Baseline (`reduceCopyPacks`, copy_kernel.cuh)

```
while (iter.hasWork()) {
    Load sources → acc[]           // memory
    Reduce additional sources      // FP add (trivial)
    Store (RTN cast)               // memory
    advance
}
```

Nearly all time is spent in memory operations (load → cast → store). Minimal
compute between loads and stores.

### SR V2 (`reduceCopyPacksSR`, reduce_copy_sr_v2.cuh, lines 465–479)

```
while (iter.hasWork()) {
    Load sources → acc[]           // memory (ld.volatile.global)
    Reduce additional sources      // FP add
    PhiloxWarpExchange → rng       // ~60+ INT instructions, no mem dependency
    Store with SR                  // uses both acc[] and rng
    advance
}
```

The Philox RNG has NO data dependency on the loads, so in theory the GPU should
overlap them. In practice, it does not.

### Memory Semantics

All loads use `ld.volatile.global` (PTX) with `asm volatile(... : "memory")`
clobber (op128.h, lines 218–223). All stores use `st.global` with
`asm volatile(... : "memory")` clobber (line 226–228). The `"memory"` clobber
on every operation acts as a compiler memory barrier, constraining instruction
scheduling at the nvcc frontend level.

---

## Hypotheses

### Hypothesis 1: L2 Cache Makes the Kernel Compute-Bound at Benchmark Scale

**Likelihood: High. This is likely the primary cause.**

The benchmark uses `kN = 4M elements`:

| Config | Read | Write | Total |
|--------|------|-------|-------|
| 1 src (float→bf16) | 16 MB | 8 MB | 24 MB |
| 2 src (f32+f32→bf16) | 32 MB | 8 MB | 40 MB |

H100 L2 cache is **50 MB**. The entire working set fits in L2. After warmup
(10 iterations), the 100 measured iterations all read from L2 at ~12 TB/s,
not HBM at ~3.35 TB/s.

This completely changes the roofline:

| Bandwidth Source | INT32 Peak | Ridge Point |
|-----------------|-----------|-------------|
| HBM (3.35 TB/s) | 22 TOPS | 6.6 ops/byte |
| **L2 (~12 TB/s)** | 22 TOPS | **1.8 ops/byte** |

At L2 bandwidth, the SR kernel's AI of 2.0–2.6 ops/byte (for the EPP=4
exchange path) is **above** the ridge point of 1.8 ops/byte. The kernel becomes
**compute-bound when running from L2**. The baseline has near-zero compute, so
it stays memory-bound even at L2 speeds. The Philox RNG becomes the bottleneck.

**Validation:** Run the benchmark with `kN = 64M` or `256M` elements
(256 MB–1 GB working set, doesn't fit L2). If the gap between SR and baseline
shrinks dramatically at large sizes, L2 caching is the explanation — and the
original HBM-based AI analysis is correct for real-world (large buffer) usage.

---

### Hypothesis 2: Register Pressure Limits Occupancy

**Likelihood: High. Co-occurs with Hypothesis 1.**

The SR kernel needs significantly more registers per thread than the baseline:

| Kernel | Live register sets | Estimated regs/thread |
|--------|-------------------|----------------------|
| Baseline (RTN) | `acc[Unroll]` | ~30–40 |
| V1 SR | `acc[U]` + `randR0..R3[U]` + seed/offset | ~50–80 |
| V2 SR | `acc[U]` + `rand_a/b[U]` + exchange temps | ~50–80 |

With `__launch_bounds__(256, 1)`, the compiler can use up to 255 registers
(`minBlocksPerMultiprocessor=1`). Combined with the benchmark's launch config
of 32 blocks × 256 threads:

- Total threads: 8192 across 132 SMs (H100)
- Active warps per SM: ~8 (only ~32 SMs get a block)
- Occupancy: 8/64 = **12.5%**

At 12.5% occupancy, latency hiding depends almost entirely on instruction-level
parallelism (ILP) within each warp, not warp-level parallelism (WLP). The
baseline achieves good ILP because its loop body is almost entirely memory
operations (load → cast → store — the GPU can have many loads in-flight per
warp). The SR kernel has a **~70+ cycle Philox compute block** where no memory
instructions issue. With only 8 warps on the SM, there may be no other warp
ready to issue loads during another warp's compute phase.

**Validation:**
- Nsight Compute: compare `launch__registers_per_thread` for SR vs baseline.
- Try `__launch_bounds__(256, 2)` or `(256, 4)` to force fewer registers and
  higher occupancy. If performance improves, occupancy was the bottleneck.

---

### Hypothesis 3: `asm volatile("memory")` Prevents Compiler Instruction Scheduling

**Likelihood: Medium-High.**

Every `ld_volatile_global` and `st_global` uses `asm volatile(... : "memory")`
(op128.h). The `"memory"` clobber tells the **nvcc frontend** that each
load/store reads/writes arbitrary memory, acting as a full compiler barrier.
This means:

- The compiler cannot move Philox instructions ahead of loads.
- The compiler cannot issue iteration N+1 loads while iteration N's Philox is
  in progress.
- Instructions are emitted in strict source order: all loads → all reduces →
  all Philox → all stores.

**ptxas** (PTX→SASS backend) can still reorder within a basic block, but
`ld.volatile` in PTX has ordering constraints relative to other volatile
operations. In practice, ptxas is conservative.

The net effect is a serialized timeline within each iteration:

```
[---- load latency ----][-- reduce --][------ Philox ------][-- SR + store --]
```

The baseline is unaffected because its loop body is almost entirely memory
instructions — there's nothing to interleave, and back-to-back loads naturally
pipeline.

**Validation:** Dump SASS with `cuobjdump --dump-sass <binary>` and examine
instruction ordering. Check whether Philox instructions (IMUL, LOP3, IADD3) are
interleaved with LDG instructions, or if they form a serialized block after all
loads.

---

### Hypothesis 4: Long Compute Block Starves the Memory Pipeline

**Likelihood: Medium.**

Even within a single warp, the GPU can have ~12–32 outstanding memory requests.
In the baseline:

```
LDG → LDG → LDG → LDG → (cast) → STG → STG → STG → STG → next LDG...
```

The memory pipeline is almost never idle — loads and stores are back-to-back.

In the SR kernel:

```
LDG → LDG → LDG → LDG → (reduce) → IMUL×56 → SHFL → CMOV×10 → SR → STG...
```

After the last LDG completes, the warp enters **70+ cycles of pure INT
compute** (Philox + exchange + SR). During this window:
- No new LDGs are issued (next iteration's loads haven't started)
- The memory subsystem goes idle for that warp
- With only 8 warps on the SM (see Hypothesis 2), there may be no other warp
  ready to feed the memory pipeline

This is the operational consequence of Hypotheses 2 and 3 combined.

**Validation:** Nsight Compute warp stall reasons:
- `smsp__warps_issue_stalled_long_scoreboard` — warps waiting for memory
- `smsp__warps_issue_stalled_math_pipe_throttle` — INT pipe saturated
- Warp Scheduler "No Eligible" — fraction of cycles with nothing to issue

---

### Hypothesis 5: Store Buffer Back-Pressure

**Likelihood: Medium-Low.**

Stores use `st.global` (non-volatile, cached). The GPU has a limited store
buffer per SM. If the Philox compute delays the SR output, stores from the
*current* iteration are held pending while loads from the *next* iteration
compete for the same L1/memory pipeline slots.

This creates a hidden dependency:

```
Philox slow → SR delayed → stores delayed → store buffer full → new loads blocked
```

This is particularly acute when the working set fits in L2 (Hypothesis 1),
because L2 reads complete fast but compute can't keep up, leading to
back-pressure.

**Validation:** Nsight Compute metrics:
- `smsp__warps_issue_stalled_lg_throttle` — stalled on local/global throttle
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` — store throughput

---

## Profiling Procedure

### Step 1: Confirm L2 vs HBM Regime

Run the benchmark at multiple data sizes to determine if the SR overhead is a
function of L2 residency.

```cpp
// Modify kN in SimpleCopySRV2Bench, or parameterize:
// kN = 4M   (16 MB float, fits L2 — current)
// kN = 64M  (256 MB, partial L2)
// kN = 256M (1 GB, HBM-dominated)
```

**Expected result:** If the gap between SR and baseline shrinks at large sizes,
L2 caching is the root cause and the kernel is truly memory-bound at HBM scale.

### Step 2: Nsight Compute Single-Kernel Profile

Profile a single invocation to get detailed per-SM metrics:

```bash
# Profile V2 SR kernel
ncu --set full \
    --kernel-name "v2_packs_sr_kernel" \
    --launch-skip 10 --launch-count 1 \
    <benchmark_binary> --gtest_filter="*PacksMatrix_FloatToBf16*"

# Profile baseline (RTN) kernel
ncu --set full \
    --kernel-name "baseline_packs_kernel" \
    --launch-skip 10 --launch-count 1 \
    <benchmark_binary> --gtest_filter="*PacksMatrix_FloatToBf16*"
```

**Key metrics to compare (SR vs baseline):**

| Category | Metric | What It Tells You |
|----------|--------|-------------------|
| Registers | `launch__registers_per_thread` | Register pressure |
| Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy |
| Memory | `dram__bytes.sum` | Actual HBM traffic (vs L2) |
| Memory | `lts__t_bytes.sum` | L2 bytes accessed |
| Memory | `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | L1 load traffic |
| Pipeline | `smsp__inst_executed_pipe_alu.sum` | INT instructions |
| Pipeline | `smsp__inst_executed_pipe_fma.sum` | FP instructions |
| Stalls | `smsp__warps_issue_stalled_long_scoreboard` | Waiting for memory |
| Stalls | `smsp__warps_issue_stalled_math_pipe_throttle` | INT pipe saturated |
| Stalls | `smsp__warps_issue_stalled_lg_throttle` | Store buffer full |
| Stalls | `smsp__warps_issue_stalled_not_selected` | Ready but not picked |
| Scheduler | `smsp__issue_active.avg.pct_of_peak_sustained_active` | % cycles issuing |
| Roofline | Use Nsight's built-in Roofline chart | Visual bound confirmation |

### Step 3: SASS Inspection for Instruction Interleaving

```bash
cuobjdump --dump-sass <benchmark_binary> | less
# Search for the SR kernel function name
```

Look for the pattern between LDG (global load) and IMUL/LOP3 (Philox)
instructions.

**Ideal (overlapped) — loads interleaved with Philox:**

```
LDG R4, [R2]           ; iteration N load
IMUL R10, R8, 0x...    ; Philox from current or prior iteration
LDG R5, [R2+0x80]      ; another load
LOP3 R11, R10, ...     ; more Philox
```

**Problematic (serialized) — all loads then all compute:**

```
LDG R4, [R2]           ; all loads first
LDG R5, [R2+0x80]
LDG R6, [R2+0x100]
LDG R7, [R2+0x180]
; --- gap: wait for loads to complete ---
IMUL R10, R8, 0x...    ; then all Philox
IMUL R11, R9, 0x...
LOP3 ...
; --- then stores ---
STG [R3], R12
```

### Step 4: Occupancy Experiment

Create kernel variants with different `__launch_bounds__` to isolate the
occupancy effect:

```cpp
__launch_bounds__(256, 2)  // Force compiler to fit 2 blocks/SM → fewer regs
__launch_bounds__(256, 4)  // Even more aggressive
```

If performance improves with forced higher occupancy (even if individual warp
throughput drops slightly due to register spilling), it confirms that latency
hiding via warp-level parallelism is the bottleneck.

### Step 5: Isolate Philox Cost with a Compute-Only Kernel

Write a kernel that does ONLY the Philox RNG (no loads/stores) to measure
raw Philox throughput:

```cpp
__global__ void philox_only_kernel(uint64_t seed, uint64_t baseOff, uint64_t nElts) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t r0, r1, r2, r3;
    uint32_t sink = 0;
    for (uint64_t i = tid; i < nElts; i += blockDim.x * gridDim.x) {
        philox_randint4x(seed, baseOff + i, r0, r1, r2, r3);
        sink ^= r0 ^ r1 ^ r2 ^ r3;
    }
    if (sink == 0xDEADBEEF) asm volatile("" :: "r"(sink)); // prevent DCE
}
```

Compare throughput (elements/sec) against the SR kernel's throughput. If Philox
alone can sustain the rate needed to match memory bandwidth, the compute is not
inherently the bottleneck — the overlap failure is. If Philox alone is slower,
the kernel is fundamentally compute-bound.

### Step 6: Source Reordering Experiment

Test whether moving Philox computation **before** loads helps the compiler
interleave instructions:

```cpp
while (iter.hasWork()) {
    // Generate RNG FIRST (addresses are known before data is loaded)
    PhiloxWarpExchange<Unroll, EltPerPack> rng;
    rng.generate(randomSeed, randomBaseOffset, threadEltBase, lane);

    // THEN load + reduce (loads can issue while Philox is in-flight)
    BytePack<...> acc[Unroll];
    loadFirstSource(acc, iter);
    ReduceSources::apply(acc, iter);

    // Store with SR
    storeFirstDestinationSR(acc, iter, rng);

    iter.advance();
    threadEltBase += ...;
}
```

If the RNG is placed before loads, the compiler may issue load instructions
while Philox integer instructions are executing, since loads have no dependency
on the RNG output. If this variant improves performance, it confirms that
instruction scheduling (Hypothesis 3) is a significant factor.

---

## Summary

| # | Hypothesis | Likelihood | Quick Validation |
|---|-----------|-----------|-----------------|
| 1 | **L2 cache makes AI above ridge point** | High | Increase `kN` to 256M elements |
| 2 | **Register pressure → low occupancy** | High | `ncu --set full`, check regs + occupancy |
| 3 | **`asm volatile("memory")` blocks scheduling** | Medium-High | SASS dump: check instruction ordering |
| 4 | **Compute block starves memory pipeline** | Medium | Warp stall reasons in Nsight Compute |
| 5 | **Store buffer back-pressure** | Medium-Low | `lg_throttle` stall metric |

Hypotheses 1 and 2 likely co-occur: at L2 bandwidth the ridge point drops to
~1.8 ops/byte, making the SR kernel compute-bound, and low occupancy removes
the ability to hide either memory or compute latency. Hypothesis 3 (compiler
barrier from `asm volatile("memory")`) exacerbates the problem by preventing
the overlap that would make the kernel memory-bound even at L2 speeds.
