# Design Document: `reduceCopyMixed` / `reduceCopyMixedImpl`

**File:** `comms/ncclx/v2_27/meta/collectives/kernels/common_kernel_quantize.h`
**NCCLX Version:** 2.27

---

## 1. Purpose

`reduceCopyMixed` is a GPU device function for NCCL collective communication (e.g., ReduceScatterQuantize via the PAT algorithm). It performs a fused **reduce + copy** across buffers that may have **different data types**:

- **AccumType** (e.g., `float`, 32-bit): high-precision type used for user data and accumulation.
- **TransportType** (e.g., `__nv_bfloat16`, 16-bit): low-precision type used for network transport buffers.

The function reads from 1--2 source buffers (each independently typed as AccumType or TransportType), reduces them in AccumType precision via a caller-supplied `RedFn` (typically `FuncSum`), optionally applies stochastic rounding if downcasting, and writes to a destination buffer.

---

## 2. Two-Layer Architecture

### 2.1 `reduceCopyMixed` --- Runtime Dispatch Wrapper (lines 404--452)

```
Template params: <Unroll, RedFn, AccumType, TransportType, IntElts>
Runtime params:  src0IsAccumType, src1IsAccumType, dst0IsAccumType (booleans)
```

This function encodes the three runtime booleans into a 3-bit integer (`typeConfig = src0*4 | src1*2 | dst0*1`) and dispatches via a `switch` to one of 8 template instantiations of `reduceCopyMixedImpl`. This converts runtime type selection into compile-time template parameters, enabling `if constexpr` elimination of dead code in the inner loop.

**All 8 type configurations:**

| Config | src0 | src1 | dst0 | Typical use case |
|--------|------|------|------|------------------|
| 0 | Transport | Transport | Transport | bf16 passthrough with re-rounding |
| 1 | Transport | Transport | Accum | bf16 -> float upcast with 2-src reduce |
| 2 | Transport | Accum | Transport | bf16+float -> bf16 mixed reduce+downcast |
| 3 | Transport | Accum | Accum | bf16+float -> float mixed reduce |
| 4 | Accum | Transport | Transport | float+bf16 -> bf16 (symmetric of 2) |
| 5 | Accum | Transport | Accum | float+bf16 -> float (symmetric of 3) |
| 6 | Accum | Accum | Transport | 2x float -> bf16 reduce+downcast |
| 7 | Accum | Accum | Accum | float -> float (same-type, like reduceCopy) |

### 2.2 `reduceCopyMixedImpl` --- Core Implementation (lines 194--381)

```
Template params: <Unroll, RedFn, AccumType, TransportType,
                  Src0IsAccumType, Src1IsAccumType, Dst0IsAccumType, IntElts>
```

The compile-time booleans `Src0IsAccumType`, `Src1IsAccumType`, `Dst0IsAccumType` drive `if constexpr` branches that select:
- The load width (e.g., 8 bytes for bf16 vs 16 bytes for float).
- Whether `convertPackToAccum` is needed after loading.
- Whether `Apply_StochasticCast` is needed before storing.

At compile time, three type aliases are derived (lines 223--225):
```c++
SrcType0 = Src0IsAccumType ? AccumType : TransportType
SrcType1 = Src1IsAccumType ? AccumType : TransportType
DstType0 = Dst0IsAccumType ? AccumType : TransportType
```

**Early-exit guards (lines 207--221):** The function returns immediately if `nDsts == 0` or `nElts <= 0`. It calls `__trap()` (GPU-level abort) if `nSrcs == 0` (invalid) or `nThreads < WARP_SIZE` / not a multiple of `WARP_SIZE` (would cause division-by-zero in warp decomposition).

---

## 3. Constants and Sizing

For the canonical case of `AccumType = float` (4 bytes), `TransportType = __nv_bfloat16` (2 bytes), `Unroll = 4`, `WARP_SIZE = 32`:

| Constant | Formula | Value (float) | Meaning |
|----------|---------|---------------|---------|
| `PackElts` | `16 / sizeof(AccumType)` | 4 | Elements per vectorized pack |
| `AccumPackBytes` | `PackElts * sizeof(AccumType)` | 16 | Bytes per AccumType pack (128-bit load) |
| `SrcPack0Bytes` | `PackElts * sizeof(SrcType0)` | 8 or 16 | Bytes per src0 pack (depends on type) |
| `SrcPack1Bytes` | `PackElts * sizeof(SrcType1)` | 8 or 16 | Bytes per src1 pack |
| `DstPack0Bytes` | `PackElts * sizeof(DstType0)` | 8 or 16 | Bytes per dst0 pack |
| `ElemsPerHunk` | `Unroll * WARP_SIZE * PackElts` | 512 | Elements one warp processes per iteration |

---

## 4. Data Layout and Work Distribution

### 4.1 Thread Hierarchy

The kernel is invoked with `nThreads` threads (must be >= `WARP_SIZE` and a multiple of it). The thread decomposition:

```
nWarps = nThreads / WARP_SIZE     (e.g., 8 warps for 256 threads)
warp   = thread / WARP_SIZE       (warp index: 0..nWarps-1)
lane   = thread % WARP_SIZE       (lane within warp: 0..31)
```

### 4.2 Hunk-Strided Distribution

A **hunk** is the contiguous block of data one warp processes per loop iteration. It contains `ElemsPerHunk = 512` elements (for float).

Warps interleave their hunks across the data array:

```
Total data: [0 ............................................................. nElts-1]

Iteration 0:
  Warp 0: elements [0,     511]     <- hunk 0
  Warp 1: elements [512,   1023]    <- hunk 1
  ...
  Warp W-1: elements [(W-1)*512, W*512-1]  <- hunk W-1

Iteration 1:
  Warp 0: elements [W*512, W*512+511]      <- hunk W
  Warp 1: elements [(W+1)*512, (W+1)*512+511] <- hunk W+1
  ...

(continues until all nHunksTotal = nElts / ElemsPerHunk hunks are consumed)
```

The stride between consecutive hunks for the same warp is:
```
strideElts = nWarps * ElemsPerHunk
```

The loop termination uses a signed countdown:
```
hunksRemaining = nHunksTotal - warp   (initially)
hunksRemaining -= nWarps              (each iteration)
```

When `hunksRemaining <= 0`, the warp exits. This naturally handles the case where `nHunksTotal` is not a multiple of `nWarps` --- the last few warps exit one iteration earlier.

### 4.3 Within a Hunk: Thread-Level Element Mapping

Inside a single hunk, a thread with lane `L` in warp `W` processes elements at these offsets (relative to the hunk start):

```
Unroll iteration u=0:  L*PackElts + 0*WARP_SIZE*PackElts  ->  elements [4L, 4L+3]
Unroll iteration u=1:  L*PackElts + 1*WARP_SIZE*PackElts  ->  elements [128+4L, 128+4L+3]
Unroll iteration u=2:  L*PackElts + 2*WARP_SIZE*PackElts  ->  elements [256+4L, 256+4L+3]
Unroll iteration u=3:  L*PackElts + 3*WARP_SIZE*PackElts  ->  elements [384+4L, 384+4L+3]
```

Concrete example for lane 0 (hunk-relative):
```
u=0: elts [0, 1, 2, 3]       <- 16-byte load at offset 0
u=1: elts [128, 129, 130, 131] <- 16-byte load at offset 512 bytes
u=2: elts [256, 257, 258, 259] <- 16-byte load at offset 1024 bytes
u=3: elts [384, 385, 386, 387] <- 16-byte load at offset 1536 bytes
```

For lane 1:
```
u=0: elts [4, 5, 6, 7]
u=1: elts [132, 133, 134, 135]
u=2: elts [260, 261, 262, 263]
u=3: elts [388, 389, 390, 391]
```

**Memory coalescing:** Within each unroll step `u`, all 32 lanes access consecutive 16-byte packs. Lane 0 accesses bytes `[0,15]`, lane 1 accesses `[16,31]`, ..., lane 31 accesses `[496,511]`. This is a perfect 512-byte coalesced transaction, ideally serviced by a single L2 cache line fetch.

### 4.4 Absolute Address Computation

The absolute byte offset for a load/store by thread `(warp=W, lane=L)` at unroll step `u` in loop iteration `i`:

```
eltOffset = W * ElemsPerHunk + L * PackElts           // initial
          + i * nWarps * ElemsPerHunk                  // hunk stride
          + u * WARP_SIZE * PackElts                   // unroll offset

byteOffset = eltOffset * sizeof(SrcType_or_DstType)
address    = baseAddr + byteOffset
```

Where `baseAddr` is the pre-computed `cvta_to_global()` address.

---

## 5. Per-Buffer Alignment Handling

128-bit loads require 16-byte alignment; 64-bit loads require 8-byte alignment. Buffer pointers can be misaligned when the per-rank element count is not a multiple of `PackElts` (4 for float). For example, in ReduceScatterQuantize PAT with 112 ranks and `count = 1,034,837` (= 1 mod 4), `sendDataRank * count` for odd ranks produces a non-16-byte-aligned byte offset. Non-aligned base pointers from cached/pooled CUDA allocators (e.g., PyTorch sub-tensor views) can also cause misalignment.

### 5.1 Per-Buffer Alignment Checks (lines 260--262)

Rather than an all-or-nothing fallback to the scalar path, each buffer's alignment is checked independently:

```c++
bool src0Aligned = nSrcs == 0 || (src0Addr % SrcPack0Bytes == 0);
bool src1Aligned = nSrcs <= 1 || (src1Addr % SrcPack1Bytes == 0);
bool dst0Aligned = nDsts == 0 || (dst0Addr % DstPack0Bytes == 0);
```

The vectorized hunk loop **always runs** regardless of alignment. `nHunksTotal` is never zeroed. Each buffer independently selects between vectorized and element-wise access within the loop body.

### 5.2 Element-Wise Load/Store Helpers

Two helper functions handle misaligned buffers by decomposing a packed access into `PackElts` individual element-sized loads/stores:

**`ld_volatile_global_elements<AccumType, SrcType, PackElts>`** (lines 153--168):
```c++
// Loads PackElts elements one-by-one from a potentially misaligned address,
// converting each from SrcType to AccumType, and assembles into a BytePack.
// Each ld_volatile_global<sizeof(SrcType)> is naturally aligned for a valid
// SrcType pointer.
template<typename AccumType, typename SrcType, int PackElts>
__device__ __forceinline__ BytePack<PackElts * sizeof(AccumType)>
ld_volatile_global_elements(uintptr_t addr) {
  BytePack<PackElts * sizeof(AccumType)> result;
  #pragma unroll
  for (int i = 0; i < PackElts; i++) {
    BytePack<sizeof(SrcType)> elem =
        ld_volatile_global<sizeof(SrcType)>(addr + i * sizeof(SrcType));
    AccumType val = convertType<AccumType, SrcType>(fromPack<SrcType>(elem));
    memcpy(reinterpret_cast<char*>(&result) + i * sizeof(AccumType),
           &val, sizeof(AccumType));
  }
  return result;
}
```

**`st_global_elements<DstType, PackElts>`** (lines 172--184):
```c++
// Stores PackElts elements one-by-one to a potentially misaligned address.
template<typename DstType, int PackElts>
__device__ __forceinline__ void
st_global_elements(uintptr_t addr, BytePack<PackElts * sizeof(DstType)> pack) {
  #pragma unroll
  for (int i = 0; i < PackElts; i++) {
    BytePack<sizeof(DstType)> elem;
    memcpy(&elem, reinterpret_cast<const char*>(&pack) + i * sizeof(DstType),
           sizeof(DstType));
    st_global<sizeof(DstType)>(addr + i * sizeof(DstType), elem);
  }
}
```

This follows the `bulkLoad` pattern in `unpack.h`.

### 5.3 Why Per-Buffer Instead of All-or-Nothing

In the common misalignment case (Scenario 2 from `rsq_alignment_issue_explained.md`), only the user input buffer (`srcs[1]`) is misaligned while transport buffers remain aligned. The per-buffer approach keeps aligned buffers on the fast vectorized path while only the misaligned buffer pays the cost of element-wise access.

A head-skip approach (skipping a few scalar elements to align all pointers, then vectorizing) was considered and rejected because the different buffers are typically misaligned by **different** offsets (transport has skip=0 while user input has skip≠0), making a uniform head-skip impossible. See `rsq_alignment_issue_explained.md` for the full analysis.

**Performance comparison (Scenario 2 — only src1 misaligned):**

| | Scalar fallback (old) | Per-buffer element-wise (current) | Fully aligned |
|---|---|---|---|
| Elements/thread/iter | 1 | 16 | 16 |
| Mem ops/element | 3 | 1.5 | 0.75 |
| Relative throughput | 1x | ~2x | ~4x |

---

## 6. Execution Pipeline (Vectorized Main Loop)

Each iteration of the `while (hunksRemaining > 0)` loop processes one hunk per warp in 4 phases:

### Phase 1: Load Source 0 (lines 273--289)

```
for u in [0, Unroll):
    eidx = eltOffset + u * WARP_SIZE * PackElts
    loadAddr = src0Addr + eidx * sizeof(SrcType0)
    if src0Aligned:
        if Src0IsAccumType:
            acc[u] = ld_volatile_global<16>(loadAddr)          // 128-bit load, 4 floats
        else:
            srcPack = ld_volatile_global<8>(loadAddr)          // 64-bit load, 4 bf16s
            acc[u] = convertPackToAccum(srcPack)               // expand to 4 floats
    else:
        acc[u] = ld_volatile_global_elements<AccumType, SrcType0, PackElts>(loadAddr)
```

All `Unroll=4` loads are issued in sequence via `#pragma unroll`. The GPU scheduler can overlap later loads with earlier ones' memory latency. Result: `acc[0..3]`, each a `BytePack<16>` holding 4 floats.

### Phase 2: Load Source 1 + Reduce (lines 292--313)

Only executed if `nSrcs > 1`:

```
// Load phase
for u in [0, Unroll):
    loadAddr = src1Addr + eidx * sizeof(SrcType1)
    if src1Aligned:
        // vectorized load + optional convert
    else:
        tmp[u] = ld_volatile_global_elements<AccumType, SrcType1, PackElts>(loadAddr)

// Reduce phase (separated from loads for ILP)
for u in [0, Unroll):
    acc[u] = applyReduce(redFn, acc[u], tmp[u])
```

`applyReduce` operates directly on `BytePack<16>` --- it performs element-wise reduction (typically summation) on the packed representation without unpacking to scalars.

### Phase 3: Store to Destination (lines 316--339)

```
for u in [0, Unroll):
    storeAddr = dst0Addr + eidx * sizeof(DstType0)
    if Dst0IsAccumType:
        if dst0Aligned:
            st_global<16>(storeAddr, acc[u])                   // 128-bit store
        else:
            st_global_elements<DstType0, PackElts>(storeAddr, acc[u])
    else:
        dstPack = Apply_StochasticCast<float, bf16, PackElts>::cast(
                      acc[u], randomSeed, elemIdx)             // 4 floats -> 4 bf16 with SR
        if dst0Aligned:
            st_global<8>(storeAddr, dstPack)                   // 64-bit store
        else:
            st_global_elements<DstType0, PackElts>(storeAddr, dstPack)
```

When the destination is TransportType (downcast), stochastic rounding is applied via `Apply_StochasticCast` **before** the alignment branch. This batches 4 Philox RNG outputs regardless of whether the store is vectorized or element-wise — the RNG cost is the same either way.

### Phase 4: Advance

```
eltOffset += strideElts        // jump to this warp's next hunk
hunksRemaining -= nWarps       // countdown
```

---

## 7. Scalar Tail Loop (lines 346--380)

After the vectorized loop, `packedElts = nHunksTotal * ElemsPerHunk` elements have been processed. The remaining `nElts - packedElts` elements (at most `ElemsPerHunk - 1 = 511` for float) are handled by a simple thread-strided scalar loop:

```
for idx = packedElts + thread; idx < nElts; idx += nThreads:
    acc = loadAndConvert<AccumType>(src0, idx)
    hasValue = true

    if nSrcs > 1:
        val1 = loadAndConvert<AccumType>(src1, idx)
        if hasValue:
            acc = reduceAccum(redFn, acc, val1)
        else:
            acc = val1
            hasValue = true

    if hasValue:
        if Dst0IsAccumType:
            dst0[idx] = acc
        else:
            Apply_StochasticCast<..., 1>::cast(...)
            dst0[idx] = result
```

Key differences from vectorized path:
- Element-by-element loads/stores (4 bytes for float, 2 bytes for bf16) --- always naturally aligned, so alignment is never an issue.
- Stochastic rounding uses `EltPerPack=1`, wasting 3 of every 4 Philox outputs.
- All `nThreads` threads participate (not warp-organized).
- Uses a `hasValue` flag to guard the reduce and store, handling the edge case where `nSrcs` changes behavior.

---

## 8. Helper Functions

### 8.1 `convertPackToAccum<AccumType, TransportType, PackElts>` (lines 133--147)

Converts a packed load of `PackElts` TransportType elements to AccumType:

```
Input:  BytePack<PackElts * sizeof(TransportType)>  e.g., BytePack<8> = 4 bf16s
Output: BytePack<PackElts * sizeof(AccumType)>      e.g., BytePack<16> = 4 floats
```

Uses `memcpy` with compile-time-constant sizes (optimized to register moves) and `convertType` (maps to hardware intrinsics like `__bfloat162float`, a single instruction on H100). The loop is `#pragma unroll`'d --- zero overhead at runtime.

### 8.2 `convertType<Dst, Src>` (lines 29--65)

Template specializations for all supported conversions:
- `float <-> float` (identity)
- `__nv_bfloat16 -> float` (`__bfloat162float`)
- `float -> __nv_bfloat16` (`__float2bfloat16`)
- `__half -> float` (`__half2float`)
- `float -> __half` (`__float2half`)
- `__half -> __half`, `__nv_bfloat16 -> __nv_bfloat16` (identity)

### 8.3 `convertWithStochasticRounding<TransportType, AccumType>` (lines 69--99)

Used only in the **scalar tail loop** (the vectorized path uses `Apply_StochasticCast` from `stochastic_cast.cuh`). Adds random noise to the bits that will be truncated:
- bf16: adds random 16-bit noise to the lower 16 bits of the float representation.
- fp16: adds random 13-bit noise to the lower 13 bits.

### 8.4 `reduceAccum` (lines 123--129)

Wraps `applyReduce` for scalar single-element reduction: converts scalars to/from `BytePack<sizeof(AccumType)>` for compatibility with the `reduce_kernel.h` API. Used only in the scalar tail.

### 8.5 `ld_volatile_global_elements<AccumType, SrcType, PackElts>` (lines 153--168)

Element-wise load helper for misaligned buffers. Performs `PackElts` individual `ld_volatile_global<sizeof(SrcType)>` loads (each naturally aligned for a valid typed pointer), converts each element from `SrcType` to `AccumType`, and assembles the results into a `BytePack<PackElts * sizeof(AccumType)>`. This is the per-buffer fallback that replaces the old all-or-nothing scalar fallback.

### 8.6 `st_global_elements<DstType, PackElts>` (lines 172--184)

Element-wise store helper for misaligned buffers. Decomposes a `BytePack<PackElts * sizeof(DstType)>` into `PackElts` individual `st_global<sizeof(DstType)>` stores. Used for both AccumType and TransportType destinations when the destination pointer is misaligned.

---

## 9. Memory Access Patterns Summary

For the typical PAT ReduceScatterQuantize case (`src0 = bf16* recv buffer`, `src1 = float* user input`, `dst0 = bf16* send buffer`):

**Aligned buffers:**

| Buffer | Type | Pack Width | Load/Store Instruction | Alignment |
|--------|------|------------|----------------------|-----------|
| src0 (recv buf) | bf16 | 8 bytes (4 x bf16) | `ld_volatile_global<8>` -> `convertPackToAccum` | 8-byte |
| src1 (user input) | float | 16 bytes (4 x float) | `ld_volatile_global<16>` | 16-byte |
| dst0 (send buf) | bf16 | 8 bytes (4 x bf16) | `Apply_StochasticCast` -> `st_global<8>` | 8-byte |

**Misaligned buffers (per-buffer fallback):**

| Buffer | Type | Pack Width | Load/Store Instruction | Alignment |
|--------|------|------------|----------------------|-----------|
| src0 (misaligned) | bf16 | 2 bytes (1 x bf16) | `ld_volatile_global<2>` x4 -> convert each | 2-byte |
| src1 (misaligned) | float | 4 bytes (1 x float) | `ld_volatile_global<4>` x4 -> convert each | 4-byte |
| dst0 (misaligned) | bf16 | 2 bytes (1 x bf16) | `Apply_StochasticCast` -> `st_global<2>` x4 | 2-byte |

Per warp per iteration: `4 (Unroll) x 32 (lanes)` = 128 packs = 512 elements.

Bandwidth per warp per iteration (all aligned):
- src0 reads: 128 x 8 = 1024 bytes
- src1 reads: 128 x 16 = 2048 bytes
- dst0 writes: 128 x 8 = 1024 bytes
- Total: 4096 bytes per warp per iteration

---

## 10. Stochastic Rounding Details

When `Dst0IsAccumType = false` (destination is TransportType), stochastic rounding is required for the downcast.

**Vectorized path:** Uses `Apply_StochasticCast<AccumType, DstType0, PackElts>::cast()` from `stochastic_cast.cuh`. The `PackElts=4` specialization calls Philox **once** to generate 4 random 32-bit numbers, using all 4 for the 4 elements in the pack. The Philox counter is `randomBaseOffset + eidx`, ensuring deterministic, reproducible randomness. The stochastic rounding is applied **before** the aligned/misaligned store branch — the same `dstPack` is produced either way, only the store instruction differs.

**Scalar tail:** Uses `Apply_StochasticCast<..., 1>::cast()` per element. Each call generates 4 random numbers but uses only 1, wasting 75% of Philox output. This is acceptable because the tail handles at most 511 elements.

---

## 11. Potential Improvements

### 11.1 Support for `nDsts > 1` (Feature Gap)

**Current:** The implementation assumes `nDsts <= 1` --- only `dst0` is written. The `dsts` array is indexed only at position 0.

**If needed:** Supporting a second destination (e.g., writing both a transport buffer and a local accumulation buffer) would require adding `dst1Addr`, a `Dst1IsAccumType` template parameter, and a second store phase. This would expand the 8-configuration dispatch to 16 configurations.

### 11.2 Reduce Scalar Tail Philox Waste

**Problem:** The scalar tail uses `Apply_StochasticCast<..., 1>`, calling Philox per element and wasting 3/4 of its output.

**Proposal:** Accumulate up to 4 scalar elements before calling Philox once with `EltPerPack=4`, even in the scalar tail. This requires buffering elements and a final partial-pack flush, but for tails of up to 511 elements, the Philox savings could be meaningful (up to 4x fewer Philox calls in the tail). In practice this is low priority because the tail is a small fraction of total work.

### 11.3 Reducing Template Bloat (Code Size)

**Problem:** The 8-way `switch` in `reduceCopyMixed` instantiates 8 copies of `reduceCopyMixedImpl`. Not all 8 configurations may be used in practice. Each instantiation consumes instruction cache and register file resources.

**Proposal:** Profile which configurations are actually invoked in production collectives and conditionally compile only those (e.g., via `#if` guards or a reduced dispatch table). For ReduceScatterQuantize PAT, only configurations 2 (Transport+Accum -> Transport) and 3 (Transport+Accum -> Accum) are likely used.

### 11.4 Use of `volatile` Loads for src0 Only

**Observation:** Source 0 uses `ld_volatile_global` (volatile semantics, preventing caching in L1), while stores use `st_global` (non-volatile). The volatile loads are necessary for src0 because it reads from a transport buffer that is written by a remote GPU via RDMA --- without volatile, the local L1 cache could serve stale data. However, src1 (user input buffer) is typically local and read-only during the kernel's execution, so a non-volatile `ld_global` could benefit from L1 caching. Currently both sources use `ld_volatile_global` uniformly (both in the aligned and element-wise paths).

**Proposal:** Use `ld_global` (non-volatile) for src1 when it is known to be a local user buffer. This could improve L1 hit rates for the user input, particularly in multi-pass algorithms where the same user data is read multiple times. This would require either a template parameter or a runtime flag to distinguish transport buffers from user buffers.

### 11.5 Async Copy with `cp.async` / TMA (Future Hardware)

**Problem:** The current approach issues vectorized loads followed by computation. On H100 and beyond, the Tensor Memory Accelerator (TMA) and `cp.async` can overlap data movement with computation more effectively by using shared memory as a staging buffer.

**Proposal:** For very large messages, a double-buffered pipeline could prefetch the next hunk's data into shared memory via `cp.async` while the current hunk is being reduced. This would further hide memory latency. The complexity is significant and likely only worthwhile for the largest message sizes.

### 11.6 Shared Memory Reduction for `nSrcs > 2`

**Current:** The implementation handles at most 2 sources. If a future collective needs more sources (e.g., an AllReduce variant accumulating from multiple peers), the current pattern of `acc[] + tmp[]` doesn't scale.

**Proposal:** For `nSrcs > 2`, load all sources into shared memory, then use warp-shuffle or shared-memory reduction to accumulate. This is speculative and only relevant if the collective protocol changes.
