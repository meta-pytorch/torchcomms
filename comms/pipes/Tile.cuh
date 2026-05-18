// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Tile: cooperative vectorized tile operations for GPU collectives.
//
// A Tile is a fixed-size block of elements distributed across threads in a
// ThreadGroup, backed by uint4 registers. All memory operations use 16-byte
// (uint4) vectorized loads/stores for optimal coalescing.
//
// TERMINOLOGY
// ===========
//   element   A single value of type T (one float, one __half, one bf16).
//   vector    A uint4 register (16 bytes). Holds kElemsPerVec elements:
//               float  → 4 elements per vector  (16B / 4B)
//               __half → 8 elements per vector  (16B / 2B)
//               bf16   → 8 elements per vector  (16B / 2B)
//   tile      A contiguous block of kTileElems elements, cooperatively owned
//             by kBlockSize threads. Each thread holds kVPT vectors in
//             registers.
//
// KEY CONSTANTS
// =============
//   kElemsPerVec (kEPV)  Elements per vector = sizeof(uint4) / sizeof(T).
//   kTileVecs            Vectors in the tile = kTileElems / kEPV.
//   kVPT                 Vectors per thread  = kTileVecs / kBlockSize.
//                        This controls ILP — higher kVPT means more work per
//                        thread per tile. kVPT=8 matches memcpy_vectorized
//                        and achieves peak bandwidth on H100.
//
// TRITON EQUIVALENCE
// ==================
//   Tile API                                  Triton
//   ──────────────────────────────────────     ────────────────────────────────
//   tile_load<T,N>(p, idx, g)                 tl.load(p + offs)
//   tile_load<T,N>(p, idx, g, valid)          tl.load(p+offs, mask=m, other=0)
//   tile_store(p, idx, tile, g)               tl.store(p + offs, tile)
//   tile_store(p, idx, tile, g, valid)        tl.store(p + offs, tile, mask=m)
//   tile_accumulate<T,SumOp>(a, b)            a += b
//   tile_accumulate<T,MaxOp>(a, b)            a = tl.maximum(a, b)
//   tile_load_accumulate<T,SumOp>(a,p,i,g)    a += tl.load(p + offs)
//   tile_zero(tile)                           tl.zeros(BLOCK, dtype)
//   num_tiles<T,N>(n)                         n // BLOCK_SIZE
//   tile_remainder<T,N>(n)                    n % BLOCK_SIZE
//
//   The key structural difference: Triton masks at element granularity via
//   boolean mask tensors. This API masks at element granularity too — partial
//   vectors at the boundary are handled with scalar element loads/stores so
//   valid_elems need not be vector-aligned.
//
// PIPELINING
// ==========
//   Multiple tiles in registers enable software pipelining:
//
//     auto t0 = tile_load(ptr, 0, group);
//     for (size_t i = 1; i < n; i++) {
//       auto t1 = tile_load(ptr, i, group);   // load next
//       tile_store(out, i-1, t0, group);       // store previous
//       t0 = t1;
//     }
//     tile_store(out, n-1, t0, group);
//
//   Register pressure scales with kVPT × pipeline depth. Profile to find
//   the sweet spot between ILP and spilling.
//
// 2D TILES
// =========
//   tile_load_2d/tile_store_2d load a kRows × kCols sub-matrix from a
//   row-major matrix with a given stride. The result is a flat
//   Tile<T, kRows*kCols, kBlockSize> compatible with all 1D tile ops.
//
//     tile_load_2d<T, kRows, kCols>(ptr, row, col, stride, group,
//                                   valid_rows, valid_cols)
//
//   This maps to Triton's 2D load pattern:
//     tl.load(ptr + rows[:,None]*stride + cols[None,:], mask=mask_2d)
//
//   Vectorization runs along the column (contiguous) dimension. Rows are
//   distributed across threads. This requires kCols >= kElemsPerVec and
//   kCols % kElemsPerVec == 0 for full vectorization.
//
// USAGE
// =====
//   auto tile = tile_load<float, 8192>(ptr, tile_idx, group);
//   auto peer = tile_load<float, 8192>(peer_ptr, tile_idx, group);
//   tile_accumulate<float, SumOp>(tile, peer);
//   tile_store(out_ptr, tile_idx, tile, group);

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes {

// ============================================================================
// Reduction ops: tag types for selecting the reduction operation.
//
// Used as template arguments to tile_accumulate / tile_load_accumulate.
// Each tag dispatches to the corresponding VecOps::reduce overload.
// Adding a new op = one reduce() overload per VecOps specialization.
// ============================================================================

struct SumOp {};
struct MaxOp {};
struct MinOp {};

// ============================================================================
// VecOps<T>: per-type vectorized reduction on uint4 vectors.
//
//   kElemsPerVec  Number of T elements packed in one uint4 (16 bytes).
//   reduce(Op, a, b)  Element-wise reduction: a = Op(a, b), operating on
//                     all kElemsPerVec elements inside the uint4 pair.
//
// To add a new element type: specialize VecOps<NewType> with kElemsPerVec
// and one reduce() overload per op tag. For 16-bit types that use the
// __hadd2/__hmax2/__hmin2 intrinsics, inherit from HalfPrecisionVecOps.
// ============================================================================

template <typename T>
struct VecOps;

template <>
struct VecOps<float> {
  static constexpr int kElemsPerVec = sizeof(uint4) / sizeof(float); // 4

  __device__ __forceinline__ static void
  reduce(SumOp, uint4& a, const uint4& b) {
    float4& va = reinterpret_cast<float4&>(a);
    const float4& vb = reinterpret_cast<const float4&>(b);
    va.x += vb.x;
    va.y += vb.y;
    va.z += vb.z;
    va.w += vb.w;
  }

  __device__ __forceinline__ static void
  reduce(MaxOp, uint4& a, const uint4& b) {
    float4& va = reinterpret_cast<float4&>(a);
    const float4& vb = reinterpret_cast<const float4&>(b);
    va.x = fmaxf(va.x, vb.x);
    va.y = fmaxf(va.y, vb.y);
    va.z = fmaxf(va.z, vb.z);
    va.w = fmaxf(va.w, vb.w);
  }

  __device__ __forceinline__ static void
  reduce(MinOp, uint4& a, const uint4& b) {
    float4& va = reinterpret_cast<float4&>(a);
    const float4& vb = reinterpret_cast<const float4&>(b);
    va.x = fminf(va.x, vb.x);
    va.y = fminf(va.y, vb.y);
    va.z = fminf(va.z, vb.z);
    va.w = fminf(va.w, vb.w);
  }

  __device__ __forceinline__ static void
  reduce_scalar(SumOp, float& a, const float& b) {
    a += b;
  }

  __device__ __forceinline__ static void
  reduce_scalar(MaxOp, float& a, const float& b) {
    a = fmaxf(a, b);
  }

  __device__ __forceinline__ static void
  reduce_scalar(MinOp, float& a, const float& b) {
    a = fminf(a, b);
  }
};

template <typename ScalarT, typename PackedT>
struct HalfPrecisionVecOps {
  static constexpr int kElemsPerVec = sizeof(uint4) / sizeof(ScalarT);
  static constexpr int kPairsPerVec = kElemsPerVec / 2;

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  __device__ __forceinline__ static void
  reduce(SumOp, uint4& a, const uint4& b) {
    PackedT* va = reinterpret_cast<PackedT*>(&a);
    const PackedT* vb = reinterpret_cast<const PackedT*>(&b);
#pragma unroll
    for (int i = 0; i < kPairsPerVec; i++) {
      va[i] = __hadd2(va[i], vb[i]);
    }
  }

  __device__ __forceinline__ static void
  reduce(MaxOp, uint4& a, const uint4& b) {
    PackedT* va = reinterpret_cast<PackedT*>(&a);
    const PackedT* vb = reinterpret_cast<const PackedT*>(&b);
#pragma unroll
    for (int i = 0; i < kPairsPerVec; i++) {
      va[i] = __hmax2(va[i], vb[i]);
    }
  }

  __device__ __forceinline__ static void
  reduce(MinOp, uint4& a, const uint4& b) {
    PackedT* va = reinterpret_cast<PackedT*>(&a);
    const PackedT* vb = reinterpret_cast<const PackedT*>(&b);
#pragma unroll
    for (int i = 0; i < kPairsPerVec; i++) {
      va[i] = __hmin2(va[i], vb[i]);
    }
  }

  __device__ __forceinline__ static void
  reduce_scalar(SumOp, ScalarT& a, const ScalarT& b) {
    a = __hadd(a, b);
  }

  __device__ __forceinline__ static void
  reduce_scalar(MaxOp, ScalarT& a, const ScalarT& b) {
    a = __hmax(a, b);
  }

  __device__ __forceinline__ static void
  reduce_scalar(MinOp, ScalarT& a, const ScalarT& b) {
    a = __hmin(a, b);
  }
#endif
};

template <>
struct VecOps<__nv_bfloat16>
    : HalfPrecisionVecOps<__nv_bfloat16, __nv_bfloat162> {};

template <>
struct VecOps<__half> : HalfPrecisionVecOps<__half, __half2> {};

// ============================================================================
// Storage policies
// ============================================================================

struct RegisterStorage {};

// ============================================================================
// Tile<T, kTileElems, kBlockSize>
//
// A fixed-size block of kTileElems elements of type T, cooperatively owned
// by kBlockSize threads. Each thread stores kVPT uint4 vectors in registers.
//
// Template parameters:
//   T            Element type (float, __half, __nv_bfloat16).
//   kTileElems   Total elements in the tile. Must be divisible by
//                kBlockSize * kElemsPerVec.
//   kBlockSize   Number of cooperating threads (default 256). Must match
//                the ThreadGroup's group_size at runtime.
//
// Derived constants:
//   kElemsPerVec = sizeof(uint4) / sizeof(T)     elements per vector
//   kTileVecs    = kTileElems / kElemsPerVec      vectors in the tile
//   kVPT         = kTileVecs / kBlockSize         vectors per thread
//
// Memory layout (thread k owns vectors at strided positions):
//   thread k → vecs[0] = global vec [k]
//              vecs[1] = global vec [k + kBlockSize]
//              vecs[j] = global vec [k + j * kBlockSize]
// ============================================================================

template <
    typename T,
    int kTileElems,
    int kBlockSize = 256,
    typename Storage = RegisterStorage>
struct Tile;

template <typename T, int kTileElems, int kBlockSize>
struct Tile<T, kTileElems, kBlockSize, RegisterStorage> {
  static constexpr int kElemsPerVec = VecOps<T>::kElemsPerVec;
  static constexpr int kTileVecs = kTileElems / kElemsPerVec;
  static constexpr int kVPT = kTileVecs / kBlockSize;
  static_assert(kVPT > 0, "kTileElems too small for kBlockSize");
  static_assert(
      kTileElems % (kBlockSize * kElemsPerVec) == 0,
      "kTileElems must be divisible by kBlockSize * kElemsPerVec");

  uint4 vecs[kVPT];
};

// ============================================================================
// tile_load — cooperative vectorized load from global memory into a tile.
//
// All threads in the group cooperate to load kTileElems elements starting at
// ptr[tile_idx * kTileElems]. Each thread loads kVPT vectors at strided
// positions: thread k loads vectors [k, k+kBlockSize, k+2*kBlockSize, ...].
//
// Partial-tile masking (Triton: tl.load with mask):
//   When valid_elems < kTileElems, elements beyond valid_elems are zeroed.
//   valid_elems need not be vector-aligned — the last partial vector is
//   loaded element-by-element and zero-padded.
//
// @param ptr          Source buffer (element-typed pointer).
// @param tile_idx     Which tile to load (0-based). Tile starts at
//                     ptr[tile_idx * kTileElems].
// @param group        ThreadGroup with group_size == kBlockSize.
// @param valid_elems  Number of valid elements (default = kTileElems).
//                     Elements beyond this are loaded as zero.
// @return             Register-backed tile with loaded data.
// ============================================================================

template <typename T, int kTileElems, int kBlockSize = 256>
__device__ __forceinline__ Tile<T, kTileElems, kBlockSize, RegisterStorage>
tile_load(
    const T* ptr,
    std::size_t tile_idx,
    const ThreadGroup& group,
    std::size_t valid_elems = kTileElems) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  using TileType = Tile<T, kTileElems, kBlockSize, RegisterStorage>;
  constexpr int kVPT = TileType::kVPT;
  constexpr int kTileVecs = TileType::kTileVecs;
  constexpr int kEPV = VecOps<T>::kElemsPerVec;

  TileType tile;
  if (reinterpret_cast<uintptr_t>(ptr) % alignof(uint4) != 0) {
    printf(
        "tile_load: ptr %p is not 16-byte aligned (required for uint4 vectorized access)\n",
        ptr);
    __trap();
  }
  const uint4* src = reinterpret_cast<const uint4*>(ptr) + tile_idx * kTileVecs;
  const std::size_t full_vecs = valid_elems / kEPV;
  const int remainder = static_cast<int>(valid_elems % kEPV);

#pragma unroll
  for (int k = 0; k < kVPT; k++) {
    std::size_t idx = group.thread_id_in_group + k * group.group_size;
    if (idx < full_vecs) {
      tile.vecs[k] = src[idx];
    } else if (idx == full_vecs && remainder > 0) {
      tile.vecs[k] = make_uint4(0, 0, 0, 0);
      const T* elem_src = ptr + tile_idx * kTileElems + idx * kEPV;
      T* elem_dst = reinterpret_cast<T*>(&tile.vecs[k]);
      for (int e = 0; e < remainder; e++) {
        elem_dst[e] = elem_src[e];
      }
    } else {
      tile.vecs[k] = make_uint4(0, 0, 0, 0);
    }
  }
  return tile;
#else
  return {};
#endif
}

// ============================================================================
// tile_store — cooperative vectorized store from tile to global memory.
//
// All threads cooperate to store kTileElems elements to
// ptr[tile_idx * kTileElems]. Partial-tile masking stores only the first
// valid_elems elements; the last partial vector is stored element-by-element.
//
// Triton equivalent: tl.store(ptr + offsets, tile, mask=mask)
//
// @param ptr          Destination buffer (element-typed pointer).
// @param tile_idx     Which tile slot to store into (0-based).
// @param tile         Register-backed tile to store.
// @param group        ThreadGroup with group_size == kBlockSize.
// @param valid_elems  Number of valid elements to store (default = kTileElems).
// ============================================================================

template <typename T, int kTileElems, int kBlockSize = 256>
__device__ __forceinline__ void tile_store(
    T* ptr,
    std::size_t tile_idx,
    const Tile<T, kTileElems, kBlockSize, RegisterStorage>& tile,
    const ThreadGroup& group,
    std::size_t valid_elems = kTileElems) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  using TileType = Tile<T, kTileElems, kBlockSize, RegisterStorage>;
  constexpr int kVPT = TileType::kVPT;
  constexpr int kTileVecs = TileType::kTileVecs;
  constexpr int kEPV = VecOps<T>::kElemsPerVec;

  if (reinterpret_cast<uintptr_t>(ptr) % alignof(uint4) != 0) {
    printf(
        "tile_store: ptr %p is not 16-byte aligned (required for uint4 vectorized access)\n",
        ptr);
    __trap();
  }
  uint4* dst = reinterpret_cast<uint4*>(ptr) + tile_idx * kTileVecs;
  const std::size_t full_vecs = valid_elems / kEPV;
  const int remainder = static_cast<int>(valid_elems % kEPV);

#pragma unroll
  for (int k = 0; k < kVPT; k++) {
    std::size_t idx = group.thread_id_in_group + k * group.group_size;
    if (idx < full_vecs) {
      dst[idx] = tile.vecs[k];
    } else if (idx == full_vecs && remainder > 0) {
      const T* elem_src = reinterpret_cast<const T*>(&tile.vecs[k]);
      T* elem_dst = ptr + tile_idx * kTileElems + idx * kEPV;
      for (int e = 0; e < remainder; e++) {
        elem_dst[e] = elem_src[e];
      }
    }
  }
#endif
}

// ============================================================================
// tile_accumulate — element-wise reduction between two register tiles.
//
// Applies Op element-wise: dst[i] = Op(dst[i], src[i]) for all elements.
// Operates entirely on registers — no memory access, no synchronization.
// Each thread reduces its own kVPT vectors independently.
//
// Triton equivalent:
//   SumOp → dst += src       MaxOp → dst = tl.maximum(dst, src)
//   MinOp → dst = tl.minimum(dst, src)
//
// @param dst  Destination tile (modified in-place).
// @param src  Source tile to accumulate from.
// ============================================================================

template <typename T, typename Op, int kTileElems, int kBlockSize = 256>
__device__ __forceinline__ void tile_accumulate(
    Tile<T, kTileElems, kBlockSize, RegisterStorage>& dst,
    const Tile<T, kTileElems, kBlockSize, RegisterStorage>& src) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  constexpr int kVPT = Tile<T, kTileElems, kBlockSize, RegisterStorage>::kVPT;
#pragma unroll
  for (int k = 0; k < kVPT; k++) {
    VecOps<T>::reduce(Op{}, dst.vecs[k], src.vecs[k]);
  }
#endif
}

// ============================================================================
// tile_load_accumulate — fused load + element-wise reduction.
//
// Loads a tile from global memory and reduces it into dst, without
// materializing the loaded tile in a separate register variable. Saves
// kVPT registers compared to separate tile_load + tile_accumulate.
//
// Triton equivalent: dst += tl.load(ptr + offsets, mask=mask)
//
// Partial-tile masking: elements beyond valid_elems are skipped (not
// accumulated). The last partial vector uses scalar element-wise reduction
// on only the valid elements, so this is correct for all ops (SumOp,
// MaxOp, MinOp).
//
// @param dst          Destination tile (accumulated in-place).
// @param ptr          Source buffer (element-typed pointer).
// @param tile_idx     Which tile to load (0-based).
// @param group        ThreadGroup with group_size == kBlockSize.
// @param valid_elems  Number of valid elements (default = kTileElems).
// ============================================================================

template <typename T, typename Op, int kTileElems, int kBlockSize = 256>
__device__ __forceinline__ void tile_load_accumulate(
    Tile<T, kTileElems, kBlockSize, RegisterStorage>& dst,
    const T* ptr,
    std::size_t tile_idx,
    const ThreadGroup& group,
    std::size_t valid_elems = kTileElems) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  using TileType = Tile<T, kTileElems, kBlockSize, RegisterStorage>;
  constexpr int kVPT = TileType::kVPT;
  constexpr int kTileVecs = TileType::kTileVecs;
  constexpr int kEPV = VecOps<T>::kElemsPerVec;

  if (reinterpret_cast<uintptr_t>(ptr) % alignof(uint4) != 0) {
    printf(
        "tile_load_accumulate: ptr %p is not 16-byte aligned (required for uint4 vectorized access)\n",
        ptr);
    __trap();
  }
  const uint4* src = reinterpret_cast<const uint4*>(ptr) + tile_idx * kTileVecs;
  const std::size_t full_vecs = valid_elems / kEPV;
  const int remainder = static_cast<int>(valid_elems % kEPV);

#pragma unroll
  for (int k = 0; k < kVPT; k++) {
    std::size_t idx = group.thread_id_in_group + k * group.group_size;
    if (idx < full_vecs) {
      uint4 v = src[idx];
      VecOps<T>::reduce(Op{}, dst.vecs[k], v);
    } else if (idx == full_vecs && remainder > 0) {
      const T* elem_src = ptr + tile_idx * kTileElems + idx * kEPV;
      T* elem_dst = reinterpret_cast<T*>(&dst.vecs[k]);
      for (int e = 0; e < remainder; e++) {
        VecOps<T>::reduce_scalar(Op{}, elem_dst[e], elem_src[e]);
      }
    }
  }
#endif
}

// ============================================================================
// tile_zero — zero-initialize all elements in a tile.
//
// Triton equivalent: tile = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
//
// Each thread zeros its own kVPT vectors. No synchronization needed.
//
// @param tile  Tile to zero (modified in-place).
// ============================================================================

template <typename T, int kTileElems, int kBlockSize = 256>
__device__ __forceinline__ void tile_zero(
    Tile<T, kTileElems, kBlockSize, RegisterStorage>& tile) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  constexpr int kVPT = Tile<T, kTileElems, kBlockSize, RegisterStorage>::kVPT;
#pragma unroll
  for (int k = 0; k < kVPT; k++) {
    tile.vecs[k] = make_uint4(0, 0, 0, 0);
  }
#endif
}

// ============================================================================
// tile_load_2d — cooperative load from a strided 2D region into a tile.
//
// Loads a kRows × kCols sub-matrix from a row-major matrix with the given
// stride (elements between consecutive rows). Returns a flat
// Tile<T, kRows*kCols, kBlockSize> — the 2D structure is only relevant for
// memory addressing. The returned tile is compatible with all 1D tile ops
// (tile_accumulate, tile_store, etc.).
//
// Triton equivalent:
//   offs_r = row_offset + tl.arange(0, ROWS)
//   offs_c = col_offset + tl.arange(0, COLS)
//   tl.load(ptr + offs_r[:,None]*stride + offs_c[None,:], mask=mask_2d)
//
// Thread-to-vector mapping:
//   kColVecs = kCols / kElemsPerVec (vectors per row)
//   Vector v → row = v / kColVecs, col_vec = v % kColVecs
//
// @param ptr          Base pointer to the 2D matrix (row-major).
// @param row_offset   Starting row index.
// @param col_offset   Starting column index.
// @param stride       Elements between consecutive rows.
// @param group        ThreadGroup with group_size == kBlockSize.
// @param valid_rows   Number of valid rows (default = kRows).
// @param valid_cols   Number of valid columns (default = kCols).
// @return             Flat tile with kRows*kCols elements.
// ============================================================================

template <typename T, int kRows, int kCols, int kBlockSize = 256>
__device__ __forceinline__ Tile<T, kRows * kCols, kBlockSize, RegisterStorage>
tile_load_2d(
    const T* ptr,
    std::size_t row_offset,
    std::size_t col_offset,
    std::size_t stride,
    const ThreadGroup& group,
    std::size_t valid_rows = kRows,
    std::size_t valid_cols = kCols) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  constexpr int kEPV = VecOps<T>::kElemsPerVec;
  constexpr int kColVecs = kCols / kEPV;
  static_assert(kCols % kEPV == 0, "kCols must be divisible by vector width");

  using TileType = Tile<T, kRows * kCols, kBlockSize, RegisterStorage>;
  constexpr int kVPT = TileType::kVPT;

  TileType tile;
  if (reinterpret_cast<uintptr_t>(ptr) % alignof(uint4) != 0) {
    printf(
        "tile_load_2d: ptr %p is not 16-byte aligned (required for uint4 vectorized access)\n",
        ptr);
    __trap();
  }
  if (col_offset % kEPV != 0) {
    printf(
        "tile_load_2d: col_offset %llu is not aligned to vector width %d\n",
        static_cast<unsigned long long>(col_offset),
        kEPV);
    __trap();
  }
  const std::size_t full_col_vecs = valid_cols / kEPV;
  const int col_rem = static_cast<int>(valid_cols % kEPV);

#pragma unroll
  for (int k = 0; k < kVPT; k++) {
    std::size_t v = group.thread_id_in_group + k * group.group_size;
    std::size_t row = v / kColVecs;
    std::size_t cv = v % kColVecs;

    if (row < valid_rows && cv < full_col_vecs) {
      const uint4* row_base = reinterpret_cast<const uint4*>(
          ptr + (row_offset + row) * stride + col_offset);
      tile.vecs[k] = row_base[cv];
    } else if (row < valid_rows && cv == full_col_vecs && col_rem > 0) {
      tile.vecs[k] = make_uint4(0, 0, 0, 0);
      const T* elem_src =
          ptr + (row_offset + row) * stride + col_offset + cv * kEPV;
      T* elem_dst = reinterpret_cast<T*>(&tile.vecs[k]);
      for (int e = 0; e < col_rem; e++) {
        elem_dst[e] = elem_src[e];
      }
    } else {
      tile.vecs[k] = make_uint4(0, 0, 0, 0);
    }
  }
  return tile;
#else
  return {};
#endif
}

// ============================================================================
// tile_store_2d — cooperative store from a tile into a strided 2D region.
//
// Stores a flat tile back into a kRows × kCols sub-matrix of a row-major
// matrix. Partial masking for valid_rows/valid_cols works the same as
// tile_load_2d.
//
// Triton equivalent:
//   tl.store(ptr + offs_r[:,None]*stride + offs_c[None,:], tile, mask=mask_2d)
//
// @param ptr          Base pointer to the destination matrix (row-major).
// @param row_offset   Starting row index.
// @param col_offset   Starting column index.
// @param stride       Elements between consecutive rows.
// @param tile         Flat tile to store.
// @param group        ThreadGroup with group_size == kBlockSize.
// @param valid_rows   Number of valid rows (default = kRows).
// @param valid_cols   Number of valid columns (default = kCols).
// ============================================================================

template <typename T, int kRows, int kCols, int kBlockSize = 256>
__device__ __forceinline__ void tile_store_2d(
    T* ptr,
    std::size_t row_offset,
    std::size_t col_offset,
    std::size_t stride,
    const Tile<T, kRows * kCols, kBlockSize, RegisterStorage>& tile,
    const ThreadGroup& group,
    std::size_t valid_rows = kRows,
    std::size_t valid_cols = kCols) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  constexpr int kEPV = VecOps<T>::kElemsPerVec;
  constexpr int kColVecs = kCols / kEPV;
  static_assert(kCols % kEPV == 0, "kCols must be divisible by vector width");

  using TileType = Tile<T, kRows * kCols, kBlockSize, RegisterStorage>;
  constexpr int kVPT = TileType::kVPT;

  if (reinterpret_cast<uintptr_t>(ptr) % alignof(uint4) != 0) {
    printf(
        "tile_store_2d: ptr %p is not 16-byte aligned (required for uint4 vectorized access)\n",
        ptr);
    __trap();
  }
  if (col_offset % kEPV != 0) {
    printf(
        "tile_store_2d: col_offset %llu is not aligned to vector width %d\n",
        static_cast<unsigned long long>(col_offset),
        kEPV);
    __trap();
  }
  const std::size_t full_col_vecs = valid_cols / kEPV;
  const int col_rem = static_cast<int>(valid_cols % kEPV);

#pragma unroll
  for (int k = 0; k < kVPT; k++) {
    std::size_t v = group.thread_id_in_group + k * group.group_size;
    std::size_t row = v / kColVecs;
    std::size_t cv = v % kColVecs;

    if (row < valid_rows && cv < full_col_vecs) {
      uint4* row_base = reinterpret_cast<uint4*>(
          ptr + (row_offset + row) * stride + col_offset);
      row_base[cv] = tile.vecs[k];
    } else if (row < valid_rows && cv == full_col_vecs && col_rem > 0) {
      const T* elem_src = reinterpret_cast<const T*>(&tile.vecs[k]);
      T* elem_dst = ptr + (row_offset + row) * stride + col_offset + cv * kEPV;
      for (int e = 0; e < col_rem; e++) {
        elem_dst[e] = elem_src[e];
      }
    }
  }
#endif
}

// ============================================================================
// Helpers
// ============================================================================

// Number of full tiles that fit in nelems elements.
// Triton equivalent: nelems // kTileElems
template <int kTileElems>
__device__ __forceinline__ std::size_t num_tiles(std::size_t nelems) {
  return nelems / kTileElems;
}

// Remainder elements after full tiles (for partial-tile masking).
// Triton equivalent: nelems % kTileElems
template <int kTileElems>
__device__ __forceinline__ std::size_t tile_remainder(std::size_t nelems) {
  return nelems % kTileElems;
}

} // namespace comms::pipes
