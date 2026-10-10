/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>

namespace rcclx::hier {

// Chunk boundaries land on whole multiples of this many elements, so every
// send/recv buffer starts at least 512-byte aligned for 1-byte types.
inline constexpr size_t kAlignElems = 512;

struct Piece {
  size_t offset{0};
  size_t count{0};
};

// Piece `i` of `n` over [0, count). Every boundary is a multiple of `align`;
// the last piece absorbs the remainder.
constexpr Piece splitEven(size_t count, int n, int i, size_t align) {
  const size_t unit = count / static_cast<size_t>(n) / align * align;
  const size_t offset = static_cast<size_t>(i) * unit;
  return Piece{offset, i == n - 1 ? count - offset : unit};
}

enum class InterAlgo {
  // Two nodes: each sends its whole node-partial shard to the other and both
  // reduce. Same bytes as reduce-scatter + all-gather, one stage fewer.
  Exchange,
  // N > 2 nodes: direct reduce-scatter then all-gather across nodes.
  ReduceScatterAllGather,
};

/**
 * Buffer layout of one hierarchical allreduce.
 *
 * [0, count) is cut into nTiles pipeline tiles. Each tile is cut into nLocal
 * shards (local rank j owns shard j) and each shard into nNodes sub-shards
 * (node m owns sub-shard m). Offsets are absolute element offsets into the
 * user buffer. Every value is a pure function of (count, nNodes, nLocal,
 * nTiles), so all ranks agree without communicating.
 */
struct Geometry {
  size_t count{0};
  int nNodes{0};
  int nLocal{0};
  int nTiles{0};

  constexpr size_t minUnit() const {
    return static_cast<size_t>(nNodes) * static_cast<size_t>(nLocal) *
        kAlignElems;
  }

  constexpr InterAlgo inter() const {
    return nNodes == 2 ? InterAlgo::Exchange
                       : InterAlgo::ReduceScatterAllGather;
  }

  constexpr Piece tile(int t) const {
    return splitEven(count, nTiles, t, minUnit());
  }

  constexpr Piece shard(int t, int j) const {
    const Piece tp = tile(t);
    const Piece sp = splitEven(
        tp.count, nLocal, j, static_cast<size_t>(nNodes) * kAlignElems);
    return Piece{tp.offset + sp.offset, sp.count};
  }

  constexpr Piece sub(int t, int j, int m) const {
    const Piece sp = shard(t, j);
    const Piece mp = splitEven(sp.count, nNodes, m, kAlignElems);
    return Piece{sp.offset + mp.offset, mp.count};
  }

  // Largest shard j across tiles; sizes the intra-node staging.
  size_t maxShard(int j) const {
    size_t m = 0;
    for (int t = 0; t < nTiles; t++) {
      m = std::max(m, shard(t, j).count);
    }
    return m;
  }

  // Largest sub-shard (j, node) across tiles; sizes the inter-node staging.
  size_t maxSub(int j, int node) const {
    size_t m = 0;
    for (int t = 0; t < nTiles; t++) {
      m = std::max(m, sub(t, j, node).count);
    }
    return m;
  }
};

// Every sub-shard must hold at least kAlignElems elements.
constexpr bool geometryFeasible(size_t count, int nNodes, int nLocal) {
  return nNodes > 0 && nLocal > 0 &&
      count >=
      static_cast<size_t>(nNodes) * static_cast<size_t>(nLocal) * kAlignElems;
}

// ceil(bytes / tileBytes), clamped to [1, maxTiles] and to the number of
// minimum-size tiles the count can hold.
constexpr int chooseTiles(
    size_t count,
    size_t elemSize,
    int nNodes,
    int nLocal,
    size_t tileBytes,
    int maxTiles) {
  const size_t bytes = count * elemSize;
  const size_t want = tileBytes == 0 ? 1 : (bytes + tileBytes - 1) / tileBytes;
  const size_t fit = count /
      (static_cast<size_t>(nNodes) * static_cast<size_t>(nLocal) * kAlignElems);
  const size_t t =
      std::min({want, fit, static_cast<size_t>(std::max(1, maxTiles))});
  return static_cast<int>(std::max<size_t>(1, t));
}

} // namespace rcclx::hier
