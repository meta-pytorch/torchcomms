// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "comms/ctran/algos/CtranAlgoDev.h" // CTRAN_MAX_NVL_PEERS
#include "comms/utils/commSpecs.h" // commDataType_t, commRedOp_t

namespace ctran::allreduce::common {

/** CUDA threads per block used by the fused AllReduce kernels. */
static constexpr int kBlockSize = 640;
/** Elements processed by one NVL tile operation in the reduction helpers. */
static constexpr int kNvlTileElems = 15360;
/** Elements processed by one IB tile operation in the reduction helpers. */
static constexpr int kIbTileElems = 5120;

/** Byte granularity required for CTREE block and tree-lane partition starts. */
static constexpr std::size_t kTileParitionAligmentBytes = 128;

/**
 * Return the element count for one aligned tile partition.
 *
 * The returned size is the `ceil(numElems / numPartitions)` element count
 * rounded up so `partitionElems * elemBytes` lands on a
 * `kTileParitionAligmentBytes` boundary. It can be larger than `numElems`;
 * callers consume the final partition as a tail whose actual size is at most
 * this value.
 */
__host__ __device__ inline std::size_t compute_aligned_tile_parition_size(
    std::size_t numElems,
    std::size_t elemBytes,
    int numPartitions) {
  if (numElems == 0 || elemBytes == 0 || numPartitions <= 1) {
    return numElems;
  }

  std::size_t a = kTileParitionAligmentBytes;
  std::size_t b = elemBytes;
  while (b != 0) {
    const std::size_t r = a % b;
    a = b;
    b = r;
  }

  const std::size_t elemsPerAlignedBoundary = kTileParitionAligmentBytes / a;
  const std::size_t nPartitions = static_cast<std::size_t>(numPartitions);
  const std::size_t partitionElems = (numElems + nPartitions - 1) / nPartitions;
  return ((partitionElems + elemsPerAlignedBoundary - 1) /
          elemsPerAlignedBoundary) *
      elemsPerAlignedBoundary;
}

} // namespace ctran::allreduce::common

namespace ctran::allreduce::nvl::direct {

/**
 * Per-NVL-variant device-state slot embedded in each algorithm's `KernArgs`.
 *
 * Empty for NvlDirect, which needs no extra device state. A future NvlSharp
 * variant defines its own `ExtraArgs` (multicast/SHARP handles) so that state
 * has a home without bloating `common::CommonKernArgs`. Defined here
 * (host-safe, no `__device__` code) so `KernArgs` headers stay includable from
 * host `.cc`.
 */
struct ExtraArgs {};

} // namespace ctran::allreduce::nvl::direct

#if defined(ENABLE_PRIMS)

namespace comms::prims {
struct Transport;
}

namespace ctran::allreduce::common {

/**
 * Topology-agnostic device kernel arguments shared by all fused AllReduce
 * algorithms.
 *
 * The fused AllReduce kernels are structured as NVL ReduceScatter (Phase 1), a
 * cross-node IB phase (Phase 2), and NVL AllGather (Phase 3). Phase 1 and Phase
 * 3 are identical across algorithms; only Phase 2 differs (dual-tree, ring,
 * ...). These fields are everything those shared phases and the kernel entry
 * need. Each algorithm layers its own Phase-2 topology (e.g. dual trees or ring
 * neighbors) on top by embedding this struct in its own KernArgs.
 *
 * User buffers are owned by the caller. NVL and IB receive staging are owned by
 * the Prims transports and are consumed only inside transport copy callbacks.
 */
struct CommonKernArgs {
  /** User input buffer; may alias `recvbuff` for in-place AllReduce. */
  const void* sendbuff;
  /** User output buffer that receives the final reduced tensor. */
  void* recvbuff;
  /**
   * Phase 2 working buffer.
   *
   * Segment owners write locally reduced segments here in Phase 1, the Phase 2
   * cross-node implementation reads and writes the same buffer, and Phase 3
   * reads from it.
   */
  void* phase2Buf;

  /** Total number of elements in the user tensor. */
  size_t count;
  /** Elements per owner segment, equal to `ceil(count / pMin)`. */
  size_t segmentElems;

  /** Number of nodes in the communicator. */
  int nNodes;
  /** Minimum number of local ranks across all nodes. */
  int pMin;
  /** Number of local ranks on this node. */
  int nLocalRanks;
  /** This GPU's local rank on its node. */
  int localRank;
  /** Number of logical data partitions processed by CUDA blocks per GPU. */
  int numBlocks;
  /**
   * Byte stride for one block-owned segment partition.
   *
   * Computed once on the launch path from `segmentElems`, datatype size, and
   * `numBlocks`; each block uses this stride to derive its segment offset, and
   * the final block consumes a tail no larger than this value.
   */
  size_t blockTileBytes;

  /** Collective datatype implemented by the kernel dispatch. */
  commDataType_t datatype;
  /** Reduction operation implemented by the kernel dispatch. */
  commRedOp_t redOp;

  /** Unified transport array containing NVL and IB transports by global rank.
   */
  comms::prims::Transport* transports;

  /** Maps local rank index `[0, nLocalRanks)` to global communicator rank. */
  int localRankToGlobalRank[CTRAN_MAX_NVL_PEERS];
};

} // namespace ctran::allreduce::common

#endif // ENABLE_PRIMS
