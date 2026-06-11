// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if defined(ENABLE_PIPES)

#include "comms/ctran/algos/AllReduce/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/topo/TreeConstants.h"
#include "comms/utils/commSpecs.h"

namespace comms::prims {
struct Transport;
}

namespace ctran::allreduce::tree {

/** Number of independent inter-node tree lanes in the dual-tree phase. */
static constexpr int kTreeLanes = 2;

/** CUDA threads per block used by the ctree kernel. */
static constexpr int kBlockSize = 640;

/**
 * Device kernel arguments for the Prims-backed CTRAN tree AllReduce.
 *
 * User buffers are owned by the caller. NVL and IB receive staging are owned by
 * the Prims transports and are consumed only inside transport copy callbacks.
 */
struct KernArgs {
  /** User input buffer; may alias `recvbuff` for in-place AllReduce. */
  const void* sendbuff;
  /** User output buffer that receives the final reduced tensor. */
  void* recvbuff;
  /**
   * Phase 2 working buffer.
   *
   * Segment owners write locally reduced segments here in Phase 1, the IB tree
   * reads and writes the same buffer in Phase 2, and Phase 3 reads from it.
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

  /** Collective datatype implemented by the kernel dispatch. */
  commDataType_t datatype;
  /** Reduction operation implemented by the kernel dispatch. */
  commRedOp_t redOp;

  /** Unified transport array containing NVL and IB transports by global rank.
   */
  comms::prims::Transport* transports;

  /** Inter-node tree used for the first half of each data partition. */
  ctran::allreduce::TreeTopology tree0;
  /** Inter-node tree used for the second half of each data partition. */
  ctran::allreduce::TreeTopology tree1;

  /** Maps local rank index `[0, nLocalRanks)` to global communicator rank. */
  int localRankToGlobalRank[CTRAN_MAX_NVL_PEERS];
};

} // namespace ctran::allreduce::tree

/**
 * CUDA kernel entry point for CTRAN tree AllReduce.
 *
 * Each block owns one tile partition and runs Phase 1, both Phase 2 tree
 * lanes, and Phase 3 for that tile. Phase 2 uses one full-block work group to
 * poll and progress the two disjoint IB tree lanes cooperatively.
 * Multi-block launches are independent tiling for performance; correctness
 * does not require inter-block sync.
 */
__global__ void ctranKernelAllReduceTree(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::tree::KernArgs args);

#endif // ENABLE_PIPES
