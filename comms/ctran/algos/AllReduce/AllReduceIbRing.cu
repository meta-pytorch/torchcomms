// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PRIMS)

#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceFused.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceFusedCommon.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceIbRing.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceNvlDirect.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

using namespace ctran::allreduce::common;

namespace {

// Non-negative (base - s) mod W.
__device__ __forceinline__ int ringRotate(int base, int s, int W) {
  return ((base - s) % W + W) % W;
}

// Largest of the W node sub-shards -- the pipeline-loop upper bound. Sub-shards
// are a TiledBuffer over the block tile, so every offset is 16B-spaced and each
// non-empty interior sub-shard is exactly tile_elements; at most one non-empty
// sub-shard (the last) may carry a short/non-16B tail.
__device__ __forceinline__ size_t
maxShardTileBytes(const comms::prims::TiledBuffer<char>& shardTiles, int W) {
  size_t m = 0;
  for (int k = 0; k < W; k++) {
    const size_t b = shardTiles.tile_bytes(k);
    m = b > m ? b : m;
  }
  return m;
}

// Valid bytes of sub-shard `k` at pipeline offset `off` (0 if past the end).
__device__ __forceinline__ size_t shardWindow(
    const comms::prims::TiledBuffer<char>& shardTiles,
    int k,
    size_t off,
    size_t pipelineWindow) {
  const size_t b = shardTiles.tile_bytes(k);
  if (off >= b) {
    return 0;
  }
  const size_t rem = b - off;
  return pipelineWindow < rem ? pipelineWindow : rem;
}

// Phase-2 ring reduce-scatter over the W node sub-shards of this block's tile.
// Per pipeline window: send(shard me) + forward(shard me-1 .. me-(W-2)) +
// recv(shard me+1). Each forward reduces this node's local sub-shard into the
// running partial via IbReduceCopy<T> and passes it on; the final recv lands
// the fully reduced sub-shard (me+1). Other sub-shards keep this node's local
// data until all-gather fills them in. `ringGroup` carries the per-block
// group_id with total_groups relabeled to the fixed IB reservation, and
// `ibSendRecvGroups` is that same reservation passed as the transport
// `active_blocks`.
template <typename T>
__device__ __forceinline__ void reduceScatterRing(
    comms::prims::P2pIbgdaTransportDevice& prev,
    comms::prims::P2pIbgdaTransportDevice& next,
    comms::prims::ThreadGroup& ringGroup,
    const comms::prims::TiledBuffer<char>& shardTiles,
    size_t maxShardBytes,
    int W,
    int me,
    int ibSendRecvGroups,
    size_t pipelineWindow,
    const comms::prims::Timeout& timeout) {
  for (size_t off = 0; off < maxShardBytes; off += pipelineWindow) {
    // step 0: initiate own owner sub-shard (raw local data).
    {
      const int k = me;
      const size_t v = shardWindow(shardTiles, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = shardTiles.tile_data(k) + off;
        next.send(ringGroup, buf, v, ibSendRecvGroups, 0, timeout);
      }
    }
    // steps 1..W-2: receive partial for shard (me-s), reduce local, forward.
    for (int s = 1; s <= W - 2; ++s) {
      const int k = ringRotate(me, s, W);
      const size_t v = shardWindow(shardTiles, k, off, pipelineWindow);
      if (v > 0) {
        char* lin = shardTiles.tile_data(k) + off;
        prev.template forward<IbReduceCopy<T>>(
            ringGroup, nullptr, next, v, ibSendRecvGroups, 0, timeout, lin);
      }
    }
    // step W-1: final receive-reduce of shard (me+1) into the tile.
    {
      const int k =
          ringRotate(me, W - 1, W); // (me - (W-1)) mod W = (me+1) mod W
      const size_t v = shardWindow(shardTiles, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = shardTiles.tile_data(k) + off;
        prev.template recv<IbReduceCopy<T>>(
            ringGroup, buf, v, ibSendRecvGroups, 0, timeout);
      }
    }
  }
}

// Phase-2 ring all-gather over the W node sub-shards. Per window: send(shard
// me+1) + forward(shard me .. me+1-(W-2), storing each) + recv(shard me+2).
// All-gather never reduces (uses the transport's default Memcpy CopyOp). Args
// mirror reduceScatterRing.
template <typename T>
__device__ __forceinline__ void allGatherRing(
    comms::prims::P2pIbgdaTransportDevice& prev,
    comms::prims::P2pIbgdaTransportDevice& next,
    comms::prims::ThreadGroup& ringGroup,
    const comms::prims::TiledBuffer<char>& shardTiles,
    size_t maxShardBytes,
    int W,
    int me,
    int ibSendRecvGroups,
    size_t pipelineWindow,
    const comms::prims::Timeout& timeout) {
  for (size_t off = 0; off < maxShardBytes; off += pipelineWindow) {
    // step 0: broadcast own reduced sub-shard (me+1).
    {
      const int k = ringRotate(me + 1, 0, W);
      const size_t v = shardWindow(shardTiles, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = shardTiles.tile_data(k) + off;
        next.send(ringGroup, buf, v, ibSendRecvGroups, 0, timeout);
      }
    }
    // steps 1..W-2: receive shard (me+1-s), store into the tile, forward.
    for (int s = 1; s <= W - 2; ++s) {
      const int k = ringRotate(me + 1, s, W);
      const size_t v = shardWindow(shardTiles, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = shardTiles.tile_data(k) + off;
        prev.forward(ringGroup, buf, next, v, ibSendRecvGroups, 0, timeout);
      }
    }
    // step W-1: final receive of shard (me+2) into the tile.
    {
      const int k = ringRotate(me + 1, W - 1, W); // (me+2) mod W
      const size_t v = shardWindow(shardTiles, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = shardTiles.tile_data(k) + off;
        prev.recv(ringGroup, buf, v, ibSendRecvGroups, 0, timeout);
      }
    }
  }
}

// Phase-2 inter-node ring (reduce-scatter then all-gather) over this block's
// owner-segment tile.
//
// Group invariant (data vs staging): `blockGroup` (total_groups = numBlocks)
// governs data ownership -- both the block tile and its sub-shard TiledBuffer
// are derived from it. `ringGroup` reuses the per-block group_id but relabels
// total_groups to the FIXED `args.ibSendRecvGroups` reservation, used ONLY for
// transport staging/signaling identity. Never tile data with `ringGroup`:
// TiledBuffer tiles by group.total_groups, so tiling with the (larger, fixed)
// reservation would leave data uncovered by the launched blocks.
template <typename T>
__device__ __noinline__ void phase2IbRing(
    const ctran::allreduce::hierring::KernArgs& args,
    comms::prims::ThreadGroup& blockGroup) {
  if (args.common.localRank >= args.common.pMin) {
    return;
  }
  if (args.common.nNodes <= 1) {
    return;
  }

  const int W = args.ring.nNodes;
  const int me = args.ring.myNodeIdx;
  const size_t actualElems = actualSegElems(
      args.common.count, args.common.segmentElems, args.common.localRank);

  // Data ownership: this block's tile of phase2Buf (from blockGroup), then
  // split into W node sub-shards with 16B-aligned offsets. blockTile is
  // group-bound so blockTile.data()/bytes() are valid; shardTiles is a plain
  // W-way split so use only tile_data(k)/tile_bytes(k).
  comms::prims::TiledBuffer<char> blockTile(
      static_cast<char*>(args.common.phase2Buf),
      actualElems * sizeof(T),
      blockGroup);

  if (blockTile.bytes() == 0) {
    // Blocks whose tile is empty (tiny counts, trailing blocks) do no work.
    return;
  }

  comms::prims::TiledBuffer<char> shardTiles(
      blockTile.data(), blockTile.bytes(), W);

  auto& prev = *args.common.transports[args.ring.prevRank].p2p_ib.ibgda;
  auto& next = *args.common.transports[args.ring.nextRank].p2p_ib.ibgda;

  // Fixed IB staging/signaling reservation, independent of numBlocks so the
  // transport staging layout is stable across launches (see
  // AllReduceIbRing.cc).
  const int ibSendRecvGroups = args.ibSendRecvGroups;
  auto ringGroup = blockGroup;
  ringGroup.total_groups = static_cast<uint32_t>(ibSendRecvGroups);

  const size_t pipelineWindow = next.pipeline_window(ibSendRecvGroups);
  PIPES_DEVICE_CHECK_MSG(
      pipelineWindow != 0, "phase2IbRing: pipeline window is zero");

  // The block tile is a view into recvbuff; the AllReduce API guarantees only
  // element alignment (alignof(T)), which is all we require. The CopyOps
  // (IbReduceCopy's tileReduce and the transport's default Memcpy) take the 16B
  // vectorized path when aligned and a scalar path otherwise, and the transport
  // rounds the wire byte count to 16B internally (D108444748), so no manual
  // rounding/valid-masking is needed. Sub-shard offsets inherit this alignment
  // (tile_elements is 16B-rounded).
  PIPES_DEVICE_CHECK_MSG(
      reinterpret_cast<uintptr_t>(blockTile.data()) % alignof(T) == 0,
      "phase2IbRing: tile must be element-aligned");

  const size_t maxShardBytes = maxShardTileBytes(shardTiles, W);
  comms::prims::Timeout timeout{};

  reduceScatterRing<T>(
      prev,
      next,
      ringGroup,
      shardTiles,
      maxShardBytes,
      W,
      me,
      ibSendRecvGroups,
      pipelineWindow,
      timeout);

  // Block-local phase transition: shard (me+1) is now reduced in the tile and
  // is read by all-gather step 0.
  blockGroup.sync();

  allGatherRing<T>(
      prev,
      next,
      ringGroup,
      shardTiles,
      maxShardBytes,
      W,
      me,
      ibSendRecvGroups,
      pipelineWindow,
      timeout);
}

} // namespace

__global__
__launch_bounds__(ctran::allreduce::hierring::kBlockSize, 1) void ctranKernelAllReduceHierarchicalRing(
    int* /* flag */,
    CtranAlgoDeviceState* /* devState */,
    ctran::allreduce::hierring::KernArgs args) {
  // Maps the physical block to a logical data group.
  auto blockGroup = comms::prims::make_block_group();
  const int blockId = static_cast<int>(blockIdx.x);
  if (blockId >= args.common.numBlocks) {
    return;
  }
  auto group = logicalDataGroup(blockGroup, blockId, args.common.numBlocks);

  if (args.common.datatype == commFloat32) {
    ctran::allreduce::fused::
        runAllReduceFused<float, ctran::allreduce::nvl::direct::Ops>(
            args.common, group, [&](comms::prims::ThreadGroup& phaseGroup) {
              phase2IbRing<float>(args, phaseGroup);
            });
  }
}

#endif // ENABLE_PRIMS
