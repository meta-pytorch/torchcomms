// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#if defined(ENABLE_PRIMS)

#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceFused.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceFusedCommon.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceIbRing.cuh"
#include "comms/ctran/algos/AllReduce/AllReduceNvlDirect.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

using namespace ctran::allreduce::common;

namespace {

// Bytes of `nbytes` that fall within the valid window `[byte_offset,
// valid_window)` (0 if the window starts at/after valid_window). Shared by the
// valid-masked send/recv/forward paths below.
__device__ __forceinline__ size_t
validClampBytes(size_t nbytes, size_t byte_offset, size_t valid_window) {
  if (byte_offset >= valid_window) {
    return 0;
  }
  const size_t rem = valid_window - byte_offset;
  return nbytes < rem ? nbytes : rem;
}

template <typename T>
struct ValidReduceStaged {
  template <typename... Args>
  __device__ __forceinline__ static void send(
      char* staging,
      const char* src,
      size_t nbytes,
      comms::prims::ThreadGroup& group,
      size_t byte_offset,
      const char*,
      size_t valid_window,
      Args...) {
    size_t copyBytes = validClampBytes(nbytes, byte_offset, valid_window);
    if (copyBytes > 0) {
      comms::prims::memcpy_vectorized(staging, src, copyBytes, group);
    }
  }

  template <typename... Args>
  __device__ __forceinline__ static void recv(
      char* dst,
      const char* staging,
      size_t nbytes,
      comms::prims::ThreadGroup& group,
      size_t byte_offset,
      const char* local_input,
      size_t valid_window,
      Args...) {
    size_t validN =
        validClampBytes(nbytes, byte_offset, valid_window) / sizeof(T);
    T* d = reinterpret_cast<T*>(dst);
    const T* s = reinterpret_cast<const T*>(staging);
    const T* l = reinterpret_cast<const T*>(local_input + byte_offset);
    for (size_t i = group.thread_id_in_group; i < validN;
         i += group.group_size) {
      d[i] = s[i] + l[i];
    }
  }

  template <typename... Args>
  __device__ __forceinline__ static void forward(
      char*,
      char* fwd_staging,
      const char* staging,
      size_t nbytes,
      comms::prims::ThreadGroup& group,
      size_t byte_offset,
      const char* local_input,
      size_t valid_window,
      Args...) {
    size_t validN =
        validClampBytes(nbytes, byte_offset, valid_window) / sizeof(T);
    T* f = reinterpret_cast<T*>(fwd_staging);
    const T* s = reinterpret_cast<const T*>(staging);
    const T* l = reinterpret_cast<const T*>(local_input + byte_offset);
    for (size_t i = group.thread_id_in_group; i < validN;
         i += group.group_size) {
      f[i] = s[i] + l[i];
    }
  }
};

struct ValidMemcpy {
  template <typename... Args>
  __device__ __forceinline__ static void send(
      char* staging,
      const char* src,
      size_t nbytes,
      comms::prims::ThreadGroup& group,
      size_t byte_offset,
      size_t valid_window,
      Args...) {
    size_t copyBytes = validClampBytes(nbytes, byte_offset, valid_window);
    if (copyBytes > 0) {
      comms::prims::memcpy_vectorized(staging, src, copyBytes, group);
    }
  }

  template <typename... Args>
  __device__ __forceinline__ static void recv(
      char* dst,
      const char* staging,
      size_t nbytes,
      comms::prims::ThreadGroup& group,
      size_t byte_offset,
      size_t valid_window,
      Args...) {
    size_t copyBytes = validClampBytes(nbytes, byte_offset, valid_window);
    if (copyBytes > 0) {
      comms::prims::memcpy_vectorized(dst, staging, copyBytes, group);
    }
  }

  template <typename... Args>
  __device__ __forceinline__ static void forward(
      char* dst,
      char* fwd_staging,
      const char* staging,
      size_t nbytes,
      comms::prims::ThreadGroup& group,
      size_t byte_offset,
      size_t valid_window,
      Args...) {
    size_t copyBytes = validClampBytes(nbytes, byte_offset, valid_window);
    if (copyBytes > 0) {
      if (dst) {
        comms::prims::memcpy_vectorized(
            dst, fwd_staging, staging, copyBytes, group);
      } else {
        comms::prims::memcpy_vectorized(fwd_staging, staging, copyBytes, group);
      }
    }
  }
};

// One of W node sub-shards within a block's owner-segment tile. TiledBuffer
// rounds tile_elements up to 16B, so every sub-shard offset is 16B-spaced and
// each non-empty interior sub-shard is exactly tile_elements. At most one
// non-empty sub-shard (the last non-empty one) may be short/non-16B; for tiny
// tiles that can be sub-shard 0 (with the remaining sub-shards empty), not
// necessarily sub-shard W-1.
struct RingShard {
  size_t offsetBytes;
  size_t bytes;
};

__device__ __forceinline__ RingShard
ringShard(size_t tileBytes, int W, int shardIdx) {
  comms::prims::TiledBuffer<char> sh(nullptr, tileBytes, W);
  return RingShard{
      .offsetBytes = static_cast<size_t>(shardIdx) * sh.tile_elements,
      .bytes = sh.tile_bytes(shardIdx),
  };
}

// Largest of the W sub-shards — the pipeline-loop upper bound.
__device__ __forceinline__ size_t maxRingShardBytes(size_t tileBytes, int W) {
  size_t m = 0;
  for (int k = 0; k < W; k++) {
    const size_t b = ringShard(tileBytes, W, k).bytes;
    m = b > m ? b : m;
  }
  return m;
}

// Non-negative (base - s) mod W.
__device__ __forceinline__ int ringRotate(int base, int s, int W) {
  return ((base - s) % W + W) % W;
}

// Valid bytes of sub-shard `k` at pipeline offset `off` (0 if past the end).
__device__ __forceinline__ size_t
shardWindow(size_t tileBytes, int W, int k, size_t off, size_t pipelineWindow) {
  const size_t b = ringShard(tileBytes, W, k).bytes;
  if (off >= b) {
    return 0;
  }
  const size_t rem = b - off;
  return pipelineWindow < rem ? pipelineWindow : rem;
}

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
  const auto tile = segmentTile(actualElems * sizeof(T), blockGroup);
  const size_t tileBytes = tile.bytes;

  if (tileBytes == 0) {
    // Block whoses tile/tail partition are empty (tiny counts, trailing blocks,
    // exhausted shards) do no Phase-2 work.
    return;
  }

  auto& prev = *args.common.transports[args.ring.prevRank].p2p_ib.ibgda;
  auto& next = *args.common.transports[args.ring.nextRank].p2p_ib.ibgda;
  const int activeGroups =
      args.common.numBlocks * ctran::allreduce::hierring::kRingLanes;

  // Cross-node pairing maps block b on each node to ring group b: ringGroup
  // carries the per-block group_id and the lane dimension is NOT folded into
  // it. This is only correct for a single ring lane; raising kRingLanes would
  // require reworking group_id so its id space spans [0, activeGroups).
  static_assert(
      ctran::allreduce::hierring::kRingLanes == 1,
      "phase2IbRing assumes kRingLanes == 1 for cross-node ring-group pairing; "
      "raising it requires folding the lane dimension into ringGroup.group_id");
  auto ringGroup = blockGroup;
  ringGroup.group_id = blockGroup.group_id;
  ringGroup.total_groups = static_cast<uint32_t>(activeGroups);

  const size_t pipelineWindow = next.pipeline_window(activeGroups);
  PIPES_DEVICE_CHECK_MSG(
      pipelineWindow != 0, "phase2IbRing: pipeline window is zero");

  char* tileBuf = static_cast<char*>(args.common.phase2Buf) + tile.offsetBytes;
  comms::prims::Timeout timeout{};

  // tileBuf is a view into the user's recvbuff, so it inherits recvbuff's
  // alignment. The AllReduce API only guarantees element alignment
  // (alignof(T)), NOT 16-byte alignment.
  //
  // Element alignment is all we require, and all we assert below: it is the
  // minimum the valid-masking path needs for its typed T* loads. We do NOT
  // require 16-byte alignment, because the fast uint4/TileReduceStaged path is
  // opt-in per window via the isAligned16(buf) check in the fork below. A
  // tileBuf that is element- but not 16-aligned therefore just routes every
  // window through the valid-masking path (correct, only slightly slower).
  PIPES_DEVICE_CHECK_MSG(
      reinterpret_cast<uintptr_t>(tileBuf) % alignof(T) == 0,
      "phase2IbRing: tileBuf must be element-aligned");

  const size_t maxShardBytes = maxRingShardBytes(tileBytes, W);

  // Peer-symmetry invariant (correctness-critical): the per-window fork below
  // is `v % 16 == 0 && isAligned16(buf)`, but the WIRE byte count it issues
  // depends only on `v % 16` (a pure function of shard size, hence globally
  // agreed): the fast path uses nbytes=v and the valid path uses
  // nbytes=roundUp16(v), which are equal when v%16==0, and when v%16!=0 BOTH
  // peers take the valid path. So even if two ranks differ in buffer alignment
  // (see the tileBuf alignment note above), the sender and receiver of a
  // sub-shard always agree on the wire byte count, keeping the persistent
  // stepState cursors in lock-step.
  //
  // Shard-size facts used below: among non-empty sub-shards at most one (the
  // last non-empty one) is non-16 in size; interior non-empty sub-shards are
  // exactly tile_elements. For tiny tiles that short shard can be sub-shard 0
  // (with the remaining sub-shards empty), not necessarily sub-shard W-1. And
  // wire <= pipelineWindow always (v <= pipelineWindow; pipelineWindow is
  // 16-aligned, so roundUp16(v) <= pipelineWindow), so no transfer exceeds the
  // staging ring (no deadlock). All control flow below is uniform across the
  // block's threads.
  using ReduceOp = comms::prims::TileReduceStaged<
      T,
      comms::prims::SumOp,
      kIbTileElems,
      ctran::allreduce::hierring::kBlockSize>;

  // ============================ Reduce-scatter ============================
  // Per window: send(shard me) + forward(shard me-1 .. me-(W-2)) + recv(shard
  // me+1). Each forward reduces this node's local sub-shard into the running
  // partial and passes it on; the final recv lands the fully reduced sub-shard
  // (me+1) in tileBuf. Other sub-shards in tileBuf keep this node's local data
  // until all-gather fills them in.
  for (size_t off = 0; off < maxShardBytes; off += pipelineWindow) {
    // step 0: initiate own owner sub-shard (raw local data).
    {
      const int k = me;
      const auto sh = ringShard(tileBytes, W, k);
      const size_t v = shardWindow(tileBytes, W, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = tileBuf + sh.offsetBytes + off;
        if (v % 16 == 0 && isAligned16(buf)) {
          next.send(ringGroup, buf, v, activeGroups, 0, timeout);
        } else {
          const size_t wire = (v + 15) & ~size_t(15);
          next.template send<ValidReduceStaged<T>>(
              ringGroup, buf, wire, activeGroups, 0, timeout, buf, v);
        }
      }
    }
    // steps 1..W-2: receive partial for shard (me-s), reduce local, forward.
    for (int s = 1; s <= W - 2; ++s) {
      const int k = ringRotate(me, s, W);
      const auto sh = ringShard(tileBytes, W, k);
      const size_t v = shardWindow(tileBytes, W, k, off, pipelineWindow);
      if (v > 0) {
        char* lin = tileBuf + sh.offsetBytes + off;
        if (v % 16 == 0 && isAligned16(lin)) {
          prev.template forward<ReduceOp>(
              ringGroup, nullptr, next, v, activeGroups, 0, timeout, lin);
        } else {
          const size_t wire = (v + 15) & ~size_t(15);
          prev.template forward<ValidReduceStaged<T>>(
              ringGroup, nullptr, next, wire, activeGroups, 0, timeout, lin, v);
        }
      }
    }
    // step W-1: final receive-reduce of shard (me+1) into tileBuf.
    {
      const int k =
          ringRotate(me, W - 1, W); // (me - (W-1)) mod W = (me+1) mod W
      const auto sh = ringShard(tileBytes, W, k);
      const size_t v = shardWindow(tileBytes, W, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = tileBuf + sh.offsetBytes + off;
        if (v % 16 == 0 && isAligned16(buf)) {
          prev.template recv<ReduceOp>(
              ringGroup, buf, v, activeGroups, 0, timeout, buf);
        } else {
          const size_t wire = (v + 15) & ~size_t(15);
          prev.template recv<ValidReduceStaged<T>>(
              ringGroup, buf, wire, activeGroups, 0, timeout, buf, v);
        }
      }
    }
  }

  // Block-local phase transition: shard (me+1) is now reduced in tileBuf and is
  // read by all-gather step 0 below.
  blockGroup.sync();

  // ============================== All-gather =============================
  // Per window: send(shard me+1) + forward(shard me .. me+1-(W-2), storing each
  // into tileBuf) + recv(shard me+2 into tileBuf). All-gather never reduces.
  for (size_t off = 0; off < maxShardBytes; off += pipelineWindow) {
    // step 0: broadcast own reduced sub-shard (me+1).
    {
      const int k = ringRotate(me + 1, 0, W);
      const auto sh = ringShard(tileBytes, W, k);
      const size_t v = shardWindow(tileBytes, W, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = tileBuf + sh.offsetBytes + off;
        if (v % 16 == 0 && isAligned16(buf)) {
          next.send(ringGroup, buf, v, activeGroups, 0, timeout);
        } else {
          const size_t wire = (v + 15) & ~size_t(15);
          next.template send<ValidMemcpy>(
              ringGroup, buf, wire, activeGroups, 0, timeout, v);
        }
      }
    }
    // steps 1..W-2: receive shard (me+1-s), store into tileBuf, forward.
    for (int s = 1; s <= W - 2; ++s) {
      const int k = ringRotate(me + 1, s, W);
      const auto sh = ringShard(tileBytes, W, k);
      const size_t v = shardWindow(tileBytes, W, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = tileBuf + sh.offsetBytes + off;
        if (v % 16 == 0 && isAligned16(buf)) {
          prev.forward(ringGroup, buf, next, v, activeGroups, 0, timeout);
        } else {
          const size_t wire = (v + 15) & ~size_t(15);
          prev.template forward<ValidMemcpy>(
              ringGroup, buf, next, wire, activeGroups, 0, timeout, v);
        }
      }
    }
    // step W-1: final receive of shard (me+2) into tileBuf.
    {
      const int k = ringRotate(me + 1, W - 1, W); // (me+2) mod W
      const auto sh = ringShard(tileBytes, W, k);
      const size_t v = shardWindow(tileBytes, W, k, off, pipelineWindow);
      if (v > 0) {
        char* buf = tileBuf + sh.offsetBytes + off;
        if (v % 16 == 0 && isAligned16(buf)) {
          prev.recv(ringGroup, buf, v, activeGroups, 0, timeout);
        } else {
          const size_t wire = (v + 15) & ~size_t(15);
          prev.template recv<ValidMemcpy>(
              ringGroup, buf, wire, activeGroups, 0, timeout, v);
        }
      }
    }
  }
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
