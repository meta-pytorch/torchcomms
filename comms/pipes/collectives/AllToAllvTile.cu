// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/collectives/AllToAllvTile.cuh"

namespace comms::pipes {

namespace {

__device__ __forceinline__ void send_to_peer(
    MultiPeerDeviceHandle& handle,
    int peer,
    ThreadGroup& group,
    void* src,
    std::size_t nbytes,
    int active_blocks,
    std::size_t max_signal_bytes,
    const Timeout& timeout) {
  if (handle.get_type(peer) == TransportType::P2P_NVL) {
    handle.get_nvl(peer).send(
        group, src, nbytes, active_blocks, max_signal_bytes, timeout);
  } else {
    handle.get_ibgda(peer).send(
        group, src, nbytes, active_blocks, max_signal_bytes, timeout);
  }
}

__device__ __forceinline__ void recv_from_peer(
    MultiPeerDeviceHandle& handle,
    int peer,
    ThreadGroup& group,
    void* dst,
    std::size_t nbytes,
    int active_blocks,
    std::size_t max_signal_bytes,
    const Timeout& timeout) {
  if (handle.get_type(peer) == TransportType::P2P_NVL) {
    handle.get_nvl(peer).recv(
        group, dst, nbytes, active_blocks, max_signal_bytes, timeout);
  } else {
    handle.get_ibgda(peer).recv(
        group, dst, nbytes, active_blocks, max_signal_bytes, timeout);
  }
}

__device__ __forceinline__ int find_peer_by_type(
    const MultiPeerDeviceHandle& handle,
    TransportType type,
    int idx) {
  int count = 0;
  for (int r = 0; r < handle.nRanks; r++) {
    if (handle.get_type(r) == type) {
      if (count == idx) {
        return r;
      }
      count++;
    }
  }
  return -1;
}

// Distribute blocks across peers with uniform blocksPerPeer.
// Uses arithmetic partitioning (floor division) instead of
// ThreadGroup::partition() to guarantee every peer gets the same
// blocksPerPeer. This is critical because sender and receiver ranks
// map each other to different peerIdx values, and non-uniform
// blocksPerPeer would cause active_blocks mismatch in send/recv.
__device__ __forceinline__ void process_peers(
    MultiPeerDeviceHandle& handle,
    const AllToAllvTileArgs& args,
    ThreadGroup& transportGroup,
    int role,
    TransportType peerType,
    int numPeers,
    const Timeout& timeout) {
  if (numPeers == 0) {
    return;
  }
  int numBlocks = transportGroup.total_groups;
  int blockId = transportGroup.group_id;
  int numPartitions = numBlocks < numPeers ? numBlocks : numPeers;
  int blocksPerPeer = numBlocks / numPartitions;
  int activeBlocks = blocksPerPeer * numPartitions;
  int peersPerGroup = (numPeers + numPartitions - 1) / numPartitions;

  if (blockId >= activeBlocks) {
    return;
  }

  int partIdx = blockId / blocksPerPeer;
  int tileIdx = blockId % blocksPerPeer;

  ThreadGroup peerGroup = transportGroup;
  peerGroup.group_id = tileIdx;
  peerGroup.total_groups = blocksPerPeer;

  for (int p = 0; p < peersPerGroup; p++) {
    int peerIdx = partIdx * peersPerGroup + p;
    if (peerIdx >= numPeers) {
      break;
    }
    int peer = find_peer_by_type(handle, peerType, peerIdx);
    if (peer < 0) {
      continue;
    }
    std::size_t bytes =
        (role == 0) ? args.send_counts[peer] : args.recv_counts[peer];
    char** ptrs = (role == 0) ? args.send_ptrs : args.recv_ptrs;
    if (bytes > 0) {
      TiledBuffer<char> tiles(ptrs[peer], bytes, blocksPerPeer);
      if (role == 0) {
        send_to_peer(
            handle,
            peer,
            peerGroup,
            tiles.tile_data(tileIdx),
            tiles.tile_bytes(tileIdx),
            blocksPerPeer,
            args.max_signal_bytes,
            timeout);
      } else {
        recv_from_peer(
            handle,
            peer,
            peerGroup,
            tiles.tile_data(tileIdx),
            tiles.tile_bytes(tileIdx),
            blocksPerPeer,
            args.max_signal_bytes,
            timeout);
      }
    }
  }
}

// NVL 2D: interleaved send/recv + uniform partition across peers.
// Uses arithmetic partitioning with floor(halfBlocks / numPeers) to guarantee
// uniform blocksPerPeer across all peer indices. This is critical because
// sender rank A and receiver rank B map each other to different peerIdx values,
// so non-uniform blocksPerPeer from ThreadGroup::partition() would cause
// active_blocks mismatch in the send/recv signal protocol (deadlock).
// When halfBlocks < numPeers, each block handles multiple peers sequentially.
__device__ __forceinline__ void nvl_2d_path(
    const AllToAllvTileArgs& args,
    ThreadGroup transportGroup,
    Timeout timeout) {
  auto handle = args.handle;
  const int myRank = handle.myRank;
  const int nRanks = handle.nRanks;
  const int numPeers = nRanks - 1;

  if (numPeers == 0) {
    std::size_t selfBytes = args.send_counts[myRank];
    if (selfBytes > 0) {
      memcpy_vectorized(
          args.recv_ptrs[myRank],
          args.send_ptrs[myRank],
          selfBytes,
          transportGroup);
    }
    return;
  }

  auto [role, halfGroup] = transportGroup.partition_interleaved(2);
  const int halfBlocks = halfGroup.total_groups;
  const int blockId = halfGroup.group_id;

  int numPartitions = halfBlocks < numPeers ? halfBlocks : numPeers;
  int blocksPerPeer = halfBlocks / numPartitions;
  int activeBlocks = blocksPerPeer * numPartitions;
  int peersPerGroup = (numPeers + numPartitions - 1) / numPartitions;

  if (blockId >= activeBlocks) {
    return;
  }

  int partIdx = blockId / blocksPerPeer;
  int tileIdx = blockId % blocksPerPeer;

  ThreadGroup peerGroup = halfGroup;
  peerGroup.group_id = tileIdx;
  peerGroup.total_groups = blocksPerPeer;

  if (role == 0 && partIdx == 0) {
    std::size_t selfBytes = args.send_counts[myRank];
    if (selfBytes > 0) {
      memcpy_vectorized(
          args.recv_ptrs[myRank], args.send_ptrs[myRank], selfBytes, peerGroup);
    }
  }

  for (int p = 0; p < peersPerGroup; p++) {
    int peerIdx = partIdx * peersPerGroup + p;
    if (peerIdx >= numPeers) {
      break;
    }

    int peer = peerIdx < myRank ? peerIdx : peerIdx + 1;

    if (role == 0) {
      std::size_t sendBytes = args.send_counts[peer];
      if (sendBytes > 0) {
        TiledBuffer<char> tiles(args.send_ptrs[peer], sendBytes, blocksPerPeer);
        send_to_peer(
            handle,
            peer,
            peerGroup,
            tiles.tile_data(tileIdx),
            tiles.tile_bytes(tileIdx),
            blocksPerPeer,
            args.max_signal_bytes,
            timeout);
      }
    } else {
      std::size_t recvBytes = args.recv_counts[peer];
      if (recvBytes > 0) {
        TiledBuffer<char> tiles(args.recv_ptrs[peer], recvBytes, blocksPerPeer);
        recv_from_peer(
            handle,
            peer,
            peerGroup,
            tiles.tile_data(tileIdx),
            tiles.tile_bytes(tileIdx),
            blocksPerPeer,
            args.max_signal_bytes,
            timeout);
      }
    }
  }
}

// NVL 1D: self-copy with all blocks, then sequential remote peer iteration.
__device__ __forceinline__ void nvl_1d_path(
    const AllToAllvTileArgs& args,
    ThreadGroup transportGroup,
    Timeout timeout) {
  auto handle = args.handle;
  const int myRank = handle.myRank;
  const int nRanks = handle.nRanks;

  // Self-copy: all blocks cooperate before splitting for remote peers
  std::size_t selfBytes = args.send_counts[myRank];
  if (selfBytes > 0) {
    TiledBuffer<char> sendTile(
        args.send_ptrs[myRank], selfBytes, transportGroup);
    TiledBuffer<char> recvTile(
        args.recv_ptrs[myRank], selfBytes, transportGroup);
    memcpy_vectorized(
        recvTile.data(), sendTile.data(), sendTile.bytes(), transportGroup);
  }

  auto [role, sub] = transportGroup.partition_interleaved(2);
  const int blockId = sub.group_id;
  const int activeBlocks = sub.total_groups;

  for (int offset = 1; offset < nRanks; offset++) {
    int peer = (role == 0) ? (myRank + offset) % nRanks
                           : (myRank - offset + nRanks) % nRanks;
    if (handle.get_type(peer) != TransportType::P2P_NVL) {
      continue;
    }
    std::size_t bytes =
        (role == 0) ? args.send_counts[peer] : args.recv_counts[peer];
    char** ptrs = (role == 0) ? args.send_ptrs : args.recv_ptrs;
    if (bytes > 0) {
      TiledBuffer<char> tiles(ptrs[peer], bytes, activeBlocks);
      if (role == 0) {
        send_to_peer(
            handle,
            peer,
            sub,
            tiles.tile_data(blockId),
            tiles.tile_bytes(blockId),
            activeBlocks,
            args.max_signal_bytes,
            timeout);
      } else {
        recv_from_peer(
            handle,
            peer,
            sub,
            tiles.tile_data(blockId),
            tiles.tile_bytes(blockId),
            activeBlocks,
            args.max_signal_bytes,
            timeout);
      }
    }
  }
}

// IB: interleaved send/recv, sequential peer iteration.
// Uses all active IB groups for each peer to saturate per-peer QPs while
// respecting sendRecv.maxGroups.
__device__ __forceinline__ void ib_path(
    const AllToAllvTileArgs& args,
    ThreadGroup transportGroup,
    Timeout timeout) {
  auto handle = args.handle;
  auto [role, sub] = transportGroup.partition_interleaved(2);
  const int myRank = handle.myRank;
  const int nRanks = handle.nRanks;
  const int blockId = sub.group_id;
  const int activeBlocks = sub.total_groups;

  for (int offset = 1; offset < nRanks; offset++) {
    int peer = (role == 0) ? (myRank + offset) % nRanks
                           : (myRank - offset + nRanks) % nRanks;
    if (handle.get_type(peer) != TransportType::P2P_IBGDA) {
      continue;
    }
    std::size_t bytes =
        (role == 0) ? args.send_counts[peer] : args.recv_counts[peer];
    char** ptrs = (role == 0) ? args.send_ptrs : args.recv_ptrs;
    if (bytes > 0) {
      TiledBuffer<char> tiles(ptrs[peer], bytes, activeBlocks);
      if (role == 0) {
        send_to_peer(
            handle,
            peer,
            sub,
            tiles.tile_data(blockId),
            tiles.tile_bytes(blockId),
            activeBlocks,
            args.max_signal_bytes,
            timeout);
      } else {
        recv_from_peer(
            handle,
            peer,
            sub,
            tiles.tile_data(blockId),
            tiles.tile_bytes(blockId),
            activeBlocks,
            args.max_signal_bytes,
            timeout);
      }
    }
  }
}

// Unified kernel implementation templated on NVL strategy.
// NvlParallel=true: NVL 2D (partition across peers)
// NvlParallel=false: NVL 1D (sequential peer iteration)
// IB always uses partition across peers.
template <bool NvlParallel>
__device__ __forceinline__ void alltoallv_tile_impl(
    const AllToAllvTileArgs& args,
    Timeout& timeout) {
  auto group = make_block_group();
  auto [transportId, transportGroup] = group.split(args.num_blocks_nvl);

  if (transportId == 0) {
    if constexpr (NvlParallel) {
      nvl_2d_path(args, transportGroup, timeout);
    } else {
      nvl_1d_path(args, transportGroup, timeout);
    }
    return;
  }
  ib_path(args, transportGroup, timeout);
}

} // namespace

__global__ __launch_bounds__(512, 1) void alltoallv_tile_1d_kernel(
    const __grid_constant__ AllToAllvTileArgs args,
    Timeout timeout) {
  timeout.start();
  alltoallv_tile_impl<false>(args, timeout);
}

__global__ __launch_bounds__(512, 1) void alltoallv_tile_2d_kernel(
    const __grid_constant__ AllToAllvTileArgs args,
    Timeout timeout) {
  timeout.start();
  alltoallv_tile_impl<true>(args, timeout);
}

} // namespace comms::pipes
