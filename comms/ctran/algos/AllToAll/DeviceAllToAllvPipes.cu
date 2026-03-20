// Copyright (c) Meta Platforms, Inc. and affiliates.

#if defined(ENABLE_PIPES)

#include <cstddef>
#include "comms/ctran/algos/AllToAll/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"

// Compute the exclusive prefix sum of counts[0..rank-1] to get the
// displacement for the given rank. Each thread computes its own peer's
// offset independently — no shared memory or __syncthreads() needed.
// For GB200 the NVL domain can have up to 72 ranks, so this is at most
// 71 additions from L1-cached global memory.
__device__ __forceinline__ int64_t
computeDisplacement(const int64_t* counts_d, int rank) {
  int64_t displ = 0;
  for (int r = 0; r < rank; r++) {
    displ += counts_d[r];
  }
  return displ;
}

__global__ void ncclKernelDeviceAllToAllvPipes(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::device_alltoallv_pipes::KernArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  const int nLocalRanks = args.nLocalRanks;
  const int myRank = args.myRank;
  const size_t elementSize = args.elementSize;
  const int64_t sendMultiplier = args.sendcountsMultiplier;
  const int64_t recvMultiplier = args.recvcountsMultiplier;
  auto* transports = args.transports;

  auto group = args.useBlockGroup ? comms::pipes::make_block_group()
                                  : comms::pipes::make_warp_group();

  if (nLocalRanks == 1) {
    // Single local rank — self-copy only
    int globalRank = args.localRankToGlobalRank[0];
    size_t sendBytes =
        args.sendcounts_d[globalRank] * sendMultiplier * elementSize;
    size_t sendOffset = computeDisplacement(args.sendcounts_d, globalRank) *
        sendMultiplier * elementSize;
    size_t recvOffset = computeDisplacement(args.recvcounts_d, globalRank) *
        recvMultiplier * elementSize;

    transports[globalRank].self.put(
        group,
        static_cast<char*>(args.recvbuff) + recvOffset,
        static_cast<const char*>(args.sendbuff) + sendOffset,
        sendBytes);
  } else {
    // Split into send and recv halves
    auto [partition_id, send_recv_group] = group.partition_interleaved(2);
    // Distribute across local peers
    auto [local_peer_idx, group_per_peer] =
        send_recv_group.partition_interleaved(nLocalRanks);

    int peerGlobalRank = args.localRankToGlobalRank[local_peer_idx];

    // Read counts from device memory (indexed by global rank)
    size_t sendBytes =
        args.sendcounts_d[peerGlobalRank] * sendMultiplier * elementSize;
    size_t recvBytes =
        args.recvcounts_d[peerGlobalRank] * recvMultiplier * elementSize;
    // Compute displacements as exclusive prefix sums of counts
    size_t sendOffset = computeDisplacement(args.sendcounts_d, peerGlobalRank) *
        sendMultiplier * elementSize;
    size_t recvOffset = computeDisplacement(args.recvcounts_d, peerGlobalRank) *
        recvMultiplier * elementSize;

    if (peerGlobalRank == myRank) {
      // Self-copy: only one partition does it
      if (partition_id == 0) {
        transports[peerGlobalRank].self.put(
            group_per_peer,
            static_cast<char*>(args.recvbuff) + recvOffset,
            static_cast<const char*>(args.sendbuff) + sendOffset,
            sendBytes);
      }
    } else if (partition_id == 0) {
      // Send to peer via NVL
      transports[peerGlobalRank].p2p_nvl.send(
          group_per_peer,
          static_cast<char*>(const_cast<void*>(args.sendbuff)) + sendOffset,
          sendBytes);
    } else {
      // Recv from peer via NVL
      transports[peerGlobalRank].p2p_nvl.recv(
          group_per_peer,
          static_cast<char*>(args.recvbuff) + recvOffset,
          recvBytes);
    }
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#endif // ENABLE_PIPES
