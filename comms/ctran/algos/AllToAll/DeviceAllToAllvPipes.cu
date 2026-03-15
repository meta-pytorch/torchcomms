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
  auto* transports = args.transports;

  auto group = args.useBlockGroup ? comms::pipes::make_block_group()
                                  : comms::pipes::make_warp_group();

  if (nLocalRanks == 1) {
    // Single local rank — self-copy only
    int globalRank = args.localRankToGlobalRank[0];
    size_t sendBytes = args.sendcounts_d[globalRank] * elementSize;
    size_t sendOffset = args.senddispls_d[globalRank] * elementSize;
    size_t recvOffset = args.recvdispls_d[globalRank] * elementSize;

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

    // Read counts and displacements from device memory (indexed by global rank)
    size_t sendBytes = args.sendcounts_d[peerGlobalRank] * elementSize;
    size_t recvBytes = args.recvcounts_d[peerGlobalRank] * elementSize;
    size_t sendOffset = args.senddispls_d[peerGlobalRank] * elementSize;
    size_t recvOffset = args.recvdispls_d[peerGlobalRank] * elementSize;

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
