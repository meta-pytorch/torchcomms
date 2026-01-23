// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/SendRecv/Types.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"

__device__ __forceinline__ void sendImpl(
    ctran::sendrecv::SendRecvOp* sends,
    size_t numSends,
    comms::pipes::P2pNvlTransportDevice* nvlTransportsBase,
    comms::pipes::ThreadGroup& group) {
  for (auto i = 0; i < numSends; i++) {
    const auto nbytes = sends[i].nbytes;
    const auto peerLocalRank = sends[i].peerLocalRank;
    nvlTransportsBase[peerLocalRank].send(group, sends[i].buff, nbytes);
  }
}

__device__ __forceinline__ void recvImpl(
    ctran::sendrecv::SendRecvOp* recvs,
    size_t numRecvs,
    comms::pipes::P2pNvlTransportDevice* nvlTransportsBase,
    comms::pipes::ThreadGroup& group) {
  for (auto i = 0; i < numRecvs; i++) {
    const auto nbytes = recvs[i].nbytes;
    const auto peerLocalRank = recvs[i].peerLocalRank;
    nvlTransportsBase[peerLocalRank].recv(group, recvs[i].buff, nbytes);
  }
}

__global__ void __launch_bounds__(1024, 1) ncclKernelSendRecvP2p(
    int* flag,
    CtranAlgoDeviceState* devState, // TODO: this is not needed for now, but
                                    // maybe needed for fault-tolerance
    ctran::sendrecv::KernArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(flag);
  }

  auto group = args.useBlockGroup ? comms::pipes::make_block_group()
                                  : comms::pipes::make_warp_group();

  // TODO: currently first args.numSendBlocks blocks allocated for send, and
  // rest for recv. Sends and recvs will happen sequentially in allocated blocks
  // we will need better allocation of blocks based on send/recv sizes.
  const uint32_t weights[] = {
      static_cast<uint32_t>(args.numSendBlocks),
      static_cast<uint32_t>(args.numRecvBlocks)};
  auto [partition_id, subgroup] =
      group.partition(comms::pipes::make_device_span(weights, 2u));

  // Use list format if enabled (fallback for > kCtranMaxNvlSendRecvOps),
  // otherwise use static arrays (fast path for common cases)
  ctran::sendrecv::SendRecvOp* sends =
      args.useList ? args.sendsList : args.sends;
  ctran::sendrecv::SendRecvOp* recvs =
      args.useList ? args.recvsList : args.recvs;

  if (partition_id == 0) {
    sendImpl(sends, args.numSends, args.nvlTransportsBase, subgroup);
  } else {
    recvImpl(recvs, args.numRecvs, args.nvlTransportsBase, subgroup);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
