// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/SendRecv/Types.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

__device__ __forceinline__ void sendImpl(
    ctran::sendrecv::SendRecvOp* sends,
    size_t numSends,
    comms::prims::P2pNvlTransportDevice* nvlTransportsBase,
    comms::prims::ThreadGroup& group) {
  for (auto i = 0; i < numSends; i++) {
    if (group.group_id >= sends[i].nGroups) {
      continue;
    }
    const auto nbytes = sends[i].nbytes;
    const auto peerLocalRank = sends[i].peerLocalRank;
    comms::prims::ThreadGroup opGroup{
        .thread_id_in_group = group.thread_id_in_group,
        .group_size = group.group_size,
        .group_id = group.group_id,
        .block_id = group.block_id,
        .total_groups = static_cast<uint32_t>(sends[i].nGroups),
        .scope = group.scope};
    comms::prims::TiledBuffer<char> tiles(
        static_cast<char*>(sends[i].buff), nbytes, opGroup);
    nvlTransportsBase[peerLocalRank].send(
        opGroup,
        tiles.data(),
        tiles.bytes(),
        /*max_signal_bytes=*/0);
  }
}

__device__ __forceinline__ void recvImpl(
    ctran::sendrecv::SendRecvOp* recvs,
    size_t numRecvs,
    comms::prims::P2pNvlTransportDevice* nvlTransportsBase,
    comms::prims::ThreadGroup& group) {
  for (auto i = 0; i < numRecvs; i++) {
    if (group.group_id >= recvs[i].nGroups) {
      continue;
    }
    const auto nbytes = recvs[i].nbytes;
    const auto peerLocalRank = recvs[i].peerLocalRank;
    comms::prims::ThreadGroup opGroup{
        .thread_id_in_group = group.thread_id_in_group,
        .group_size = group.group_size,
        .group_id = group.group_id,
        .block_id = group.block_id,
        .total_groups = static_cast<uint32_t>(recvs[i].nGroups),
        .scope = group.scope};
    comms::prims::TiledBuffer<char> tiles(
        static_cast<char*>(recvs[i].buff), nbytes, opGroup);
    nvlTransportsBase[peerLocalRank].recv(
        opGroup,
        tiles.data(),
        tiles.bytes(),
        /*max_signal_bytes=*/0);
  }
}

__global__ __launch_bounds__(512, 1) void ncclKernelSendRecvP2p(
    int* flag,
    CtranAlgoDeviceState* devState, // TODO: this is not needed for now, but
                                    // maybe needed for fault-tolerance
    ctran::sendrecv::KernArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  auto group = args.useBlockGroup ? comms::prims::make_block_group()
                                  : comms::prims::make_warp_group();

  // TODO: currently first args.numSendBlocks blocks allocated for send, and
  // rest for recv. Sends and recvs will happen sequentially in allocated blocks
  // we will need better allocation of blocks based on send/recv sizes.
  const uint32_t weights[] = {
      static_cast<uint32_t>(args.numSendBlocks),
      static_cast<uint32_t>(args.numRecvBlocks)};
  auto [partition_id, subgroup] =
      group.partition(comms::prims::make_device_span(weights, 2u));

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
