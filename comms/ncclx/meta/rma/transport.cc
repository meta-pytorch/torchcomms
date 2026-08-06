// Copyright (c) Meta Platforms, Inc. and affiliates.
// Pipes transport API implementations (non-window transport operations).

#if defined(ENABLE_PRIMS)

#include <vector>

#include "checks.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"
#include "comms/prims/transport/MultiPeerTransport.h"

#include "nccl.h"

NCCL_API(
    ncclResult_t,
    ncclGetMultiPeerDeviceHandle,
    ncclComm_t comm,
    void** outTransportsPtr,
    int* outMyRank,
    int* outNRanks,
    int* outNumNvlPeers,
    int* outNumIbPeers);
ncclResult_t ncclGetMultiPeerDeviceHandle(
    ncclComm_t comm,
    void** outTransportsPtr,
    int* outMyRank,
    int* outNRanks,
    int* outNumNvlPeers,
    int* outNumIbPeers) {
  if (comm == nullptr || outTransportsPtr == nullptr || outMyRank == nullptr ||
      outNRanks == nullptr || outNumNvlPeers == nullptr ||
      outNumIbPeers == nullptr) {
    return ncclInvalidArgument;
  }

  if (!ctranInitialized(comm->ctranComm_.get())) {
    WARN("ncclGetMultiPeerDeviceHandle: ctran not initialized");
    return ncclInternalError;
  }

  auto* mpt = comm->ctranComm_->multiPeerTransport_.get();
  if (mpt == nullptr) {
    WARN(
        "ncclGetMultiPeerDeviceHandle: MultiPeerTransport not initialized. "
        "Set NCCL_CTRAN_USE_PIPES=1");
    return ncclInternalError;
  }

  std::vector<comms::prims::PeerChannelDemand> demands;
  demands.reserve(mpt->ib_peer_ranks().size());
  for (int peer : mpt->ib_peer_ranks()) {
    demands.push_back(
        {.peerRank = peer, .ibChannels = mpt->ib_channel_capacity()});
  }
  auto handle = mpt->get_device_handle(demands);
  *outTransportsPtr = handle.transports.data();
  *outMyRank = handle.myRank;
  *outNRanks = handle.nRanks;
  *outNumNvlPeers = handle.numNvlPeers;
  *outNumIbPeers = handle.numIbPeers;
  return ncclSuccess;
}

#endif // ENABLE_PRIMS
