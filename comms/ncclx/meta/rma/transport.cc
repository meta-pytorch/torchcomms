// Copyright (c) Meta Platforms, Inc. and affiliates.
// Pipes transport API implementations (non-window transport operations).

#if defined(ENABLE_PRIMS)

#include "checks.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "meta/wrapper/NcclCommCtran.h"

#include <exception>

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

  if (!ctranInitialized(meta::comms::ncclx::ncclCommCtran(comm).get())) {
    WARN("ncclGetMultiPeerDeviceHandle: ctran not initialized");
    return ncclInternalError;
  }

  auto* mpt =
      meta::comms::ncclx::ncclCommCtran(comm)->multiPeerTransport_.get();
  if (mpt == nullptr) {
    WARN(
        "ncclGetMultiPeerDeviceHandle: MultiPeerTransport not initialized. "
        "Set NCCL_CTRAN_USE_PIPES=1");
    return ncclInternalError;
  }

  try {
    auto handle = mpt->get_device_handle(mpt->ib_peer_ranks());
    *outTransportsPtr = handle.transports.data();
    *outMyRank = handle.myRank;
    *outNRanks = handle.nRanks;
    *outNumNvlPeers = handle.numNvlPeers;
    *outNumIbPeers = handle.numIbPeers;
  } catch (const std::exception& ex) {
    WARN("ncclGetMultiPeerDeviceHandle failed: %s", ex.what());
    return ncclInternalError;
  } catch (...) {
    WARN("ncclGetMultiPeerDeviceHandle failed with unknown exception");
    return ncclInternalError;
  }
  return ncclSuccess;
}

#endif // ENABLE_PRIMS
