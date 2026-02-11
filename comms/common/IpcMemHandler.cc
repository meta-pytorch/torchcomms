// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/common/IpcMemHandler.h"
#include "comms/utils/checks.h"

namespace meta::comms {

IpcMemHandler::IpcMemHandler(
    std::shared_ptr<IBootstrap> commBootstrap,
    int32_t selfRank,
    int32_t nRanks)
    : commBootstrap_(std::move(commBootstrap)),
      selfRank_(selfRank),
      nRanks_(nRanks),
      memPtrs_(nRanks) {}

IpcMemHandler::~IpcMemHandler() {
  if (!exchanged_) {
    return;
  }
  for (int i = 0; i < nRanks_; ++i) {
    if (i == selfRank_) {
      continue;
    }
    CUDA_CHECK(cudaIpcCloseMemHandle(memPtrs_.at(i)));
  }
}

void IpcMemHandler::addSelfDeviceMemPtr(void* deviceMemPtr) {
  memPtrs_.at(selfRank_) = deviceMemPtr;
}

void* IpcMemHandler::getPeerDeviceMemPtr(int32_t rank) {
  if (!exchanged_) {
    throw std::runtime_error("MemPtrs not exchanged yet");
  }
  return memPtrs_.at(rank);
}

void IpcMemHandler::exchangeMemPtrs() {
  if (exchanged_) {
    return;
  }

  // bootstrapAllGather only works with CPU buffer, so we use host memory rather
  // than device memory here.
  cudaIpcMemHandle_t ipcHandles[nRanks_];
  CUDA_CHECK(
      cudaIpcGetMemHandle(&ipcHandles[selfRank_], memPtrs_.at(selfRank_)));

  if (commBootstrap_
          ->allGather(
              ipcHandles, sizeof(cudaIpcMemHandle_t), selfRank_, nRanks_)
          .get() != 0) {
    throw std::runtime_error(
        "IpcMemHandler::exchangeMemPtrs allGather failed.");
  }

  for (int i = 0; i < nRanks_; ++i) {
    if (i == selfRank_) {
      continue;
    }
    CUDA_CHECK(cudaIpcOpenMemHandle(
        &memPtrs_[i], ipcHandles[i], cudaIpcMemLazyEnablePeerAccess));
  }
  exchanged_ = true;
}

} // namespace meta::comms
