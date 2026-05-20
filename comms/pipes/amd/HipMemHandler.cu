#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "HipMemHandler.h"

#include <glog/logging.h>
#include <stdexcept>
#include <string>

namespace comms::pipes {

namespace {

#define HIP_CHECK(call)                                               \
  do {                                                                \
    hipError_t err = (call);                                          \
    if (err != hipSuccess) {                                          \
      throw std::runtime_error(                                       \
          std::string(#call) + " failed: " + hipGetErrorString(err)); \
    }                                                                 \
  } while (0)

// IPC handle exchange info (fixed-size for allGather)
struct IpcExchangeInfo {
  hipIpcMemHandle_t handle;
  size_t allocatedSize;
};

} // namespace

HipMemHandler::HipMemHandler(
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    size_t size)
    : bootstrap_(std::move(bootstrap)), selfRank_(selfRank), nRanks_(nRanks) {
  // Align to 256 bytes (HIP IPC requirement)
  allocatedSize_ = (size + 255) & ~255ULL;

  HIP_CHECK(hipMalloc(&localPtr_, allocatedSize_));
  HIP_CHECK(hipMemset(localPtr_, 0, allocatedSize_));
  HIP_CHECK(hipIpcGetMemHandle(&localHandle_, localPtr_));

  VLOG(1) << "HipMemHandler: rank " << selfRank_ << " allocated "
          << allocatedSize_ << " bytes at " << localPtr_;

  // Auto-exchange on construction (matches GpuMemHandler behavior)
  exchangeMemPtrs();
}

HipMemHandler::HipMemHandler(
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    size_t size,
    const std::vector<int32_t>& localPeerRanks)
    : bootstrap_(std::move(bootstrap)),
      selfRank_(selfRank),
      nRanks_(nRanks),
      filtered_(true),
      localPeerRanks_(localPeerRanks) {
  allocatedSize_ = (size + 255) & ~255ULL;

  HIP_CHECK(hipMalloc(&localPtr_, allocatedSize_));
  HIP_CHECK(hipMemset(localPtr_, 0, allocatedSize_));
  HIP_CHECK(hipIpcGetMemHandle(&localHandle_, localPtr_));

  VLOG(1) << "HipMemHandler(filtered): rank " << selfRank_ << " allocated "
          << allocatedSize_ << " bytes, " << localPeerRanks_.size()
          << " local peers";

  exchangeMemPtrs();
}

HipMemHandler::~HipMemHandler() {
  // Close peer IPC handles
  for (int i = 0; i < nRanks_; i++) {
    if (i == selfRank_)
      continue;
    int idx = (i < selfRank_) ? i : (i - 1);
    if (idx < static_cast<int>(peerPtrs_.size()) && peerPtrs_[idx]) {
      hipError_t err = hipIpcCloseMemHandle(peerPtrs_[idx]);
      if (err != hipSuccess) {
        LOG(WARNING) << "HipMemHandler: hipIpcCloseMemHandle failed for peer "
                     << i << " (error=" << hipGetErrorString(err) << ")";
      }
    }
  }
  peerPtrs_.clear();

  // Free local allocation
  if (localPtr_) {
    hipFree(localPtr_);
    localPtr_ = nullptr;
  }
}

void HipMemHandler::exchangeMemPtrs() {
  if (exchanged_)
    return;

  // Exchange IPC handles via allGather
  std::vector<IpcExchangeInfo> allInfo(nRanks_);
  allInfo[selfRank_].handle = localHandle_;
  allInfo[selfRank_].allocatedSize = allocatedSize_;

  auto result =
      bootstrap_
          ->allGather(
              allInfo.data(), sizeof(IpcExchangeInfo), selfRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error("HipMemHandler: allGather failed");
  }

  // Open peer IPC handles
  const int numPeers = nRanks_ - 1;
  peerPtrs_.resize(numPeers, nullptr);

  for (int rank = 0; rank < nRanks_; rank++) {
    if (rank == selfRank_)
      continue;

    int peerIdx = (rank < selfRank_) ? rank : (rank - 1);

    // In filtered mode, only IPC-open handles for local peers
    if (filtered_) {
      bool isLocal = false;
      for (int32_t lr : localPeerRanks_) {
        if (lr == rank) {
          isLocal = true;
          break;
        }
      }
      if (!isLocal) {
        continue; // peerPtrs_[peerIdx] stays nullptr
      }
    }

    HIP_CHECK(hipIpcOpenMemHandle(
        &peerPtrs_[peerIdx],
        allInfo[rank].handle,
        hipIpcMemLazyEnablePeerAccess));

    VLOG(1) << "HipMemHandler: rank " << selfRank_ << " opened peer " << rank
            << " at " << peerPtrs_[peerIdx];
  }

  exchanged_ = true;
}

void* HipMemHandler::getLocalDeviceMemPtr() const {
  return localPtr_;
}

void* HipMemHandler::getPeerDeviceMemPtr(int32_t rank) const {
  if (!exchanged_) {
    throw std::runtime_error("HipMemHandler: exchangeMemPtrs() not called");
  }
  if (rank == selfRank_) {
    return localPtr_;
  }
  int peerIdx = (rank < selfRank_) ? rank : (rank - 1);
  return peerPtrs_[peerIdx];
}

} // namespace comms::pipes
#endif
