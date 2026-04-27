// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/NcclxGlobalApi.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"

namespace torch::comms {

const char* DefaultNcclxGlobalApi::getErrorString(ncclResult_t result) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGetErrorString(result);
}

ncclResult_t DefaultNcclxGlobalApi::commDumpAll(
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>& map) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ::ncclCommDumpAll(map);
}

void DefaultNcclxGlobalApi::initCachingAllocatorHook() {
  // Create and immediately destroy a dummy single-rank communicator.
  // ncclCommInitRankConfig internally calls initEnv() + ncclCudaLibraryInit(),
  // which initializes the NCCL environment (folly singletons, cvars, logging,
  // etc.) and the CUDA driver library.
  ncclUniqueId id;
  ncclResult_t result = ncclGetUniqueId(&id);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        std::string("ncclGetUniqueId failed: ") + ncclGetErrorString(result));
  }
  ncclComm_t comm;
  result = ncclCommInitRankConfig(&comm, 1, id, 0, nullptr);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        std::string("ncclCommInitRankConfig failed: ") +
        ncclGetErrorString(result));
  }
  ncclCommDestroy(comm);
  CachingAllocatorHook::getInstance();
}

} // namespace torch::comms
