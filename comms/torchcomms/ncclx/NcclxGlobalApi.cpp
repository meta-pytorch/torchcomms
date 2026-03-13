// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/NcclxGlobalApi.hpp"

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

} // namespace torch::comms
