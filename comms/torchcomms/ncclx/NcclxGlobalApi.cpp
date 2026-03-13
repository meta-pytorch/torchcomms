// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/NcclxGlobalApi.hpp"

namespace torch::comms {

const char* DefaultNcclxGlobalApi::getErrorString(ncclResult_t result) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return ncclGetErrorString(result);
}

} // namespace torch::comms
