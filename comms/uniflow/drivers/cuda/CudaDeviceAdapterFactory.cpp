// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <utility>

#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"

namespace uniflow {

bool isNvlinkAvailable() {
  return true;
}

std::shared_ptr<DeviceAdapter> createDeviceAdapter(
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi) {
  if (!cudaApi) {
    cudaApi = std::make_shared<CudaApi>();
  }
  if (!cudaDriverApi) {
    cudaDriverApi = std::make_shared<CudaDriverApi>();
  }
  return std::make_shared<CudaDeviceAdapter>(
      std::move(cudaApi), std::move(cudaDriverApi));
}

} // namespace uniflow
