// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "TorchCommNCCLXPersistentRequest.hpp"
#include "NcclxApi.hpp"
#include "TorchCommNCCLX.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch::comms {
TorchCommNCCLXPersistentRequest::TorchCommNCCLXPersistentRequest(
    std::shared_ptr<TorchCommNCCLX> comm,
    void* hdl,
    std::optional<cudaStream_t> stream)
    : comm_(std::move(comm)), hdl_(hdl), stream_(stream) {}

TorchCommNCCLXPersistentRequest::~TorchCommNCCLXPersistentRequest() noexcept {
  // TorchComm should have aborted process if commAbort is called (see
  // TorchCommNCCLX::abortNcclComm).
  auto nccl_api = comm_->getNcclApi();
  NCCLX_CHECK_IGNORE(nccl_api, nccl_api->pFree(hdl_), "NCCLX pFree failed");
  TC_LOG(INFO, nullptr) << "Finalized persistent request";
}

void* TorchCommNCCLXPersistentRequest::getRequestPtr() const {
  return hdl_;
}

std::optional<cudaStream_t> TorchCommNCCLXPersistentRequest::getStream() const {
  return stream_;
}

} // namespace torch::comms
