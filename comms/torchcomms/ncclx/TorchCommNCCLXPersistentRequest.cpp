// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "TorchCommNCCLXPersistentRequest.hpp"
#include "TorchCommNCCLX.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"

namespace torch {
namespace comms {
TorchCommNCCLXPersistentRequest::TorchCommNCCLXPersistentRequest(
    std::shared_ptr<TorchCommNCCLX> comm,
    void* hdl,
    std::optional<cudaStream_t> stream)
    : comm_(std::move(comm)), hdl_(hdl), stream_(stream) {}

TorchCommNCCLXPersistentRequest::~TorchCommNCCLXPersistentRequest() {
  // TorchComm should have aborted process if commAbort is called (see
  // TorchCommNCCLX::abortNcclComm).
  auto nccl_api = comm_->getNcclApi();
  ncclResult_t result = nccl_api->pFree(hdl_);
  if (result != ncclSuccess) {
    throw NCCLException(*nccl_api, "NCCL pFree failed", result);
  }
  TC_LOG(INFO, nullptr) << "Finalized persistent request";
}

void* TorchCommNCCLXPersistentRequest::getRequestPtr() const {
  return hdl_;
}

std::optional<cudaStream_t> TorchCommNCCLXPersistentRequest::getStream() const {
  return stream_;
}

} // namespace comms
} // namespace torch
