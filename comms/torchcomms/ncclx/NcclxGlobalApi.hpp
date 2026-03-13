// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <mutex>
#include <string>

#include <nccl.h> // @manual=//comms/ncclx:nccl

namespace torch::comms {

/**
 * Abstract interface for communicator-independent NCCL API operations.
 * This allows for dependency injection and testing by providing
 * a way to override global NCCL API calls.
 */
class NcclxGlobalApi {
 public:
  virtual ~NcclxGlobalApi() = default;

  // Error handling
  virtual const char* getErrorString(ncclResult_t result) = 0;
};

/**
 * Default implementation that calls the underlying NCCL APIs directly.
 */
class DefaultNcclxGlobalApi : public NcclxGlobalApi {
 public:
  ~DefaultNcclxGlobalApi() override = default;

  const char* getErrorString(ncclResult_t result) override;

 private:
  mutable std::mutex api_mutex_;
};

} // namespace torch::comms
