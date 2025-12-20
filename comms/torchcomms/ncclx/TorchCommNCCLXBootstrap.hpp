// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/device/CudaApi.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"
#include "nccl.h" // @manual=//comms/ncclx:nccl

namespace torch {
namespace comms {

constexpr uint16_t kTCPStorePort = 29500;

class TorchCommNCCLXBootstrap {
 public:
  TorchCommNCCLXBootstrap(
      c10::intrusive_ptr<c10d::Store> store,
      c10::Device device,
      std::shared_ptr<NcclxApi> nccl_api,
      std::shared_ptr<CudaApi> cuda_api,
      std::chrono::milliseconds timeout);
  ~TorchCommNCCLXBootstrap();

  // Delete copy and move operations
  TorchCommNCCLXBootstrap(const TorchCommNCCLXBootstrap&) = delete;
  TorchCommNCCLXBootstrap& operator=(const TorchCommNCCLXBootstrap&) = delete;
  TorchCommNCCLXBootstrap(TorchCommNCCLXBootstrap&&) = delete;
  TorchCommNCCLXBootstrap& operator=(TorchCommNCCLXBootstrap&&) = delete;

  ncclComm_t createNcclComm(
      const std::string& name,
      const CommOptions& options = {});
  static std::string getNCCLXStoreKey();
  static std::string getNCCLXStoreKeyPrefix();
  static int getNCCLXStoreKeyCounter();

  int getRank() {
    return rank_;
  }
  int getSize() {
    return comm_size_;
  }
  c10::Device getDevice() {
    return device_;
  }

 private:
  ncclUniqueId exchangeUniqueId();
  void createStore(std::string_view name);
  bool useFastInit(ncclConfig_t config);
  void cleanupTCPStore(ncclComm_t nccl_comm);

 private:
  const std::chrono::milliseconds timeout_;
  static int counter_;

  c10::intrusive_ptr<c10d::Store> store_;
  bool created_internal_store_;
  c10::Device device_;
  std::shared_ptr<NcclxApi> nccl_api_;
  std::shared_ptr<CudaApi> cuda_api_;
  void* barrier_buffer_{nullptr};
  int rank_;
  int comm_size_;

  std::string uniqueid_xchg_method_;
};

// Helper function to populate NCCL config from hints
void populateNcclConfigFromHints(
    ncclConfig_t& config,
    const CommOptions& options,
    const std::string& name);

} // namespace comms
} // namespace torch
