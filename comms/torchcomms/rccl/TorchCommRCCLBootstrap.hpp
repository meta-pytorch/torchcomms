// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <hip_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/rccl/HipApi.hpp" // @manual
#include "comms/torchcomms/rccl/RcclApi.hpp" // @manual
#include "rccl.h" // @manual=//comms/rccl:rccl-dev

namespace torch {
namespace comms {

constexpr uint16_t kTCPStorePort = 29500;

class TorchCommRCCLBootstrap {
 public:
  TorchCommRCCLBootstrap(
      c10::intrusive_ptr<c10d::Store> store,
      c10::Device device,
      std::shared_ptr<RcclApi> rccl_api,
      std::shared_ptr<HipApi> hip_api,
      std::chrono::milliseconds timeout);
  ~TorchCommRCCLBootstrap() noexcept;

  // Delete copy and move operations
  TorchCommRCCLBootstrap(const TorchCommRCCLBootstrap&) = delete;
  TorchCommRCCLBootstrap& operator=(const TorchCommRCCLBootstrap&) = delete;
  TorchCommRCCLBootstrap(TorchCommRCCLBootstrap&&) = delete;
  TorchCommRCCLBootstrap& operator=(TorchCommRCCLBootstrap&&) = delete;

  ncclComm_t createNcclComm(const std::string& name);
  static std::string getRCCLStoreKey();
  static std::string getRCCLStoreKeyPrefix();
  static int getRCCLStoreKeyCounter();

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
  ncclUniqueId exchangeUniqueId(std::string_view name);
  ncclUniqueId exchangeUniqueIdStore();
  ncclUniqueId exchangeUniqueIdTCPStore(std::string_view name);
  bool isTCPStoreEnabled();
  void cleanupTCPStore(ncclComm_t nccl_comm);

 private:
  const std::chrono::milliseconds timeout_;
  static int counter_;

  c10::intrusive_ptr<c10d::Store> store_;
  bool created_internal_store_;
  c10::Device device_;
  std::shared_ptr<RcclApi> rccl_api_;
  std::shared_ptr<HipApi> hip_api_;
  void* barrier_buffer_{nullptr};
  int rank_;
  int comm_size_;

  std::string uniqueid_xchg_method_;
};

} // namespace comms
} // namespace torch
