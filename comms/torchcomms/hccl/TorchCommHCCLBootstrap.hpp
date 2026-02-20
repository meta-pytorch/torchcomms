#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/device/npu/NpuApi.hpp"
#include "comms/torchcomms/hccl/HcclApi.hpp"

namespace torch::comms {

constexpr uint16_t kTCPStorePort = 29500;

class TorchCommHCCLBootstrap {
 public:
  TorchCommHCCLBootstrap(
      c10::intrusive_ptr<c10d::Store> store,
      c10::Device device,
      std::shared_ptr<HcclApi> hccl_api,
      std::shared_ptr<NpuApi> npu_api,
      std::chrono::milliseconds timeout);
  ~TorchCommHCCLBootstrap();

  // Delete copy and move operations
  TorchCommHCCLBootstrap(const TorchCommHCCLBootstrap&) = delete;
  TorchCommHCCLBootstrap& operator=(const TorchCommHCCLBootstrap&) = delete;
  TorchCommHCCLBootstrap(TorchCommHCCLBootstrap&&) = delete;
  TorchCommHCCLBootstrap& operator=(TorchCommHCCLBootstrap&&) = delete;

  HcclComm createHcclComm(
      const std::string& name,
      const CommOptions& options = {});
  static std::string getHCCLStoreKey();
  static std::string getHCCLStoreKeyPrefix();
  static int getHCCLStoreKeyCounter();

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
  HcclRootInfo exchangeUniqueId(std::string_view name);
  HcclRootInfo exchangeUniqueIdStore();
  HcclRootInfo exchangeUniqueIdTCPStore(std::string_view name);
  bool isTCPStoreEnabled();
  void cleanupTCPStore(HcclComm hccl_comm);

 private:
  const std::chrono::milliseconds timeout_;
  static int counter_;

  c10::intrusive_ptr<c10d::Store> store_;
  bool created_internal_store_;
  c10::Device device_;
  std::shared_ptr<HcclApi> hccl_api_;
  std::shared_ptr<NpuApi> npu_api_;
  void* barrier_buffer_{nullptr};
  int rank_;
  int comm_size_;

  std::string uniqueid_xchg_method_;
};

// Helper function to populate HCCL config from hints
void populateHcclConfigFromHints(
  HcclCommConfig& config,
    const CommOptions& options,
    const std::string& name);

} // namespace torch::comms
