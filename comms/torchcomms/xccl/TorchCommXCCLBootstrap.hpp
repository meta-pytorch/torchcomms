#pragma once

#include <memory>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include <oneapi/ccl.h>
#include <oneapi/ccl.hpp>
#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/device/XpuApi.hpp"
#include "comms/torchcomms/xccl/XcclApi.hpp"

namespace torch {
namespace comms {

constexpr uint16_t kTCPStorePort = 29500;

class TorchCommXCCLBootstrap {
 public:
  TorchCommXCCLBootstrap(
      c10::intrusive_ptr<c10d::Store> store,
      c10::Device device,
      std::shared_ptr<XcclApi> xccl_api,
      std::shared_ptr<XpuApi> xpu_api,
      std::chrono::milliseconds timeout);
  ~TorchCommXCCLBootstrap();

  // Delete copy and move operations
  TorchCommXCCLBootstrap(const TorchCommXCCLBootstrap&) = delete;
  TorchCommXCCLBootstrap& operator=(const TorchCommXCCLBootstrap&) = delete;
  TorchCommXCCLBootstrap(TorchCommXCCLBootstrap&&) = delete;
  TorchCommXCCLBootstrap& operator=(TorchCommXCCLBootstrap&&) = delete;

  onecclComm_t createXcclComm(
      const std::string& name,
      const CommOptions& options = {});
  static std::string getXCCLStoreKey();
  static std::string getXCCLStoreKeyPrefix();
  static int getXCCLStoreKeyCounter();

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
  onecclUniqueId exchangeUniqueId(std::string_view name);
  onecclUniqueId exchangeUniqueIdStore();
  onecclUniqueId exchangeUniqueIdTCPStore(std::string_view name);
  bool isTCPStoreEnabled();
  void cleanupTCPStore(onecclComm_t xccl_comm);

 private:
  const std::chrono::milliseconds timeout_;
  static int counter_;

  c10::intrusive_ptr<c10d::Store> store_;
  bool created_internal_store_;
  c10::Device device_;
  std::shared_ptr<XcclApi> xccl_api_;
  std::shared_ptr<XpuApi> xpu_api_;
  void* barrier_buffer_{nullptr};
  int rank_;
  int comm_size_;

  std::string uniqueid_xchg_method_;
};

// Helper function to populate XCCL config from hints
void populateXcclConfigFromHints(
    onecclConfig_t& config,
    const CommOptions& options,
    const std::string& name);

} // namespace comms
} // namespace torch
