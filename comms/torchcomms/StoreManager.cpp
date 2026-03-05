// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/torchcomms/StoreManager.hpp"

#include <comms/torchcomms/TorchCommLogging.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include "comms/torchcomms/utils/Utils.hpp"

namespace torch::comms {

namespace {
c10::intrusive_ptr<c10d::Store> createRoot(
    const std::chrono::milliseconds& timeout) {
  const char* store_path = std::getenv("TORCHCOMM_STORE_PATH");
  if (store_path) {
    TC_LOG(INFO) << "Creating root FileStore at " << store_path;
    return c10::make_intrusive<c10d::FileStore>(store_path, -1);
  }

  const char* master_addr_env = std::getenv("MASTER_ADDR");
  TORCH_INTERNAL_ASSERT(
      master_addr_env != nullptr, "MASTER_ADDR env is not set");
  std::string host{master_addr_env};
  const char* master_port_env = std::getenv("MASTER_PORT");
  TORCH_INTERNAL_ASSERT(
      master_port_env != nullptr, "MASTER_PORT env is not set");
  int port{std::stoi(master_port_env)};

  auto [rank, comm_size] = query_ranksize();
  (void)comm_size; // unused

  c10d::TCPStoreOptions opts;
  opts.port = port;
  opts.isServer = (rank == 0);
  opts.waitWorkers = false;
  opts.useLibUV = true;
  opts.timeout = timeout;

  return c10::make_intrusive<c10d::TCPStore>(host, opts);
}
} // namespace

StoreManager& StoreManager::get() {
  static StoreManager storeManager;
  return storeManager;
}

c10::intrusive_ptr<c10d::Store> StoreManager::getStore(
    std::string_view backendName,
    std::string_view commName,
    std::chrono::milliseconds timeout) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string prefix =
      fmt::format("torchcomm(backend={},name={})", backendName, commName);

  // Prevent prefix reuse to avoid key collisions in the underlying store.
  // Each communicator should have a unique namespace.
  if (storeNames_.contains(prefix)) {
    throw std::runtime_error("Store prefix has been reused for: " + prefix);
  }
  storeNames_.insert(prefix);

  if (!root_) {
    root_ = createRoot(timeout);
  }

  return c10::make_intrusive<c10d::PrefixStore>(prefix, root_->clone());
}

} // namespace torch::comms
