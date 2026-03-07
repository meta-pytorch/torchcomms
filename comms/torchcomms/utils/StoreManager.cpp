// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/torchcomms/utils/StoreManager.hpp"

#include <comms/torchcomms/utils/Logging.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include "comms/torchcomms/utils/Utils.hpp"

namespace torch::comms {

c10::intrusive_ptr<c10d::Store> createTCPStore(
    std::chrono::milliseconds timeout) {
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

c10::intrusive_ptr<c10d::Store> dupTCPStore(
    const c10::intrusive_ptr<c10d::Store>& store,
    std::chrono::milliseconds timeout) {
  const char* master_addr_env = std::getenv("MASTER_ADDR");
  TORCH_INTERNAL_ASSERT(
      master_addr_env != nullptr, "MASTER_ADDR env is not set");
  std::string host{master_addr_env};

  auto [rank, comm_size] = query_ranksize();
  (void)comm_size;

  const std::string key = "dup_store_port";

  if (rank == 0) {
    c10d::TCPStoreOptions opts;
    opts.port = 0;
    opts.isServer = true;
    opts.waitWorkers = false;
    opts.useLibUV = true;
    opts.timeout = timeout;
    auto newStore = c10::make_intrusive<c10d::TCPStore>(host, opts);

    std::string portStr = std::to_string(newStore->getPort());
    store->set(key, std::vector<uint8_t>(portStr.begin(), portStr.end()));

    return newStore;
  }

  store->wait({key}, timeout);
  auto portVec = store->get(key);
  uint16_t port = static_cast<uint16_t>(
      std::stoi(std::string(portVec.begin(), portVec.end())));

  c10d::TCPStoreOptions opts;
  opts.port = port;
  opts.isServer = false;
  opts.waitWorkers = false;
  opts.useLibUV = true;
  opts.timeout = timeout;
  return c10::make_intrusive<c10d::TCPStore>(host, opts);
}

} // namespace torch::comms
