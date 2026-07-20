// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/torchcomms/utils/StoreManager.hpp"

#include <comms/torchcomms/utils/Logging.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <string>
#include "comms/torchcomms/utils/Utils.hpp"

namespace torch::comms {

c10::intrusive_ptr<c10d::Store> createPrefixStore(
    const std::string& prefix,
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
  (void)comm_size;

  c10d::TCPStoreOptions opts;
  opts.port = port;
  opts.isServer = (rank == 0);
  opts.waitWorkers = false;
  opts.useLibUV = true;
  opts.timeout = timeout;

  return c10::make_intrusive<c10d::PrefixStore>(
      prefix, c10::make_intrusive<c10d::TCPStore>(host, opts));
}

c10::intrusive_ptr<c10d::Store> dupPrefixStore(
    const std::string& prefix,
    const c10::intrusive_ptr<c10d::Store>& bootstrapStore,
    std::chrono::milliseconds timeout) {
  // Open an INDEPENDENT client connection to the same store server the
  // bootstrap store already talks to, and wrap it under `prefix`. Because the
  // clone is a distinct TCPStore object, it has its own socket and its own
  // activeOpLock_, so this comm's blocking rendezvous (gloo connectFullMesh)
  // never shares a serialized channel with other comms/backends on the parent
  // store -- eliminating the head-of-line blocking a shared store causes.
  //
  // No per-comm server is stood up, so no server election or address exchange
  // is needed: the job's existing (rank 0) server serves every comm. A subgroup
  // that excludes global rank 0 is served fine, because rank 0 is the store
  // *server*, not a *participant* in the subgroup's rendezvous.
  auto* prefixStore = dynamic_cast<c10d::PrefixStore*>(bootstrapStore.get());
  TORCH_INTERNAL_ASSERT(
      prefixStore != nullptr,
      "dupPrefixStore expects a PrefixStore-wrapped bootstrap store");

  // Clone the underlying (non-prefix) store rather than the PrefixStore, so the
  // fresh connection is unprefixed and re-wrapping below does not
  // double-prefix.
  auto clientConn = prefixStore->getUnderlyingNonPrefixStore()->clone();
  clientConn->setTimeout(timeout);

  return c10::make_intrusive<c10d::PrefixStore>(prefix, clientConn);
}

} // namespace torch::comms
