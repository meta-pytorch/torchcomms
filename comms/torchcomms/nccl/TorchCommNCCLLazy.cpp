// Copyright (c) Meta Platforms, Inc. and affiliates.

// "nccl-lazy" backend: wraps TorchCommNCCL in LazyBackend so each
// point-to-point peer pair gets its own 2-rank ncclComm (and therefore
// its own CUDA stream), matching the per-pair affinity of the older c10d
// ProcessGroupNCCL.
//
// This file owns both halves of the lazy plumbing:
//   - the registration that exposes "nccl-lazy" through TorchCommFactory
//   - the out-of-line definition of TorchCommNCCL::createPairComm, which
//     bootstraps the 2-rank sibling comm via a PrefixStore + ncclUniqueId
//     exchange and ncclCommInitRankConfig. The caller (LazyBackend)
//     supplies a unique `name` for each invocation; this file just uses
//     it as the PrefixStore prefix.

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <c10/cuda/CUDAException.h>

#include <fmt/core.h>
#include <nccl.h> // @manual

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/lazy/LazyBackend.hpp"
#include "comms/torchcomms/nccl/TorchCommNCCL.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

namespace torch::comms {

std::shared_ptr<TorchCommNCCL> TorchCommNCCL::createPairComm(
    int peer_rank,
    const std::string& name) {
  checkInitialized();
  if (peer_rank == rank_ || peer_rank < 0 || peer_rank >= comm_size_) {
    throw std::runtime_error(
        fmt::format(
            "TorchCommNCCL::createPairComm: invalid peer rank {} (self={}, size={})",
            peer_rank,
            rank_,
            comm_size_));
  }

  const bool is_lower = (rank_ < peer_rank);
  const int pair_rank = is_lower ? 0 : 1;

  // Use the bootstrap store preserved from init, wrapped in a PrefixStore
  // keyed by the pair name. This avoids creating a new TCPStore from env
  // vars (MASTER_ADDR/MASTER_PORT), which fails in multiprocessing contexts
  // where the env-var store is unreachable or uses a different rank mapping.
  TORCH_CHECK(
      bootstrap_store_,
      "TorchCommNCCL::createPairComm: no bootstrap store available for ",
      name,
      " — was the comm initialized with a store?");
  auto store = c10::make_intrusive<c10d::PrefixStore>(name, bootstrap_store_);

  ncclUniqueId unique_id;
  static const char* kUniqueIdKey = "unique_id";
  if (is_lower) {
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->getUniqueId(&unique_id),
        "getUniqueId failed for pair comm");
    store->set(
        kUniqueIdKey,
        std::vector<uint8_t>(
            reinterpret_cast<uint8_t*>(&unique_id),
            reinterpret_cast<uint8_t*>(&unique_id) + sizeof(unique_id)));
  } else {
    store->wait({kUniqueIdKey}, options_.timeout);
    auto vec = store->get(kUniqueIdKey);
    TORCH_CHECK(
        vec.size() == sizeof(ncclUniqueId),
        "TorchCommNCCL::createPairComm: bad unique id size for ",
        name);
    std::memcpy(&unique_id, vec.data(), sizeof(ncclUniqueId));
  }

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  config.commName = name.c_str();
#endif

  // Drain pending NCCL kernels on this device before creating a new comm.
  // ncclCommInitRankConfig serializes with in-flight operations on other
  // comms (e.g. a world-group barrier allreduce), so we must ensure they
  // have completed to avoid blocking the init handshake.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  ncclComm_t pair_comm = nullptr;
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commInitRankConfig(
          &pair_comm, /*nranks=*/2, unique_id, pair_rank, &config),
      "commInitRankConfig failed for pair comm");

  // Use the private raw-comm constructor: we're a member function of
  // TorchCommNCCL, so we have access. Inherit the parent's options
  // wholesale but strip anything tied to its bootstrap so init() takes
  // the "comm already supplied" fast path.
  auto sub = std::shared_ptr<TorchCommNCCL>(new TorchCommNCCL(pair_comm));
  sub->setNcclApi(nccl_api_);
  CommOptions sub_opts = options_;
  sub_opts.store.reset();
  sub_opts.enable_reconfigure = false;
  sub->init(device_, name, sub_opts);

  TC_LOG(INFO, this) << "Built P2P pair comm " << name << " (peer=" << peer_rank
                     << ", pair_rank=" << pair_rank << ")";
  return sub;
}

// Explicitly instantiate so the vtable + out-of-line members live in this
// translation unit (and inside the same DSO as TorchCommNCCL).
template class LazyBackend<TorchCommNCCL>;

using TorchCommNCCLLazy = LazyBackend<TorchCommNCCL>;

namespace {

class NCCLLazyRegistration {
 public:
  NCCLLazyRegistration() {
    TorchCommFactory::get().register_backend(
        "nccl-lazy", []() { return std::make_shared<TorchCommNCCLLazy>(); });
  }
};

static const NCCLLazyRegistration registration{};

} // namespace
} // namespace torch::comms
