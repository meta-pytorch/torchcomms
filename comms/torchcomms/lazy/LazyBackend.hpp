// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>

#include <comms/torchcomms/TorchCommBackend.hpp>
#include <comms/torchcomms/TorchCommBatch.hpp>
#include <comms/torchcomms/TorchCommHooks.hpp>
#include <comms/torchcomms/TorchCommOptions.hpp>
#include <comms/torchcomms/TorchCommTypes.hpp>
#include <comms/torchcomms/TorchCommWindow.hpp>
#include <comms/torchcomms/TorchWork.hpp>

namespace torch::comms {

/**
 * LazyBackend wraps an underlying TorchCommBackend implementation of type T
 * and lazily creates per-peer 2-rank sibling communicators dedicated to
 * point-to-point traffic.
 *
 * Motivation: in older versions of PyTorch's ProcessGroupNCCL, each
 * point-to-point peer pair owned its own 2-rank ncclComm (and therefore
 * its own stream), bootstrapped on first send/recv between the two ranks
 * without any global coordination. send/recv to different peers could
 * overlap with each other and with collectives. LazyBackend reproduces
 * that behaviour: collectives stay on the primary comm, while P2P traffic
 * to peer X transparently uses a 2-rank sub-comm built on demand.
 *
 * Contract on T: this wrapper is generic but expects T to expose
 *   std::shared_ptr<T> T::createPairComm(int peer_rank,
 *                                        const std::string& name);
 * which must return a fully initialized 2-rank sibling communicator for
 * the pair (this->getRank(), peer_rank), bootstrapped out of band (not
 * via split — split() is collective on the parent, which makes it
 * unsuitable for true lazy creation by only the two participating
 * ranks). Inside the returned comm, the lower-numbered global rank is
 * local rank 0 and the higher is local rank 1. The wrapper supplies a
 * `name` that is unique across allocations for the same {min,max} pair
 * so any store-based bootstrap inside createPairComm can safely use
 * `name` as its key prefix without collisions across LazyBackend
 * instances within the same process. T must also expose
 *   void T::setBootstrapStore(c10::intrusive_ptr<c10d::Store>);
 * which LazyBackend calls once, right after init(), to hand the primary
 * the store its createPairComm needs — the primary's own init() releases
 * the store, so only lazy primaries keep it alive.
 *
 * Lifetime: pair comms are cached in a map keyed by the remote peer's
 * global rank. finalize() drains and tears them all down before the
 * primary.
 *
 * Dispatch: send(dst)/recv(src) are routed to the pair comm for that
 * peer. The peer's rank within the 2-rank pair comm is
 * (self < peer) ? 1 : 0 (the opposite of self's index).
 *
 * batch_op_issue stays on the primary because grouped ncclSend/Recv may
 * touch multiple peers in a single ncclGroupStart/End.
 *
 * Thread safety: user-facing collective/p2p calls still follow the base
 * TorchCommBackend contract — the caller serializes them. The wrapper
 * additionally takes an internal mutex around all access to the pair
 * map so abort() (which may fire from a background watchdog) and the
 * other fan-out paths (hooks, reconfigure, tensor (de)registration) can
 * iterate the map without racing with a concurrent first-touch
 * channelFor on the user thread. The lock is intentionally released
 * around the slow createPairComm bootstrap so abort() stays effectively
 * non-blocking.
 */
template <typename T>
class LazyBackend : public TorchCommBackend {
 public:
  static_assert(
      std::is_base_of_v<TorchCommBackend, T>,
      "LazyBackend<T> requires T to derive from TorchCommBackend");

  LazyBackend() : device_(at::kCPU) {}

  ~LazyBackend() override = default;

  LazyBackend(const LazyBackend&) = delete;
  LazyBackend(LazyBackend&&) = delete;
  LazyBackend& operator=(const LazyBackend&) = delete;
  LazyBackend& operator=(LazyBackend&&) = delete;

  // ---------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------

  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override {
    if (initialized_) {
      throw std::runtime_error("LazyBackend already initialized");
    }

    device_ = device;
    name_ = name;
    options_ = options;

    primary_ = std::make_shared<T>();
    primary_->init(device, name, options);
    // Hand the bootstrap store to the primary so lazy pair-comm creation
    // (createPairComm) can exchange ncclUniqueIds through it. Only lazy
    // primaries retain the store; init() itself releases it, so plain comms
    // free their TCPStore port once the caller drops its reference. We keep
    // our own reference alive in options_ for the LazyBackend's lifetime.
    primary_->setBootstrapStore(options.store);
    device_ = primary_->getDevice();
    rank_ = primary_->getRank();
    size_ = primary_->getSize();

    initialized_ = true;
  }

  void finalize() override {
    checkInitialized();
    // Drain any pending P2P work before tearing down the primary, since
    // some backends share global state (memory hooks, watchdog) between
    // siblings.
    std::unordered_map<int, std::shared_ptr<T>> drained;
    {
      std::lock_guard<std::mutex> lk(p2p_mu_);
      drained.swap(p2p_comms_);
    }
    for (auto& [_, channel] : drained) {
      channel->finalize();
    }
    primary_->finalize();
    initialized_ = false;
  }

  int getRank() const override {
    checkInitialized();
    return rank_;
  }

  int getSize() const override {
    checkInitialized();
    return size_;
  }

  std::string_view getBackendName() const override {
    return primary_ ? primary_->getBackendName() : std::string_view{};
  }

  std::string_view getCommName() const override {
    return name_;
  }

  // ---------------------------------------------------------------
  // Point-to-Point — dispatched to per-peer 2-rank pair comms
  // ---------------------------------------------------------------

  c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) override {
    return channelFor(dst).send(tensor, peerInPair(dst), async_op, options);
  }

  c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) override {
    return channelFor(src).recv(tensor, peerInPair(src), async_op, options);
  }

  // Batch P2P may touch multiple peers within a single ncclGroupStart/End,
  // so it has to share one comm — keep it on the primary.
  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) override {
    return primary_->batch_op_issue(ops, async_op, options);
  }

  // ---------------------------------------------------------------
  // Collectives — forwarded to the primary comm
  // ---------------------------------------------------------------

  c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) override {
    return primary_->broadcast(tensor, root, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) override {
    return primary_->all_reduce(tensor, op, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {}) override {
    return primary_->reduce(tensor, root, op, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override {
    return primary_->all_gather(tensor_list, tensor, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override {
    return primary_->all_gather_v(tensor_list, tensor, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) override {
    return primary_->all_gather_single(output, input, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override {
    return primary_->reduce_scatter(output, input_list, op, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override {
    return primary_->reduce_scatter_v(
        output, input_list, op, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) override {
    return primary_->reduce_scatter_single(
        output, input, op, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) override {
    return primary_->all_to_all_single(output, input, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) override {
    return primary_->all_to_all_v_single(
        output,
        input,
        output_split_sizes,
        input_split_sizes,
        async_op,
        options);
  }

  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) override {
    return primary_->all_to_all(
        output_tensor_list, input_tensor_list, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) override {
    return primary_->barrier(async_op, options);
  }

  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) override {
    return primary_->scatter(
        output_tensor, input_tensor_list, root, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) override {
    return primary_->gather(
        output_tensor_list, input_tensor, root, async_op, options);
  }

  c10::intrusive_ptr<TorchWork> gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      int root,
      bool async_op,
      const GatherSingleOptions& options = {}) override {
    return primary_->gather_single(output, input, root, async_op, options);
  }

  // ---------------------------------------------------------------
  // Communicator management
  // ---------------------------------------------------------------

  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override {
    return primary_->split(ranks, name, options);
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    checkInitialized();
    primary_->setTimeout(timeout);
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->setTimeout(timeout);
    }
  }

  const CommOptions& getOptions() const override {
    return options_;
  }

  const at::Device& getDevice() const override {
    return device_;
  }

  // ---------------------------------------------------------------
  // Window & one-sided ops
  // ---------------------------------------------------------------

  std::shared_ptr<TorchCommWindow> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override {
    return primary_->new_window(tensor);
  }

  // ---------------------------------------------------------------
  // Hooks — fan out to primary + all currently-allocated pair comms so
  // user-registered hooks observe events from every comm we own.
  // ---------------------------------------------------------------

  void registerAbortHook(int64_t hookId, AbortHook hook) override {
    TorchCommBackend::registerAbortHook(hookId, hook);
    if (primary_) {
      primary_->registerAbortHook(hookId, hook);
    }
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->registerAbortHook(hookId, hook);
    }
  }

  void unregisterAbortHook(int64_t hookId) override {
    TorchCommBackend::unregisterAbortHook(hookId);
    if (primary_) {
      primary_->unregisterAbortHook(hookId);
    }
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->unregisterAbortHook(hookId);
    }
  }

  void registerGraphReplayHook(int64_t hookId, GraphReplayHook hook) override {
    TorchCommBackend::registerGraphReplayHook(hookId, hook);
    if (primary_) {
      primary_->registerGraphReplayHook(hookId, hook);
    }
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->registerGraphReplayHook(hookId, hook);
    }
  }

  void unregisterGraphReplayHook(int64_t hookId) override {
    TorchCommBackend::unregisterGraphReplayHook(hookId);
    if (primary_) {
      primary_->unregisterGraphReplayHook(hookId);
    }
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->unregisterGraphReplayHook(hookId);
    }
  }

  // ---------------------------------------------------------------
  // Fault tolerance — abort/reconfigure must reach every comm we own.
  // ---------------------------------------------------------------

  bool supportsReconfigure() const override {
    return primary_ ? primary_->supportsReconfigure() : false;
  }

  bool isAbortSupported() const override {
    return primary_ ? primary_->isAbortSupported() : false;
  }

  bool isAborted() const override {
    if (primary_ && primary_->isAborted()) {
      return true;
    }
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (const auto& [_, channel] : p2p_comms_) {
      if (channel->isAborted()) {
        return true;
      }
    }
    return false;
  }

  bool isInitialized() const override {
    if (!initialized_ || !primary_) {
      return false;
    }
    return primary_->isInitialized();
  }

  InitHandle getInitHandle() const override {
    return primary_->getInitHandle();
  }

  c10::intrusive_ptr<TorchWork> reconfigure(
      const ReconfigureOptions& opts) override {
    auto work = primary_->reconfigure(opts);
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->reconfigure(opts);
    }
    return work;
  }

  // abort() is allowed to fire from a background watchdog thread.  We
  // hold p2p_mu_ only long enough to walk the pair map; each child's
  // abort() is itself non-blocking per the base contract, so the lock
  // is released quickly.
  void abort() override {
    if (primary_) {
      primary_->abort();
    }
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->abort();
    }
  }

  // ---------------------------------------------------------------
  // Tensor registration — register on every comm so the buffer is
  // usable from both collectives and P2P paths.
  // ---------------------------------------------------------------

  void tensor_register(const at::Tensor& tensor) override {
    primary_->tensor_register(tensor);
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->tensor_register(tensor);
    }
  }

  void tensor_deregister(const at::Tensor& tensor) override {
    primary_->tensor_deregister(tensor);
    std::lock_guard<std::mutex> lk(p2p_mu_);
    for (auto& [_, channel] : p2p_comms_) {
      channel->tensor_deregister(tensor);
    }
  }

  int64_t get_device_transport() override {
    return primary_->get_device_transport();
  }

  // ---------------------------------------------------------------
  // Test / introspection helpers
  // ---------------------------------------------------------------

  // Returns the currently-allocated pair comm for the given peer, or
  // nullptr if none has been created yet. Does NOT trigger lazy
  // creation — exposed for tests that want to inspect what got built.
  std::shared_ptr<T> getPeerChannel(int peer) const {
    std::lock_guard<std::mutex> lk(p2p_mu_);
    auto it = p2p_comms_.find(peer);
    return it == p2p_comms_.end() ? nullptr : it->second;
  }

  std::shared_ptr<T> getPrimary() const {
    return primary_;
  }

  size_t numActiveChannels() const {
    std::lock_guard<std::mutex> lk(p2p_mu_);
    return p2p_comms_.size();
  }

 private:
  void checkInitialized() const {
    if (!initialized_) {
      throw std::runtime_error("LazyBackend not initialized");
    }
  }

  // Returns the peer's local rank in the 2-rank pair comm. By
  // PairFactory contract, the lower-numbered global rank is local rank 0
  // and the higher is local rank 1; the local peer index is therefore
  // the opposite of self's index.
  int peerInPair(int peer) const {
    return (rank_ < peer) ? 1 : 0;
  }

  T& channelFor(int peer) {
    checkInitialized();
    {
      std::lock_guard<std::mutex> lk(p2p_mu_);
      auto it = p2p_comms_.find(peer);
      if (it != p2p_comms_.end()) {
        return *it->second;
      }
    }

    // Slow path: bootstrap the pair comm without holding the lock so
    // abort() and other map-walking paths stay responsive while the
    // store-based ncclUniqueId exchange is in flight. Under the
    // base-class single-threaded user contract no other thread is
    // racing us on this peer, so re-inserting after the I/O is safe.
    const int lo = std::min(rank_, peer);
    const int hi = std::max(rank_, peer);
    const std::string pair_name = name_ + "/p2p-" + std::to_string(lo) + "-" +
        std::to_string(hi) + "-" + std::to_string(nextPairAttempt(lo, hi));

    auto sub = primary_->createPairComm(peer, pair_name);
    if (!sub) {
      throw std::runtime_error(
          "LazyBackend: createPairComm returned nullptr for peer " +
          std::to_string(peer));
    }

    std::lock_guard<std::mutex> lk(p2p_mu_);
    auto [inserted_it, ok] = p2p_comms_.emplace(peer, std::move(sub));
    return *inserted_it->second;
  }

  // Per-pair monotonically increasing counter so that successive
  // pair-comm allocations for the same {lo,hi} produce distinct names
  // (and therefore distinct store-bootstrap key namespaces inside
  // createPairComm). Both ranks of a pair increment in lockstep
  // because each create call on one side has a matching create call
  // on the other, so the counters stay in sync without explicit
  // coordination. Static so it's shared across all LazyBackend<T>
  // instances within the process.
  static int nextPairAttempt(int lo, int hi) {
    static std::mutex mu;
    static std::unordered_map<int64_t, int> counters;
    const int64_t key = (static_cast<int64_t>(lo) << 32) |
        static_cast<int64_t>(static_cast<uint32_t>(hi));
    std::lock_guard<std::mutex> guard(mu);
    return counters[key]++;
  }

  std::shared_ptr<T> primary_;
  // Mutex protects p2p_comms_ against background-thread access from
  // abort() and isAborted() (the only two TorchCommBackend methods
  // permitted to fire from outside the main thread).
  mutable std::mutex p2p_mu_;
  std::unordered_map<int, std::shared_ptr<T>> p2p_comms_;
  CommOptions options_;
  at::Device device_;
  std::string name_;
  int rank_{-1};
  int size_{-1};
  bool initialized_{false};
};

} // namespace torch::comms
