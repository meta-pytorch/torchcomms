// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/core/ivalue.h> // @manual=//caffe2:ATen-core
#include <torch/csrc/distributed/c10d/Backend.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/Work.hpp> // @manual=//caffe2:torch-cpp-cpu
#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchCommBatch.hpp"
#include "comms/torchcomms/TorchCommTypes.hpp"
#include "comms/torchcomms/TorchWork.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

namespace torch::comms {

class WorkWrapper : public c10d::Work {
 public:
  explicit WorkWrapper(
      c10::intrusive_ptr<TorchWork> work,
      std::vector<at::Tensor> outputTensors = {},
      bool hostBlocking = false);
  explicit WorkWrapper(
      std::vector<c10::intrusive_ptr<TorchWork>> works,
      std::vector<at::Tensor> outputTensors = {},
      bool hostBlocking = false);
  ~WorkWrapper() override = default;

  void synchronize() override;
  void blockCurrentStream() override;
  bool isCompleted() override;
  bool isSuccess() const override;
  std::exception_ptr exception() const override;
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
  std::vector<at::Tensor> result() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override;

 private:
  struct CompletionState {
    std::mutex mutex;
    std::condition_variable cv;
    std::optional<TorchWork::WorkStatus> status;
    TorchWork::WorkStatus aggregate{TorchWork::WorkStatus::COMPLETED};
    size_t remaining{0};
    bool terminalPublished{false};
    std::vector<at::Tensor> outputTensors;
  };

  TorchWork::WorkStatus refreshStatus();
  static void joinWorkNoThrow(
      const c10::intrusive_ptr<TorchWork>& work) noexcept;
  static void joinWorksNoThrow(
      const std::vector<c10::intrusive_ptr<TorchWork>>& works) noexcept;
  void terminalizeFailure(std::exception_ptr error) noexcept;

  friend class BackendWrapper;
  std::vector<c10::intrusive_ptr<TorchWork>> works_;
  c10::intrusive_ptr<c10::ivalue::Future> completionFuture_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
  c10::intrusive_ptr<c10::ivalue::Future> resultFuture_;
  std::shared_ptr<CompletionState> completionState_;
  std::vector<at::Tensor> outputTensors_;
  // When set, wait()/synchronize() call hostSynchronize() after each
  // stream-ordered wait.
  bool hostBlocking_;
};

using c10d::kUnsetTimeout;

class BackendWrapper : public c10d::Backend {
 public:
  struct TORCH_API Options : c10d::Backend::Options {
    bool abort_process_on_timeout_or_error{true};
    std::chrono::milliseconds timeout{kDefaultTimeout};
    bool is_high_priority_stream{false};
    c10::intrusive_ptr<c10d::Store> store{nullptr};
    std::unordered_map<std::string, std::string> hints;

    explicit Options() : c10d::Backend::Options("torchcomms") {}
  };

  explicit BackendWrapper(std::shared_ptr<TorchComm> comm);
  ~BackendWrapper() override = default;

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;
  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;
  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override;
  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;
  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& output_lists,
      std::vector<at::Tensor>& input_list,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;
  // Older 2.14 nightlies do not expose this virtual despite their version.
  // @lint-ignore CLANGTIDY clang-diagnostic-suggest-override
  // @lint-ignore CLANGTIDY facebook-hte-MissingOverride
  c10::intrusive_ptr<c10d::Work> gather_into_tensor(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const c10d::GatherOptions& opts = c10d::GatherOptions());
  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;
  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;
  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;
  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;
  // Health-checking barrier used by torch.distributed.monitored_barrier. Only
  // meaningful on the gloo (CPU) backend; reimplements
  // c10d::ProcessGroupGloo::monitoredBarrier on top of TorchComms P2P.
  void monitoredBarrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions(),
      bool waitAllRanks = false) override;
  c10::intrusive_ptr<c10d::Work>
  send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override;
  c10::intrusive_ptr<c10d::Work>
  recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override;

  // Coalescing hooks: c10d's _coalescing_manager (used by
  // dist.batch_isend_irecv) calls these around send/recv operations so the
  // backend can issue them as one group and avoid mixed-P2P deadlocks.
  bool supportsCoalescing() const override {
    return true;
  }
  void startCoalescing() override;
  c10::intrusive_ptr<c10d::Work> endCoalescing() override;

  // Get the underlying backend comm for backend-specific operations
  std::shared_ptr<TorchComm> getComm() const;

  // Returns the symmetric (VMM-backed) CUDA allocator associated with this
  // communicator's backend. See `TorchComm::getMemAllocator()`.
  std::shared_ptr<c10::Allocator> getMemAllocator() override;

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  const std::string getBackendName() const override;

  // c10d does not have a getBackendVersion method so no override required here
  std::string_view getBackendVersion() const;

  c10::intrusive_ptr<c10d::Backend::Options> getBackendOptions() override;

  // Verify that a work object has the expected timeout.
  // Used for testing timeout propagation.
  bool verifyWorkTimeoutForTest(
      const c10::intrusive_ptr<c10d::Work>& work,
      const std::chrono::milliseconds& timeout);

  // Set the default timeout for this backend.
  void setTimeout(std::chrono::milliseconds timeout) override;

  // Split communicator into a subgroup and return a new BackendWrapper
  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<c10d::Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<c10d::Backend::Options>& opts) override;

  // Called by torch.distributed.destroy_process_group(). Calls
  // TorchComm::finalize() to drain in-flight work and close the comm
  // gracefully — without this override the inherited base no-op leaves the
  // communicator alive and the destructor's synchronous ncclCommDestroy
  // can deadlock against the NCCL GC thread holding Work refs.
  void shutdown() override;

  // Called by destroy_process_group when the user wants forceful teardown.
  // Delegates to TorchComm::abort() which uses graceful revoke in
  // reconfigurable mode and destructive abort otherwise.
  void abort() override;

 protected:
  class CoalescingOperationScope {
   public:
    ~CoalescingOperationScope() noexcept;
    CoalescingOperationScope(const CoalescingOperationScope&) = delete;
    CoalescingOperationScope& operator=(const CoalescingOperationScope&) =
        delete;
    CoalescingOperationScope(CoalescingOperationScope&&) = delete;
    CoalescingOperationScope& operator=(CoalescingOperationScope&&) = delete;

    void dismiss();

   private:
    friend class BackendWrapper;
    explicit CoalescingOperationScope(BackendWrapper* owner);
    BackendWrapper* owner_;
  };

  CoalescingOperationScope coalescingOperationScope();

  enum class TensorPairAliasPolicy {
    DISALLOW,
    ALL_GATHER_RANK_SLICE,
  };

  static void validateCoalescedTensors(
      const std::vector<at::Tensor>& tensors,
      std::string_view operation);
  void validateCoalescedTensorPairs(
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      int64_t inputMultiplier,
      int64_t outputMultiplier,
      std::string_view operation,
      TensorPairAliasPolicy aliasPolicy =
          TensorPairAliasPolicy::DISALLOW) const;

  template <typename Launch>
  static std::vector<c10::intrusive_ptr<TorchWork>> launchWorks(
      size_t count,
      Launch&& launch) {
    std::vector<c10::intrusive_ptr<TorchWork>> works;
    works.reserve(count);
    try {
      for (size_t index = 0; index < count; ++index) {
        auto work = launch(index);
        TORCH_CHECK(work, "TorchComms returned a null work");
        works.push_back(std::move(work));
      }
    } catch (...) {
      WorkWrapper::joinWorksNoThrow(works);
      throw;
    }
    return works;
  }

  template <typename Launch>
  c10::intrusive_ptr<c10d::Work> launchAndWrapWork(
      Launch&& launch,
      std::vector<at::Tensor> outputTensors = {},
      bool hostBlocking = false) {
    prepareCollectiveForCoalescing();
    try {
      return wrapWork(
          std::forward<Launch>(launch)(),
          std::move(outputTensors),
          hostBlocking);
    } catch (...) {
      if (coalescing_batch_.has_value()) {
        if (active_coalescing_scope_ != nullptr) {
          active_coalescing_scope_->dismiss();
        }
        resetCoalescingState();
      }
      throw;
    }
  }

  c10::intrusive_ptr<c10d::Work> wrapWork(
      c10::intrusive_ptr<TorchWork> work,
      std::vector<at::Tensor> outputTensors = {},
      bool hostBlocking = false);
  c10::intrusive_ptr<c10d::Work> wrapWork(
      const std::vector<c10::intrusive_ptr<TorchWork>>& works,
      std::vector<at::Tensor> outputTensors = {},
      bool hostBlocking = false);

 private:
  void prepareCollectiveForCoalescing();
  void resetCoalescingState() noexcept;
  std::shared_ptr<TorchComm> comm_;
  c10::intrusive_ptr<Options> options_;

  // One active window may contain deferred P2P or launched collectives.
  std::optional<BatchSendRecv> coalescing_batch_;
  std::vector<c10::intrusive_ptr<WorkWrapper>> coalesced_collective_wrappers_;
  std::vector<at::Tensor> coalesced_collective_outputs_;
  CoalescingOperationScope* active_coalescing_scope_{nullptr};

  // Per-call tag sequence for monitoredBarrier's check-in/ack P2P. Kept
  // per-BackendWrapper (not process-global) so each ProcessGroup owns its own
  // counter: monitoredBarrier is collective per PG, so every rank advances
  // this instance's counter in lockstep and derives identical tags. A shared
  // process-global counter could instead be advanced a different number of
  // times per rank when unrelated PGs run concurrent barriers, yielding
  // mismatched send/recv tags across ranks.
  std::atomic<uint32_t> monitoredBarrierTagCounter_{0};
};

} // namespace torch::comms
