// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/BackendWrapper.hpp"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

#include <ATen/MemoryOverlap.h> // @manual=//caffe2:torch-cpp-cpu
#include <c10/core/DeviceGuard.h> // @manual=//caffe2:c10

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>

namespace torch::comms {

namespace {

// Extract the scaling factor from NCCL's PREMUL_SUM operation supplement.
// NCCLPreMulSumSupplement stores either a tensor or double scaling factor
// that is applied before summation.
PreMulSumFactorT getPreMulSumFactor(const c10d::ReduceOp& op) {
  TORCH_CHECK(
      op.supplement_ != nullptr,
      "PREMUL_SUM operation requires a supplement, but none was provided");

  const auto* preMulSupplement =
      dynamic_cast<const c10d::NCCLPreMulSumSupplement*>(op.supplement_.get());
  TORCH_CHECK(
      preMulSupplement != nullptr,
      "PREMUL_SUM operation supplement must be of type NCCLPreMulSumSupplement");

  if (preMulSupplement->tensor_factor.defined()) {
    return preMulSupplement->tensor_factor;
  }
  return preMulSupplement->double_factor;
}

ReduceOp toReduceOp(const c10d::ReduceOp& op) {
  switch (op) {
    case c10d::ReduceOp::SUM:
      return ReduceOp::SUM;
    case c10d::ReduceOp::AVG:
      return ReduceOp::AVG;
    case c10d::ReduceOp::MIN:
      return ReduceOp::MIN;
    case c10d::ReduceOp::MAX:
      return ReduceOp::MAX;
    case c10d::ReduceOp::PRODUCT:
      return ReduceOp::PRODUCT;
    case c10d::ReduceOp::BAND:
      return ReduceOp::BAND;
    case c10d::ReduceOp::BOR:
      return ReduceOp::BOR;
    case c10d::ReduceOp::BXOR:
      return ReduceOp::BXOR;
    case c10d::ReduceOp::PREMUL_SUM:
      return ReduceOp::make_nccl_premul_sum(getPreMulSumFactor(op));
    default:
      throw std::runtime_error("Unsupported reduce op");
  }
}

std::vector<uint64_t> toVecUint64(const std::vector<int64_t>& vec) {
  std::vector<uint64_t> vecUint64;
  vecUint64.reserve(vec.size());
  for (auto i : vec) {
    TORCH_CHECK(i >= 0, "All-to-all split sizes must be non-negative");
    vecUint64.push_back(static_cast<uint64_t>(i));
  }
  return vecUint64;
}

void validatePairwiseDisjointTensors(
    const std::vector<at::Tensor>& tensors,
    std::string_view operation,
    std::string_view role) {
  for (size_t first = 0; first < tensors.size(); ++first) {
    for (size_t second = first + 1; second < tensors.size(); ++second) {
      TORCH_CHECK(
          at::get_overlap_status(tensors[first], tensors[second]) ==
              at::MemOverlapStatus::No,
          operation,
          " ",
          role,
          " tensors must not alias; indices ",
          first,
          " and ",
          second,
          " overlap or cannot be proven disjoint");
    }
  }
}

void validateDisjointTensorSets(
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs,
    std::string_view operation) {
  for (size_t input = 0; input < inputs.size(); ++input) {
    for (size_t output = 0; output < outputs.size(); ++output) {
      TORCH_CHECK(
          at::get_overlap_status(inputs[input], outputs[output]) ==
              at::MemOverlapStatus::No,
          operation,
          " input and output tensors must not alias; input ",
          input,
          " and output ",
          output,
          " overlap or cannot be proven disjoint");
    }
  }
}

bool isExactAllGatherRankSlice(
    const at::Tensor& input,
    const at::Tensor& output,
    int rank) {
  TORCH_INTERNAL_ASSERT(rank >= 0);
  if (!input.is_alias_of(output)) {
    return false;
  }
  if (rank != 0 && input.numel() > std::numeric_limits<int64_t>::max() / rank) {
    return false;
  }
  const auto rankOffsetElements = input.numel() * static_cast<int64_t>(rank);
  return output.storage_offset() <=
      std::numeric_limits<int64_t>::max() - rankOffsetElements &&
      input.storage_offset() == output.storage_offset() + rankOffsetElements;
}

int checkedRootRank(
    int64_t rootRank,
    int worldSize,
    std::string_view operation) {
  TORCH_CHECK(
      rootRank >= 0 && rootRank < static_cast<int64_t>(worldSize),
      operation,
      " root rank must be in [0, ",
      worldSize,
      "), got ",
      rootRank);
  return static_cast<int>(rootRank);
}

bool isTerminal(TorchWork::WorkStatus status) {
  return status == TorchWork::WorkStatus::COMPLETED ||
      status == TorchWork::WorkStatus::ERROR ||
      status == TorchWork::WorkStatus::TIMEDOUT;
}

std::exception_ptr exceptionForStatus(TorchWork::WorkStatus status) {
  if (status == TorchWork::WorkStatus::ERROR) {
    return std::make_exception_ptr(
        std::runtime_error("TorchComms operation failed"));
  }
  if (status == TorchWork::WorkStatus::TIMEDOUT) {
    return std::make_exception_ptr(
        std::runtime_error("TorchComms operation timed out"));
  }
  return nullptr;
}

std::optional<c10d::WorkResult> resultForStatus(TorchWork::WorkStatus status) {
  if (status == TorchWork::WorkStatus::COMPLETED) {
    return c10d::WorkResult::SUCCESS;
  }
  if (status == TorchWork::WorkStatus::TIMEDOUT) {
    return c10d::WorkResult::TIMEOUT;
  }
  if (status == TorchWork::WorkStatus::ERROR) {
    return c10d::WorkResult::COMM_ERROR;
  }
  return std::nullopt;
}

TorchWork::WorkStatus combineTerminalStatus(
    TorchWork::WorkStatus current,
    TorchWork::WorkStatus next) {
  if (current == TorchWork::WorkStatus::TIMEDOUT ||
      next == TorchWork::WorkStatus::TIMEDOUT) {
    return TorchWork::WorkStatus::TIMEDOUT;
  }
  if (current == TorchWork::WorkStatus::ERROR ||
      next == TorchWork::WorkStatus::ERROR) {
    return TorchWork::WorkStatus::ERROR;
  }
  return TorchWork::WorkStatus::COMPLETED;
}

TorchWork::WorkStatus aggregateStatus(
    const std::vector<c10::intrusive_ptr<TorchWork>>& works) {
  auto terminal = TorchWork::WorkStatus::COMPLETED;
  bool pending = false;
  for (const auto& work : works) {
    const auto status = work->status();
    if (status == TorchWork::WorkStatus::ERROR ||
        status == TorchWork::WorkStatus::TIMEDOUT) {
      terminal = combineTerminalStatus(terminal, status);
    } else if (status != TorchWork::WorkStatus::COMPLETED) {
      pending = true;
    }
  }
  if (terminal != TorchWork::WorkStatus::COMPLETED) {
    return terminal;
  }
  return pending ? TorchWork::WorkStatus::INPROGRESS
                 : TorchWork::WorkStatus::COMPLETED;
}

TorchWork::WorkStatus aggregatePolledStatus(
    const std::vector<c10::intrusive_ptr<TorchWork>>& works) {
  auto terminal = TorchWork::WorkStatus::COMPLETED;
  bool pending = false;
  for (const auto& work : works) {
    const auto current = work->status();
    const auto status = isTerminal(current) ? current : work->pollStatus();
    if (status == TorchWork::WorkStatus::ERROR ||
        status == TorchWork::WorkStatus::TIMEDOUT) {
      terminal = combineTerminalStatus(terminal, status);
    } else if (status != TorchWork::WorkStatus::COMPLETED) {
      pending = true;
    }
  }
  if (terminal != TorchWork::WorkStatus::COMPLETED) {
    return terminal;
  }
  return pending ? TorchWork::WorkStatus::INPROGRESS
                 : TorchWork::WorkStatus::COMPLETED;
}

bool supportsActivePolling(
    const std::vector<c10::intrusive_ptr<TorchWork>>& works) {
  return std::all_of(works.begin(), works.end(), [](const auto& work) {
    return work->supportsActivePolling() || isTerminal(work->status());
  });
}

constexpr auto kFiniteWaitPollInterval = std::chrono::milliseconds(1);

void synchronizeCoalescedWorkNoThrow(
    const c10::intrusive_ptr<WorkWrapper>& work) noexcept {
  try {
    work->synchronize();
  } catch (const std::exception& error) {
    TC_LOG(WARNING) << "Failed to synchronize coalesced work during cleanup: "
                    << error.what();
  } catch (...) {
    TC_LOG(WARNING) << "Failed to synchronize coalesced work during cleanup";
  }
}

} // namespace

WorkWrapper::WorkWrapper(
    c10::intrusive_ptr<TorchWork> work,
    std::vector<at::Tensor> outputTensors,
    bool hostBlocking)
    : WorkWrapper(
          std::vector<c10::intrusive_ptr<TorchWork>>{std::move(work)},
          std::move(outputTensors),
          hostBlocking) {}

WorkWrapper::WorkWrapper(
    std::vector<c10::intrusive_ptr<TorchWork>> works,
    std::vector<at::Tensor> outputTensors,
    bool hostBlocking)
    : works_(std::move(works)),
      completionState_(std::make_shared<CompletionState>()),
      outputTensors_(std::move(outputTensors)),
      hostBlocking_(hostBlocking) {
  TORCH_CHECK(!works_.empty(), "WorkWrapper requires at least one work");
  TORCH_CHECK(
      std::all_of(
          works_.begin(),
          works_.end(),
          [](const auto& work) { return work != nullptr; }),
      "WorkWrapper received a null work");
  completionState_->remaining = works_.size();
  std::vector<c10::Device> devices;
  if (!exceptionForStatus(aggregateStatus(works_))) {
    for (const auto& tensor : outputTensors_) {
      if (tensor.device().type() != c10::DeviceType::CPU) {
        devices.push_back(tensor.device());
        break;
      }
    }
  }
  completionFuture_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  resultFuture_ =
      c10::make_intrusive<c10::ivalue::Future>(c10::AnyEnumType::get());
  if (devices.empty()) {
    completionState_->outputTensors = outputTensors_;
  }
  completionFuture_->addCallback(
      [future = future_](c10::ivalue::Future& completionFuture) {
        if (future->completed()) {
          return;
        }
        try {
          if (completionFuture.hasError()) {
            future->setError(completionFuture.exception_ptr());
          } else {
            future->markCompleted(
                completionFuture.value(), completionFuture.storages());
          }
        } catch (...) {
          if (!future->completed()) {
            throw;
          }
        }
      });

  for (const auto& work : works_) {
    work->registerWorkEndHook([state = completionState_,
                               future = completionFuture_,
                               resultFuture = resultFuture_,
                               work = work.get(),
                               cpuFuture = devices.empty()]() {
      bool signal = false;
      std::optional<TorchWork::WorkStatus> finalStatus;
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->aggregate =
            combineTerminalStatus(state->aggregate, work->status());
        TORCH_INTERNAL_ASSERT(state->remaining > 0);
        --state->remaining;
        if (!state->terminalPublished &&
            (work->status() == TorchWork::WorkStatus::ERROR ||
             work->status() == TorchWork::WorkStatus::TIMEDOUT ||
             state->remaining == 0)) {
          state->status = state->aggregate;
          finalStatus = state->status;
          state->terminalPublished = true;
          signal = true;
        }
      }
      if (signal) {
        state->cv.notify_all();
      }
      if (!finalStatus.has_value()) {
        return;
      }
      if (const auto result = resultForStatus(*finalStatus);
          result.has_value()) {
        try {
          if (!resultFuture->completed()) {
            resultFuture->markCompleted(
                c10::IValue(static_cast<std::uint8_t>(*result)));
          }
        } catch (...) {
          if (!resultFuture->completed()) {
            throw;
          }
        }
      }
      if (!cpuFuture || future->completed()) {
        return;
      }
      try {
        if (const auto error = exceptionForStatus(*finalStatus)) {
          future->setError(error);
        } else {
          future->markCompleted(c10::IValue(state->outputTensors));
        }
      } catch (...) {
        if (!future->completed()) {
          throw;
        }
      }
    });
  }

  if (!devices.empty()) {
    if (const auto error = exceptionForStatus(aggregateStatus(works_))) {
      completionFuture_->setError(error);
    } else {
      // GPU getFuture reports enqueue ordering; getFutureResult carries health.
      // The final stream event covers earlier launches; waits visit every
      // child.
      works_.back()->markCompleted(
          c10::intrusive_ptr<c10::ivalue::Future>(completionFuture_),
          outputTensors_);
    }
  }
}

void WorkWrapper::terminalizeFailure(std::exception_ptr error) noexcept {
  for (const auto& work : works_) {
    if (!isTerminal(work->status())) {
      try {
        work->setStatus(TorchWork::WorkStatus::ERROR);
      } catch (...) {
      }
    }
  }
  try {
    finish(std::move(error));
  } catch (...) {
  }
}

TorchWork::WorkStatus WorkWrapper::refreshStatus() {
  try {
    return supportsActivePolling(works_) ? aggregatePolledStatus(works_)
                                         : aggregateStatus(works_);
  } catch (...) {
    terminalizeFailure(std::current_exception());
    return aggregateStatus(works_);
  }
}

bool WorkWrapper::isCompleted() {
  const auto status = refreshStatus();
  if (isTerminal(status) && !c10d::Work::isCompleted()) {
    finish(exceptionForStatus(status));
  }
  return isTerminal(status);
}

bool WorkWrapper::isSuccess() const {
  return aggregateStatus(works_) == TorchWork::WorkStatus::COMPLETED &&
      c10d::Work::isSuccess();
}

std::exception_ptr WorkWrapper::exception() const {
  auto error = c10d::Work::exception();
  return error ? error : exceptionForStatus(aggregateStatus(works_));
}

bool WorkWrapper::wait(std::chrono::milliseconds timeout) {
  if (timeout != kNoTimeout) {
    TORCH_CHECK(timeout.count() > 0, "Work wait timeout must be positive");
    TORCH_CHECK(
        supportsActivePolling(works_),
        "Finite Work wait is not supported by this TorchComms backend");
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    auto polledStatus = [this]() { return refreshStatus(); };
    while (!isTerminal(polledStatus())) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        TORCH_CHECK(isTerminal(refreshStatus()), "Operation timed out!");
        break;
      }
      std::unique_lock<std::mutex> lock(completionState_->mutex);
      completionState_->cv.wait_until(
          lock, std::min(deadline, now + kFiniteWaitPollInterval), [this]() {
            return completionState_->status.has_value();
          });
    }
  }
  synchronize();
  return true;
}

void WorkWrapper::synchronize() {
  auto firstError = c10d::Work::exception();
  for (const auto& work : works_) {
    if (const auto error = exceptionForStatus(work->status())) {
      if (!firstError) {
        firstError = error;
      }
      continue;
    }
    try {
      work->wait();
      if (hostBlocking_) {
        work->hostSynchronize();
      }
      if (auto error = exceptionForStatus(work->status())) {
        std::rethrow_exception(error);
      }
    } catch (...) {
      if (!firstError) {
        firstError = std::current_exception();
      }
    }
  }
  if (firstError) {
    terminalizeFailure(firstError);
    std::rethrow_exception(firstError);
  }
  finish();
}

void WorkWrapper::blockCurrentStream() {
  std::exception_ptr firstError;
  for (const auto& work : works_) {
    try {
      work->wait();
    } catch (...) {
      if (!firstError) {
        firstError = std::current_exception();
      }
    }
  }
  if (firstError) {
    terminalizeFailure(firstError);
    std::rethrow_exception(firstError);
  }
}

std::vector<at::Tensor> WorkWrapper::result() {
  return outputTensors_;
}
c10::intrusive_ptr<c10::ivalue::Future> WorkWrapper::getFuture() {
  return future_;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkWrapper::getFutureResult() {
  return resultFuture_;
}

BackendWrapper::BackendWrapper(std::shared_ptr<TorchComm> comm)
    : Backend(comm->getRank(), comm->getSize()),
      comm_(std::move(comm)),
      options_(c10::make_intrusive<Options>()) {}

BackendWrapper::CoalescingOperationScope::CoalescingOperationScope(
    BackendWrapper* owner)
    : owner_(owner) {
  if (owner_) {
    TORCH_INTERNAL_ASSERT(owner_->active_coalescing_scope_ == nullptr);
    owner_->active_coalescing_scope_ = this;
  }
}

BackendWrapper::CoalescingOperationScope::~CoalescingOperationScope() noexcept {
  if (owner_ && owner_->active_coalescing_scope_ == this) {
    owner_->active_coalescing_scope_ = nullptr;
    auto* owner = owner_;
    owner_ = nullptr;
    owner->resetCoalescingState();
  }
}

void BackendWrapper::CoalescingOperationScope::dismiss() {
  if (owner_) {
    TORCH_INTERNAL_ASSERT(owner_->active_coalescing_scope_ == this);
    owner_->active_coalescing_scope_ = nullptr;
    owner_ = nullptr;
  }
}

BackendWrapper::CoalescingOperationScope
BackendWrapper::coalescingOperationScope() {
  return CoalescingOperationScope(
      coalescing_batch_.has_value() ? this : nullptr);
}

void BackendWrapper::validateCoalescedTensors(
    const std::vector<at::Tensor>& tensors,
    std::string_view operation) {
  TORCH_CHECK(!tensors.empty(), operation, " requires a nonempty tensor list");
  TORCH_CHECK(
      tensors.front().defined(), operation, " received an undefined tensor");
  const auto device = tensors.front().device();
  const auto type = tensors.front().scalar_type();
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.defined(), operation, " received an undefined tensor");
    TORCH_CHECK(
        tensor.layout() == at::kStrided &&
            tensor.is_non_overlapping_and_dense() && tensor.is_contiguous(),
        operation,
        " requires contiguous non-overlapping dense tensors");
    TORCH_CHECK(
        !tensor.is_conj() && !tensor.is_neg(),
        operation,
        " does not support conjugate or negative views");
    TORCH_CHECK(
        tensor.device() == device,
        operation,
        " requires tensors on one device");
    TORCH_CHECK(
        tensor.scalar_type() == type,
        operation,
        " requires tensors with one dtype");
  }
}

void BackendWrapper::validateCoalescedTensorPairs(
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs,
    int64_t inputMultiplier,
    int64_t outputMultiplier,
    std::string_view operation,
    TensorPairAliasPolicy aliasPolicy) const {
  TORCH_CHECK(
      inputs.size() == outputs.size(),
      operation,
      " requires one output per input");
  TORCH_INTERNAL_ASSERT(inputMultiplier > 0 && outputMultiplier > 0);
  validateCoalescedTensors(inputs, operation);
  validateCoalescedTensors(outputs, operation);
  TORCH_CHECK(
      inputs.front().device() == outputs.front().device(),
      operation,
      " requires inputs and outputs on one device");
  TORCH_CHECK(
      inputs.front().scalar_type() == outputs.front().scalar_type(),
      operation,
      " requires matching input and output dtypes");
  for (size_t index = 0; index < inputs.size(); ++index) {
    TORCH_CHECK(
        inputs[index].numel() <=
                std::numeric_limits<int64_t>::max() / inputMultiplier &&
            outputs[index].numel() <=
                std::numeric_limits<int64_t>::max() / outputMultiplier &&
            inputs[index].numel() * inputMultiplier ==
                outputs[index].numel() * outputMultiplier,
        operation,
        " received incompatible input and output sizes at index ",
        index);
  }
  validatePairwiseDisjointTensors(outputs, operation, "output");
  if (aliasPolicy == TensorPairAliasPolicy::DISALLOW) {
    validateDisjointTensorSets(inputs, outputs, operation);
    return;
  }
  TORCH_INTERNAL_ASSERT(
      aliasPolicy == TensorPairAliasPolicy::ALL_GATHER_RANK_SLICE);
  for (size_t input = 0; input < inputs.size(); ++input) {
    for (size_t output = 0; output < outputs.size(); ++output) {
      const bool isAllowedRankSlice = input == output &&
          isExactAllGatherRankSlice(inputs[input], outputs[output], getRank());
      TORCH_CHECK(
          isAllowedRankSlice ||
              at::get_overlap_status(inputs[input], outputs[output]) ==
                  at::MemOverlapStatus::No,
          operation,
          " input and output tensors must not alias outside the input's "
          "rank-local output slice; input ",
          input,
          " and output ",
          output,
          " overlap or cannot be proven disjoint");
    }
  }
}

void WorkWrapper::joinWorkNoThrow(
    const c10::intrusive_ptr<TorchWork>& work) noexcept {
  if (!work) {
    return;
  }
  try {
    work->wait();
    return;
  } catch (const std::exception& error) {
    TC_LOG(WARNING) << "Failed to join work during wrapper cleanup: "
                    << error.what();
  } catch (...) {
    TC_LOG(WARNING) << "Failed to join work during wrapper cleanup";
  }
  if (isTerminal(work->status())) {
    return;
  }
  try {
    work->setStatus(TorchWork::WorkStatus::ERROR);
  } catch (const std::exception& error) {
    TC_LOG(WARNING) << "Failed to publish work error during wrapper cleanup: "
                    << error.what();
  } catch (...) {
    TC_LOG(WARNING) << "Failed to publish work error during wrapper cleanup";
  }
}

void WorkWrapper::joinWorksNoThrow(
    const std::vector<c10::intrusive_ptr<TorchWork>>& works) noexcept {
  for (const auto& work : works) {
    joinWorkNoThrow(work);
  }
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::wrapWork(
    c10::intrusive_ptr<TorchWork> work,
    std::vector<at::Tensor> outputTensors,
    bool hostBlocking) {
  TORCH_CHECK(work, "TorchComms returned a null work");
  std::vector<c10::intrusive_ptr<TorchWork>> works;
  try {
    works.reserve(1);
    works.push_back(work);
  } catch (...) {
    WorkWrapper::joinWorkNoThrow(work);
    throw;
  }
  return wrapWork(works, std::move(outputTensors), hostBlocking);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::wrapWork(
    const std::vector<c10::intrusive_ptr<TorchWork>>& works,
    std::vector<at::Tensor> outputTensors,
    bool hostBlocking) {
  TORCH_CHECK(!works.empty(), "Cannot wrap an empty work list");
  c10::intrusive_ptr<WorkWrapper> wrapped;
  try {
    wrapped = c10::make_intrusive<WorkWrapper>(
        works, std::move(outputTensors), hostBlocking);
  } catch (...) {
    WorkWrapper::joinWorksNoThrow(works);
    throw;
  }
  if (!coalescing_batch_.has_value()) {
    return wrapped;
  }
  try {
    coalesced_collective_wrappers_.push_back(wrapped);
  } catch (...) {
    synchronizeCoalescedWorkNoThrow(wrapped);
    throw;
  }
  coalesced_collective_outputs_.insert(
      coalesced_collective_outputs_.end(),
      wrapped->outputTensors_.begin(),
      wrapped->outputTensors_.end());
  TORCH_INTERNAL_ASSERT(active_coalescing_scope_ != nullptr);
  active_coalescing_scope_->dismiss();
  return wrapped;
}

void BackendWrapper::prepareCollectiveForCoalescing() {
  if (coalescing_batch_.has_value() && !coalescing_batch_->ops.empty()) {
    TORCH_CHECK(
        false,
        "A coalescing window cannot mix point-to-point and collective operations");
  }
}

void BackendWrapper::resetCoalescingState() noexcept {
  auto wrappers = std::move(coalesced_collective_wrappers_);
  coalesced_collective_wrappers_.clear();
  coalescing_batch_.reset();
  coalesced_collective_outputs_.clear();
  for (const auto& wrapper : wrappers) {
    synchronizeCoalescedWorkNoThrow(wrapper);
  }
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  const int rootRank = checkedRootRank(opts.rootRank, getSize(), "Broadcast");
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  TORCH_CHECK(opts.rootTensor == 0, "Only rootTensor 0 is supported");
  BroadcastOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return comm_->broadcast(tensors.at(0), rootRank, opts.asyncOp, bopts);
      },
      tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  AllReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  const auto reduceOp = toReduceOp(opts.reduceOp);
  return launchAndWrapWork(
      [&]() {
        return comm_->all_reduce(tensors.at(0), reduceOp, opts.asyncOp, bopts);
      },
      tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceCoalescedOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  validateCoalescedTensors(tensors, "Coalesced all-reduce");
  validatePairwiseDisjointTensors(tensors, "Coalesced all-reduce", "in-place");
  AllReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  const auto reduceOp = toReduceOp(opts.reduceOp);
  return launchAndWrapWork(
      [&]() {
        return launchWorks(tensors.size(), [&](size_t index) {
          return comm_->all_reduce(
              tensors[index], reduceOp, opts.asyncOp, bopts);
        });
      },
      tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce(
    std::vector<at::Tensor>& tensors,
    const c10d::ReduceOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  const int rootRank = checkedRootRank(opts.rootRank, getSize(), "Reduce");
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  TORCH_CHECK(opts.rootTensor == 0, "Only rootTensor 0 is supported");
  ReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return comm_->reduce(
            tensors.at(0),
            rootRank,
            toReduceOp(opts.reduceOp),
            opts.asyncOp,
            bopts);
      },
      tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor list supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  const auto& input = inputTensors.at(0);
  auto& outputList = outputTensors.at(0);
  TORCH_CHECK(
      static_cast<int>(outputList.size()) == getSize(),
      "Expected ",
      getSize(),
      " output tensors (one per rank), but got ",
      outputList.size());

  AllGatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }

  // Fast path: when per-rank output tensors point to distinct memory,
  // delegate straight to the backend's list-based all_gather (no extra
  // alloc/copy). Simply check the first two ranks.
  bool aliased = outputList.size() > 1 &&
      outputList[0].data_ptr() == outputList[1].data_ptr();
  if (!aliased) {
    return launchAndWrapWork(
        [&]() {
          return comm_->all_gather(outputList, input, opts.asyncOp, bopts);
        },
        outputList);
  }

  // Slow path (aliased outputs): each outputList[r] points to the same
  // K-element buffer, but the gather needs world_size * K bytes. Allocate
  // a contiguous staging tensor shaped {world_size, K}, gather into it,
  // then copy each rank's row back into the caller's per-rank tensor.
  AllGatherSingleOptions sopts;
  sopts.timeout = bopts.timeout;
  auto staging = at::empty(
      {getSize() * input.numel()},
      input.options().memory_format(at::MemoryFormat::Contiguous));
  auto work = launchAndWrapWork(
      [&]() {
        return comm_->all_gather_single(staging, input, opts.asyncOp, sopts);
      },
      outputList);
  try {
    work->wait(kNoTimeout);
    auto rows = staging.view({getSize(), input.numel()});
    for (int rank = 0; rank < getSize(); ++rank) {
      outputList.at(rank).copy_(rows[rank].view_as(outputList.at(rank)));
    }
  } catch (...) {
    if (coalescing_batch_.has_value()) {
      resetCoalescingState();
    }
    throw;
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  validateCoalescedTensors(inputTensors, "Coalesced all-gather");
  TORCH_CHECK(
      outputTensorLists.size() == inputTensors.size(),
      "Coalesced all-gather requires one output list per input tensor");
  std::vector<at::Tensor> outputs;
  for (const auto& outputList : outputTensorLists) {
    TORCH_CHECK(
        static_cast<int>(outputList.size()) == getSize(),
        "Coalesced all-gather requires one output tensor per rank");
    outputs.insert(outputs.end(), outputList.begin(), outputList.end());
  }
  validateCoalescedTensors(outputs, "Coalesced all-gather");
  for (size_t input = 0; input < inputTensors.size(); ++input) {
    for (const auto& output : outputTensorLists[input]) {
      TORCH_CHECK(
          output.device() == inputTensors[input].device() &&
              output.scalar_type() == inputTensors[input].scalar_type() &&
              output.sizes().equals(inputTensors[input].sizes()),
          "Coalesced all-gather requires each output to match its input");
    }
  }
  validatePairwiseDisjointTensors(outputs, "Coalesced all-gather", "output");
  validateDisjointTensorSets(inputTensors, outputs, "Coalesced all-gather");
  AllGatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return launchWorks(inputTensors.size(), [&](size_t index) {
          return comm_->all_gather(
              outputTensorLists[index],
              inputTensors[index],
              opts.asyncOp,
              bopts);
        });
      },
      outputs);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  validateCoalescedTensorPairs(
      inputTensors,
      output_tensors,
      getSize(),
      1,
      "Coalesced all-gather-into-tensor",
      TensorPairAliasPolicy::ALL_GATHER_RANK_SLICE);
  AllGatherSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return launchWorks(inputTensors.size(), [&](size_t index) {
          return comm_->all_gather_single(
              output_tensors[index], inputTensors[index], opts.asyncOp, bopts);
        });
      },
      output_tensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::AllgatherOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  validateCoalescedTensorPairs(
      {inputTensor},
      {outputTensor},
      getSize(),
      1,
      "All-gather-into-tensor",
      TensorPairAliasPolicy::ALL_GATHER_RANK_SLICE);
  AllGatherSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return comm_->all_gather_single(
            outputTensor, inputTensor, opts.asyncOp, bopts);
      },
      std::vector<at::Tensor>{outputTensor});
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::GatherOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  const int rootRank = checkedRootRank(opts.rootRank, getSize(), "Gather");
  if (getRank() == rootRank) {
    TORCH_CHECK(
        outputTensors.size() == 1,
        "Only single output tensor list supported on root rank, but got ",
        outputTensors.size());
  } else if (outputTensors.empty()) {
    // Normalize non-root c10d gather outputs to wrapper's empty list shape
    outputTensors = {};
    outputTensors.emplace_back();
  } else {
    TORCH_CHECK(
        outputTensors.size() == 1,
        "Only single output tensor list supported on non-root ranks, but got ",
        outputTensors.size());
  }
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  GatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return comm_->gather(
            outputTensors.at(0),
            inputTensors.at(0),
            rootRank,
            opts.asyncOp,
            bopts);
      },
      outputTensors.at(0));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::gather_into_tensor(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::GatherOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  const int rootRank =
      checkedRootRank(opts.rootRank, getSize(), "Gather-into-tensor");
  validateCoalescedTensors({inputTensor}, "Gather-into-tensor");
  const bool isRoot = getRank() == rootRank;
  if (isRoot) {
    validateCoalescedTensorPairs(
        {inputTensor}, {outputTensor}, getSize(), 1, "Gather-into-tensor");
  }
  GatherSingleOptions bopts;
  bopts.timeout =
      opts.timeout != kUnsetTimeout ? opts.timeout : options_->timeout;
  return launchAndWrapWork(
      [&]() {
        return comm_->gather_single(
            outputTensor, inputTensor, rootRank, opts.asyncOp, bopts);
      },
      isRoot ? std::vector<at::Tensor>{outputTensor}
             : std::vector<at::Tensor>{});
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  const int rootRank = checkedRootRank(opts.rootRank, getSize(), "Scatter");
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor supported, but got ",
      outputTensors.size());
  ScatterOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  if (getRank() == rootRank) {
    TORCH_CHECK(
        inputTensors.size() == 1,
        "Only single input tensor list supported on root rank, but got ",
        inputTensors.size());
  } else {
    TORCH_CHECK(
        inputTensors.empty(),
        "Scatter input tensors must be empty on non-root ranks");
  }
  const std::vector<at::Tensor> emptyInputs;
  const auto& inputs = getRank() == rootRank ? inputTensors.at(0) : emptyInputs;
  return launchAndWrapWork(
      [&]() {
        return comm_->scatter(
            outputTensors.at(0), inputs, rootRank, opts.asyncOp, bopts);
      },
      outputTensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor list supported, but got ",
      inputTensors.size());
  ReduceScatterOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return comm_->reduce_scatter(
            outputTensors.at(0),
            inputTensors.at(0),
            toReduceOp(opts.reduceOp),
            opts.asyncOp,
            bopts);
      },
      outputTensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  validateCoalescedTensorPairs(
      inputTensors,
      outputTensors,
      1,
      getSize(),
      "Coalesced reduce-scatter-tensor");
  ReduceScatterSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  const auto reduceOp = toReduceOp(opts.reduceOp);
  return launchAndWrapWork(
      [&]() {
        return launchWorks(inputTensors.size(), [&](size_t index) {
          return comm_->reduce_scatter_single(
              outputTensors[index],
              inputTensors[index],
              reduceOp,
              opts.asyncOp,
              bopts);
        });
      },
      outputTensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::ReduceScatterOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  ReduceScatterSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return comm_->reduce_scatter_single(
            outputTensor,
            inputTensor,
            toReduceOp(opts.reduceOp),
            opts.asyncOp,
            bopts);
      },
      std::vector<at::Tensor>{outputTensor});
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const c10d::AllToAllOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    AllToAllSingleOptions bopts;
    if (opts.timeout != kUnsetTimeout) {
      bopts.timeout = opts.timeout;
    } else {
      bopts.timeout = options_->timeout;
    }
    return launchAndWrapWork(
        [&]() {
          return comm_->all_to_all_single(
              outputTensor, inputTensor, opts.asyncOp, bopts);
        },
        std::vector<at::Tensor>{outputTensor});
  } else {
    AllToAllvSingleOptions bopts;
    if (opts.timeout != kUnsetTimeout) {
      bopts.timeout = opts.timeout;
    } else {
      bopts.timeout = options_->timeout;
    }
    return launchAndWrapWork(
        [&]() {
          return comm_->all_to_all_v_single(
              outputTensor,
              inputTensor,
              toVecUint64(outputSplitSizes),
              toVecUint64(inputSplitSizes),
              opts.asyncOp,
              bopts);
        },
        std::vector<at::Tensor>{outputTensor});
  }
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllToAllOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  AllToAllOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  return launchAndWrapWork(
      [&]() {
        return comm_->all_to_all(
            outputTensors, inputTensors, opts.asyncOp, bopts);
      },
      outputTensors);
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::barrier(
    const c10d::BarrierOptions& opts) {
  auto coalescingScope = coalescingOperationScope();
  BarrierOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  } else {
    bopts.timeout = options_->timeout;
  }
  // Mirror stock ProcessGroupNCCL: a synchronous barrier host-blocks the CPU
  // thread until the collective (and prior stream work) completes, so callers
  // relying on the barrier to flush async device work -- e.g. clearing IPC
  // buffers on the stream before the first all_reduce -- do not race it and
  // deadlock. The host block lives entirely at this c10d layer: WorkWrapper
  // calls work_->hostSynchronize() after wait(), so the native TorchComm
  // barrier and TorchWork::wait() keep uniform semantics with every other
  // collective. Async barriers keep the non-blocking, stream-ordered behavior
  // via the work.
  return launchAndWrapWork(
      [&]() { return comm_->barrier(opts.asyncOp, bopts); },
      std::vector<at::Tensor>{},
      /*hostBlocking=*/!opts.asyncOp);
}

void BackendWrapper::monitoredBarrier(
    const c10d::BarrierOptions& opts,
    bool waitAllRanks) {
  // Mirror c10d's Gloo protocol with synchronous P2P because UnboundBuffer
  // has no safe active poll for finite Work waits. Per-op transport timeouts
  // remain catchable, so rank 0 can keep probing after one rank times out.
  // c10d already restricts monitored_barrier to Gloo groups.
  const int rank = getRank();
  const int worldSize = getSize();

  const std::chrono::milliseconds timeout =
      (opts.timeout != kUnsetTimeout) ? opts.timeout : options_->timeout;

  // Phase-1 (worker -> rank 0) and phase-2 (rank 0 -> worker) tags, generated
  // per call. Identical on every rank because monitoredBarrier is collective
  // and all ranks advance this PG's counter in lockstep (the counter is a
  // per-BackendWrapper member, so concurrent barriers on other PGs cannot
  // desync it across ranks).
  const uint32_t tagBase = monitoredBarrierTagCounter_.fetch_add(2);
  // Mask into the non-negative int range: c10d/gloo tags are ints, and the
  // counter would otherwise wrap past INT_MAX in a long-lived process and
  // produce negative tags. The mask is deterministic, so every rank still
  // derives identical tags; fetch_add(2) keeps the two tags distinct
  // (even/odd) and the low-30-bit mask never merges them.
  constexpr uint32_t kTagMask = 0x3FFFFFFFu;
  const int tagToZero = static_cast<int>(tagBase & kTagMask);
  const int tagFromZero = static_cast<int>((tagBase + 1) & kTagMask);

  auto makeCommTensor = [&]() {
    auto t =
        at::empty({1}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    t.fill_(rank);
    return t;
  };

  // Workers report in to rank 0, then block until rank 0 acks. Only rank 0
  // enforces the timeout, so a dead/slow rank is named by rank 0 instead of
  // every worker timing out and hiding the culprit.
  if (rank != 0) {
    try {
      auto outTensor = makeCommTensor();
      SendOptions sopts;
      sopts.tag = tagToZero; // blocking: timeout stays kNoTimeout
      comm_->send(outTensor, 0, /*async_op=*/false, sopts);

      auto inTensor = makeCommTensor();
      RecvOptions ropts;
      ropts.tag = tagFromZero; // blocking
      comm_->recv(inTensor, 0, /*async_op=*/false, ropts);
    } catch (const std::exception& e) {
      TORCH_CHECK(
          false,
          "Rank ",
          rank,
          " successfully reached monitoredBarrier, but received errors while "
          "waiting for send/recv from rank 0. Please check rank 0 logs for the "
          "faulty rank.\n Original exception: \n",
          e.what());
    }
    return;
  }

  // Rank 0 is the coordinator.
  //
  // Fast-fail (waitAllRanks == false) matches native
  // ProcessGroupGloo::monitoredBarrier: on the first straggler rank 0 raises
  // immediately (below) and never reaches the ack loop, so workers that already
  // checked in stay blocked in their ack recv (kNoTimeout) until the process is
  // torn down. This is intentional -- a failed monitoredBarrier is not a clean
  // barrier exit. waitAllRanks == true instead probes every worker and reports
  // all stragglers before raising.
  const auto startTime = std::chrono::steady_clock::now();
  auto remainingTime = [&]() -> std::chrono::milliseconds {
    if (waitAllRanks) {
      // Give every worker the full timeout: spending it all on worker n must
      // not starve probing of workers n+1.. (see the native gloo impl).
      return timeout;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime);
    return timeout - elapsed;
  };

  auto joinInts = [](const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) {
        s += ", ";
      }
      s += std::to_string(v[i]);
    }
    return s;
  };

  std::vector<int> processedRanks;
  processedRanks.reserve(static_cast<size_t>(worldSize - 1));
  for (int srcRank = 1; srcRank < worldSize; ++srcRank) {
    const auto remaining = remainingTime();
    if (!waitAllRanks && remaining.count() <= 0) {
      TORCH_CHECK(
          false,
          "Rank 0 timed out in monitoredBarrier after ",
          timeout.count(),
          " ms. Successfully processed ranks: ",
          joinInts(processedRanks));
    }
    try {
      // Fresh buffer per rank: a timed-out recv leaves a pending registration
      // on the gloo transport, so reusing the tensor could race a late message
      // into memory in use by the next probe.
      auto inTensor = makeCommTensor();
      RecvOptions ropts;
      ropts.tag = tagToZero;
      ropts.timeout = (remaining.count() > 0) ? remaining : timeout;
      comm_->recv(inTensor, srcRank, /*async_op=*/false, ropts);
      processedRanks.push_back(srcRank);
    } catch (const std::exception& e) {
      if (!waitAllRanks) {
        TORCH_CHECK(
            false,
            "[Rank 0]: Rank ",
            srcRank,
            " failed to pass monitoredBarrier in ",
            timeout.count(),
            " ms\n Original exception: \n",
            e.what());
      }
      // waitAllRanks: keep going and collect every failure below.
    }
  }

  if (waitAllRanks &&
      processedRanks.size() != static_cast<size_t>(worldSize - 1)) {
    std::vector<int> failedRanks;
    for (int i = 1; i < worldSize; ++i) {
      if (std::find(processedRanks.begin(), processedRanks.end(), i) ==
          processedRanks.end()) {
        failedRanks.push_back(i);
      }
    }
    TORCH_CHECK(
        false,
        "[Rank 0]: Ranks ",
        joinInts(failedRanks),
        " failed to pass monitoredBarrier in ",
        timeout.count(),
        " ms");
  }

  // Every worker checked in: ack each so all ranks leave the barrier together
  // (a true barrier -- all exit or none do). Ack every remaining worker even if
  // one send throws: a worker that died between check-in and ack must not leave
  // the healthy workers blocked forever in their ack recv. Collect failures and
  // raise only after every other worker has been acked.
  std::vector<int> ackFailedRanks;
  for (int dstRank = 1; dstRank < worldSize; ++dstRank) {
    try {
      auto outTensor = makeCommTensor();
      SendOptions sopts;
      sopts.tag = tagFromZero;
      comm_->send(outTensor, dstRank, /*async_op=*/false, sopts);
    } catch (const std::exception&) {
      ackFailedRanks.push_back(dstRank);
    }
  }
  TORCH_CHECK(
      ackFailedRanks.empty(),
      "[Rank 0]: failed to ack ranks ",
      joinInts(ackFailedRanks),
      " in monitoredBarrier; these ranks may remain blocked");
}

c10::intrusive_ptr<c10d::Work>
BackendWrapper::send(std::vector<at::Tensor>& tensors, int dstRank, int tag) {
  auto coalescingScope = coalescingOperationScope();
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  if (coalescing_batch_.has_value()) {
    if (!coalesced_collective_wrappers_.empty()) {
      TORCH_CHECK(
          false,
          "A coalescing window cannot mix collective and point-to-point operations");
    }
    // NOTE: `tag` is intentionally not threaded through the coalesced path.
    // BatchSendRecv/P2POp carry no per-op tag, and only the Gloo backend
    // consumes SendOptions::tag; coalescing is used for NCCL-style grouped
    // P2P (batch_isend_irecv), which matches by order, not tag.
    coalescing_batch_->send(tensors.at(0), dstRank);
    // Per-op Work returned during coalescing is a no-op sentinel; the real
    // Work covering the whole batch is returned by endCoalescing(). c10d's
    // batch_isend_irecv discards these per-op returns.
    auto work = c10::make_intrusive<WorkWrapper>(
        c10::make_intrusive<TorchWorkCompleted>(), tensors);
    coalescingScope.dismiss();
    return work;
  }
  SendOptions opts;
  opts.timeout = options_->timeout;
  opts.tag = tag;
  return wrapWork(
      comm_->send(tensors.at(0), dstRank, /*async_op=*/true, opts), tensors);
}

c10::intrusive_ptr<c10d::Work>
BackendWrapper::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  auto coalescingScope = coalescingOperationScope();
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  if (coalescing_batch_.has_value()) {
    if (!coalesced_collective_wrappers_.empty()) {
      TORCH_CHECK(
          false,
          "A coalescing window cannot mix collective and point-to-point operations");
    }
    // See the note in send(): the coalesced path does not thread `tag`.
    coalescing_batch_->recv(tensors.at(0), srcRank);
    auto work = c10::make_intrusive<WorkWrapper>(
        c10::make_intrusive<TorchWorkCompleted>(), tensors);
    coalescingScope.dismiss();
    return work;
  }
  RecvOptions opts;
  opts.timeout = options_->timeout;
  opts.tag = tag;
  return wrapWork(
      comm_->recv(tensors.at(0), srcRank, /*async_op=*/true, opts), tensors);
}

void BackendWrapper::startCoalescing() {
  TORCH_CHECK(
      !coalescing_batch_.has_value(),
      "BackendWrapper::startCoalescing called while a batch is already active");
  TORCH_INTERNAL_ASSERT(coalesced_collective_wrappers_.empty());
  TORCH_INTERNAL_ASSERT(coalesced_collective_outputs_.empty());
  coalescing_batch_.emplace(comm_->batch_op_create());
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::endCoalescing() {
  TORCH_CHECK(
      coalescing_batch_.has_value(),
      "BackendWrapper::endCoalescing called without a matching startCoalescing");
  // Move all state out before issue or wrapping so failures cannot wedge the
  // next coalescing window.
  auto batch = std::move(*coalescing_batch_);
  auto collectiveWrappers = std::move(coalesced_collective_wrappers_);
  auto collectiveOutputs = std::move(coalesced_collective_outputs_);
  resetCoalescingState();
  TORCH_CHECK(
      batch.ops.empty() || collectiveWrappers.empty(),
      "A coalescing window cannot mix point-to-point and collective operations");
  if (!collectiveWrappers.empty()) {
    try {
      size_t workCount = 0;
      for (const auto& wrapper : collectiveWrappers) {
        workCount += wrapper->works_.size();
      }
      std::vector<c10::intrusive_ptr<TorchWork>> collectiveWorks;
      collectiveWorks.reserve(workCount);
      for (const auto& wrapper : collectiveWrappers) {
        collectiveWorks.insert(
            collectiveWorks.end(),
            wrapper->works_.begin(),
            wrapper->works_.end());
      }
      return c10::make_intrusive<WorkWrapper>(
          std::move(collectiveWorks), std::move(collectiveOutputs));
    } catch (...) {
      for (const auto& wrapper : collectiveWrappers) {
        try {
          wrapper->synchronize();
        } catch (...) {
        }
      }
      throw;
    }
  }
  if (batch.ops.empty()) {
    // Empty coalescing window — return a completed sentinel so callers can
    // .wait() without blocking.
    return wrapWork(c10::make_intrusive<TorchWorkCompleted>());
  }
  BatchP2POptions bopts;
  bopts.timeout = options_->timeout;
  return wrapWork(batch.issue(/*async_op=*/true, bopts));
}

std::shared_ptr<TorchComm> BackendWrapper::getComm() const {
  return comm_;
}

std::shared_ptr<c10::Allocator> BackendWrapper::getMemAllocator() {
  return comm_->getMemAllocator();
}

const std::string BackendWrapper::getBackendName() const {
  return comm_->getBackend();
}

std::string_view BackendWrapper::getBackendVersion() const {
  return comm_->getBackendVersion();
}

c10::intrusive_ptr<c10d::Backend::Options> BackendWrapper::getBackendOptions() {
  return c10::static_intrusive_pointer_cast<c10d::Backend::Options>(options_);
}

bool BackendWrapper::verifyWorkTimeoutForTest(
    const c10::intrusive_ptr<c10d::Work>& work,
    const std::chrono::milliseconds& timeout) {
  // The work must be a WorkWrapper that wraps a TorchWork
  auto workWrapper = c10::dynamic_intrusive_pointer_cast<WorkWrapper>(work);
  if (!workWrapper) {
    TORCH_CHECK(false, "Work is not a WorkWrapper");
  }

  // Get the timeout from the underlying TorchWork
  return workWrapper->works_.size() == 1 &&
      workWrapper->works_.front()->getTimeout() == timeout;
}

void BackendWrapper::setTimeout(std::chrono::milliseconds timeout) {
  options_->timeout = timeout;
}
void BackendWrapper::shutdown() {
  // Idempotent: destroy_process_group iterates all backends and calls
  // shutdown() on each, but multiple BackendWrappers can share the same
  // underlying TorchComm (mixed cpu:gloo,cuda:nccl PGs registered through
  // the backendType-to-wrapper dedup path). Finalize-on-already-finalized
  // throws "TorchCommNCCL already finalized" — log and continue so destroy
  // is always safe to call.
  if (comm_) {
    try {
      comm_->finalize();
    } catch (const std::exception& e) {
      TC_LOG(WARNING)
          << "BackendWrapper::shutdown: TorchComm::finalize() raised, "
          << "treating as no-op (likely already finalized): " << e.what();
    }
  }
}

void BackendWrapper::abort() {
  if (comm_) {
    try {
      comm_->abort();
    } catch (const std::exception& e) {
      TC_LOG(WARNING) << "BackendWrapper::abort: TorchComm::abort() raised, "
                      << "treating as no-op (likely already aborted): "
                      << e.what();
    }
  }
}

c10::intrusive_ptr<c10d::Backend> BackendWrapper::split(
    const c10::intrusive_ptr<c10d::Store>& /* store */,
    const std::vector<int>& ranks,
    const c10::intrusive_ptr<c10d::Backend::Options>& opts) {
  auto comm = getComm();
  CommOptions commOpts;
  auto backendOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  if (backendOpts) {
    commOpts.abort_process_on_timeout_or_error =
        backendOpts->abort_process_on_timeout_or_error;
    commOpts.timeout = backendOpts->timeout;
    commOpts.is_high_priority_stream = backendOpts->is_high_priority_stream;
    commOpts.store = backendOpts->store;
    commOpts.hints = backendOpts->hints;
  }
  auto new_comm = comm->split(ranks, opts->group_name, commOpts);
  if (new_comm == nullptr) {
    return nullptr;
  }
  return c10::make_intrusive<BackendWrapper>(new_comm);
}

} // namespace torch::comms
