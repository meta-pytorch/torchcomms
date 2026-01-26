// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/BackendWrapper.hpp"
#include "comms/torchcomms/TorchComm.hpp"

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
    vecUint64.push_back(i);
  }
  return vecUint64;
}

} // namespace

WorkWrapper::WorkWrapper(c10::intrusive_ptr<TorchWork> work)
    : work_(std::move(work)) {}

bool WorkWrapper::isCompleted() {
  return work_->isCompleted();
}
bool WorkWrapper::isSuccess() const {
  // Note: Error state tracking is not implemented. This method returns
  // isCompleted() as a simplification. The underlying TorchWork does not
  // expose separate success/error states, so we assume completion implies
  // success. Callers that need error detection should use try/catch around
  // wait() instead.
  return work_->isCompleted();
}
std::exception_ptr WorkWrapper::exception() const {
  // Note: Exception capture is not implemented. The underlying TorchWork
  // interface does not provide a mechanism to retrieve exceptions after
  // completion. Errors are raised during wait() calls instead.
  return nullptr;
}
bool WorkWrapper::wait(std::chrono::milliseconds timeout) {
  if (timeout != kNoTimeout) {
    throw std::runtime_error("wait timeout not supported");
  }
  work_->wait();
  return true;
}
void WorkWrapper::synchronize() {
  // Note: Ideally this should only synchronize on the CUDA stream without
  // blocking the CPU thread. However, the current TorchWork interface does
  // not expose stream-only synchronization, so we fall back to a full wait()
  // which blocks until the operation completes.
  return work_->wait();
}
std::vector<at::Tensor> WorkWrapper::result() {
  return {};
}

BackendWrapper::BackendWrapper(std::shared_ptr<TorchComm> comm)
    : Backend(comm->getRank(), comm->getSize()),
      comm_(comm),
      backend_(comm->unsafeGetBackend()),
      options_(c10::make_intrusive<Options>()) {}

c10::intrusive_ptr<c10d::Work> BackendWrapper::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  BroadcastOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->broadcast(
      tensors.at(0), static_cast<int>(opts.rootRank), opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  AllReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->all_reduce(
      tensors.at(0), toReduceOp(opts.reduceOp), opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceCoalescedOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  AllReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->all_reduce(
      tensors.at(0), toReduceOp(opts.reduceOp), opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce(
    std::vector<at::Tensor>& tensors,
    const c10d::ReduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  ReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->reduce(
      tensors.at(0),
      static_cast<int>(opts.rootRank),
      toReduceOp(opts.reduceOp),
      opts.asyncOp,
      bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor list supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  AllGatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->all_gather(
      outputTensors.at(0), inputTensors.at(0), opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      outputTensorLists.size() == 1,
      "Only single output tensor list supported, but got ",
      outputTensorLists.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  AllGatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->all_gather(
      outputTensorLists.at(0), inputTensors.at(0), opts.asyncOp, bopts));
}

// Note: Coalesced operations with multiple input/output tensors are not yet
// supported. Currently only single tensor is supported. When extending this,
// iterate over all tensors and coalesce them into a single backend call.
c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_CHECK(
      output_tensors.size() == 1,
      "Only single output tensor supported, but got ",
      output_tensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  AllGatherSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->all_gather_single(
      output_tensors.at(0), inputTensors.at(0), opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::AllgatherOptions& opts) {
  AllGatherSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->all_gather_single(
      outputTensor, inputTensor, opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::GatherOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor list supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  GatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->gather(
      outputTensors.at(0),
      inputTensors.at(0),
      static_cast<int>(opts.rootRank),
      opts.asyncOp));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor supported, but got ",
      outputTensors.size());
  ScatterOptions bopts;
  bopts.timeout = opts.timeout;
  if (getRank() == opts.rootRank) {
    TORCH_CHECK(
        inputTensors.size() == 1,
        "Only single input tensor list supported on root rank, but got ",
        inputTensors.size());
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    inputTensors = {};
    inputTensors.emplace_back();
  }
  return c10::make_intrusive<WorkWrapper>(backend_->scatter(
      outputTensors.at(0),
      inputTensors.at(0),
      static_cast<int>(opts.rootRank),
      opts.asyncOp));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
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
  }
  return c10::make_intrusive<WorkWrapper>(backend_->reduce_scatter(
      outputTensors.at(0),
      inputTensors.at(0),
      toReduceOp(opts.reduceOp),
      opts.asyncOp,
      bopts));
}

// Note: Coalesced operations with multiple input/output tensors are not yet
// supported. Currently only single tensor is supported. When extending this,
// iterate over all tensors and coalesce them into a single backend call.
c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  ReduceScatterSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->reduce_scatter_single(
      outputTensors.at(0),
      inputTensors.at(0),
      toReduceOp(opts.reduceOp),
      opts.asyncOp,
      bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::ReduceScatterOptions& opts) {
  ReduceScatterSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->reduce_scatter_single(
      outputTensor,
      inputTensor,
      toReduceOp(opts.reduceOp),
      opts.asyncOp,
      bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const c10d::AllToAllOptions& opts) {
  AllToAllvSingleOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->all_to_all_v_single(
      outputTensor,
      inputTensor,
      toVecUint64(outputSplitSizes),
      toVecUint64(inputSplitSizes),
      opts.asyncOp,
      bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllToAllOptions& opts) {
  TORCH_CHECK(
      outputTensors.size() == 1,
      "Only single output tensor supported, but got ",
      outputTensors.size());
  TORCH_CHECK(
      inputTensors.size() == 1,
      "Only single input tensor supported, but got ",
      inputTensors.size());
  AllToAllOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      backend_->all_to_all(outputTensors, inputTensors, opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::barrier(
    const c10d::BarrierOptions& opts) {
  BarrierOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      backend_->barrier(opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work>
BackendWrapper::send(std::vector<at::Tensor>& tensors, int dstRank, int tag) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  return c10::make_intrusive<WorkWrapper>(
      backend_->send(tensors.at(0), dstRank, tag));
}

c10::intrusive_ptr<c10d::Work>
BackendWrapper::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  TORCH_CHECK(
      tensors.size() == 1,
      "Only single tensor supported, but got ",
      tensors.size(),
      " tensors");
  return c10::make_intrusive<WorkWrapper>(
      backend_->recv(tensors.at(0), srcRank, tag));
}

std::shared_ptr<TorchComm> BackendWrapper::getComm() const {
  return comm_;
}

const std::string BackendWrapper::getBackendName() const {
  return comm_->getBackend();
}

c10::intrusive_ptr<c10d::Backend::Options> BackendWrapper::getBackendOptions() {
  return c10::static_intrusive_pointer_cast<c10d::Backend::Options>(options_);
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
    commOpts.high_priority_stream = backendOpts->high_priority_stream;
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
