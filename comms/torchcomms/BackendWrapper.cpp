// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/BackendWrapper.hpp"
#include "comms/torchcomms/TorchComm.hpp"

namespace torch {
namespace comms {

namespace {

PreMulSumFactorT getPreMulSumFactor(const c10d::ReduceOp& op) {
  const auto* preMulSupplement =
      reinterpret_cast<c10d::NCCLPreMulSumSupplement*>(op.supplement_.get());
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
  // TODO: implement error states
  return work_->isCompleted();
}
std::exception_ptr WorkWrapper::exception() const {
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
  // TODO: this should only wait on stream
  return work_->wait();
}
std::vector<at::Tensor> WorkWrapper::result() {
  return {};
}

BackendWrapper::BackendWrapper(std::shared_ptr<TorchComm> comm)
    : Backend(comm->getRank(), comm->getSize()),
      backend_(comm->unsafeGetBackend()) {}

c10::intrusive_ptr<c10d::Work> BackendWrapper::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
  TORCH_INTERNAL_ASSERT(tensors.size() == 1, "Only single tensor supported");
  BroadcastOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(
      backend_->broadcast(tensors.at(0), opts.rootRank, opts.asyncOp, bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {
  TORCH_INTERNAL_ASSERT(tensors.size() == 1, "Only single tensor supported");
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
  TORCH_INTERNAL_ASSERT(tensors.size() == 1, "Only single tensor supported");
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
  TORCH_INTERNAL_ASSERT(tensors.size() == 1, "Only single tensor supported");
  ReduceOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->reduce(
      tensors.at(0),
      opts.rootRank,
      toReduceOp(opts.reduceOp),
      opts.asyncOp,
      bopts));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_INTERNAL_ASSERT(
      outputTensors.size() == 1, "Only single tensor supported");
  TORCH_INTERNAL_ASSERT(
      inputTensors.size() == 1, "Only single tensor supported");
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
  TORCH_INTERNAL_ASSERT(
      outputTensorLists.size() == 1, "Only single tensor supported");
  TORCH_INTERNAL_ASSERT(
      inputTensors.size() == 1, "Only single tensor supported");
  AllGatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->all_gather(
      outputTensorLists.at(0), inputTensors.at(0), opts.asyncOp, bopts));
}

// TODO: Need to implement the case when input/output tensors are larger than
// one. since this a coalesced version. We only support one input/output tensor
// for now.
c10::intrusive_ptr<c10d::Work> BackendWrapper::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_INTERNAL_ASSERT(
      output_tensors.size() == 1, "Only single tensor supported");
  TORCH_INTERNAL_ASSERT(
      inputTensors.size() == 1, "Only single tensor supported");
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
  TORCH_INTERNAL_ASSERT(
      outputTensors.size() == 1, "Only single tensor supported");
  TORCH_INTERNAL_ASSERT(
      inputTensors.size() == 1, "Only single tensor supported");
  GatherOptions bopts;
  if (opts.timeout != kUnsetTimeout) {
    bopts.timeout = opts.timeout;
  }
  return c10::make_intrusive<WorkWrapper>(backend_->gather(
      outputTensors.at(0), inputTensors.at(0), opts.rootRank, opts.asyncOp));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
  TORCH_INTERNAL_ASSERT(
      outputTensors.size() == 1, "Only single tensor supported");
  ScatterOptions bopts;
  bopts.timeout = opts.timeout;
  if (getRank() == opts.rootRank) {
    TORCH_INTERNAL_ASSERT(
        inputTensors.size() == 1, "Only single tensor supported");
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    inputTensors = {};
    inputTensors.emplace_back();
  }
  return c10::make_intrusive<WorkWrapper>(backend_->scatter(
      outputTensors.at(0), inputTensors.at(0), opts.rootRank, opts.asyncOp));
}

c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  TORCH_INTERNAL_ASSERT(
      outputTensors.size() == 1, "Only single tensor supported");
  TORCH_INTERNAL_ASSERT(
      inputTensors.size() == 1, "Only single tensor supported");
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

// TODO: Need to implement the case when input/output tensors are larger than
// one. since this a coalesced version. We only support one input/output tensor
// for now.
c10::intrusive_ptr<c10d::Work> BackendWrapper::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  TORCH_INTERNAL_ASSERT(
      outputTensors.size() == 1, "Only single tensor supported");
  TORCH_INTERNAL_ASSERT(
      inputTensors.size() == 1, "Only single tensor supported");
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
  TORCH_INTERNAL_ASSERT(
      outputTensors.size() == 1, "Only single tensor supported");
  TORCH_INTERNAL_ASSERT(
      inputTensors.size() == 1, "Only single tensor supported");
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
  TORCH_INTERNAL_ASSERT(tensors.size() == 1, "Only single tensor supported");
  return c10::make_intrusive<WorkWrapper>(
      backend_->send(tensors.at(0), dstRank, tag));
}

c10::intrusive_ptr<c10d::Work>
BackendWrapper::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  TORCH_INTERNAL_ASSERT(tensors.size() == 1, "Only single tensor supported");
  return c10::make_intrusive<WorkWrapper>(
      backend_->recv(tensors.at(0), srcRank, tag));
}

} // namespace comms
} // namespace torch
