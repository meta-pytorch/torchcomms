// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommFactory.hpp"

namespace torch::comms {

TorchComm::TorchComm(
    const std::string& backend_name,
    std::shared_ptr<TorchCommBackend> impl)
    : backend_(backend_name), impl_(std::move(impl)) {}

std::shared_ptr<TorchComm> new_comm(
    const std::string& backend_name,
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  auto backend_impl = TorchCommFactory::get().create_backend(
      backend_name, device, name, options);
  return std::shared_ptr<TorchComm>(
      new TorchComm(backend_name, std::move(backend_impl)));
}

void TorchComm::finalize() {
  impl_->finalize();
}

int TorchComm::getRank() const {
  return impl_->getRank();
}

int TorchComm::getSize() const {
  return impl_->getSize();
}

std::string_view TorchComm::getCommName() const {
  return impl_->getCommName();
}

void TorchComm::validateRank(int rank, const char* param_name) const {
  TORCH_CHECK(
      rank >= 0 && rank < getSize(),
      param_name,
      " must be in range [0, ",
      getSize(),
      "), but got ",
      rank);
}

// Point-to-Point Operations
c10::intrusive_ptr<TorchWork> TorchComm::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  validateRank(dst, "dst");

  preHook(
      PreHookArgs{
          .name = OpName::send,
          .async_op = async_op,
          .input_tensor = &tensor,
          .root = dst,
      });

  auto work = impl_->send(tensor, dst, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::send,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  validateRank(src, "src");

  preHook(
      PreHookArgs{
          .name = OpName::recv,
          .async_op = async_op,
          .output_tensor = &tensor,
          .root = src,
      });

  auto work = impl_->recv(tensor, src, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::recv,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchComm::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  validateRank(root, "root");

  preHook(
      PreHookArgs{
          .name = OpName::broadcast,
          .async_op = async_op,
          .input_tensor = &tensor,
          .root = root,
      });

  auto work = impl_->broadcast(tensor, root, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::broadcast,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::all_reduce,
          .async_op = async_op,
          .input_tensor = &tensor,
      });

  auto work = impl_->all_reduce(tensor, op, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::all_reduce,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  validateRank(root, "root");

  preHook(
      PreHookArgs{
          .name = OpName::reduce,
          .async_op = async_op,
          .input_tensor = &tensor,
          .root = root,
      });

  auto work = impl_->reduce(tensor, root, op, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::reduce,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::all_gather,
          .async_op = async_op,
          .input_tensor = &tensor,
      });

  auto work = impl_->all_gather(tensor_list, tensor, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::all_gather,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::all_gather_v,
          .async_op = async_op,
          .input_tensor = &tensor,
      });

  auto work = impl_->all_gather_v(tensor_list, tensor, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::all_gather_v,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::all_gather_single,
          .async_op = async_op,
          .input_tensor = &input,
          .output_tensor = &output,
      });

  auto work = impl_->all_gather_single(output, input, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::all_gather_single,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::reduce_scatter,
          .async_op = async_op,
          .output_tensor = &output,
      });

  auto work = impl_->reduce_scatter(output, input_list, op, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::reduce_scatter,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::reduce_scatter_v,
          .async_op = async_op,
          .output_tensor = &output,
      });

  auto work =
      impl_->reduce_scatter_v(output, input_list, op, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::reduce_scatter_v,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::reduce_scatter_single,
          .async_op = async_op,
          .input_tensor = &input,
          .output_tensor = &output,
      });

  auto work =
      impl_->reduce_scatter_single(output, input, op, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::reduce_scatter_single,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::all_to_all_single,
          .async_op = async_op,
          .input_tensor = &input,
          .output_tensor = &output,
      });

  auto work = impl_->all_to_all_single(output, input, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::all_to_all_single,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::all_to_all_v_single,
          .async_op = async_op,
          .input_tensor = &input,
          .output_tensor = &output,
          .output_split_sizes = &output_split_sizes,
          .input_split_sizes = &input_split_sizes,
      });

  auto work = impl_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::all_to_all_v_single,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::all_to_all,
          .async_op = async_op,
      });

  auto work = impl_->all_to_all(
      output_tensor_list, input_tensor_list, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::all_to_all,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::barrier(
    bool async_op,
    const BarrierOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::barrier,
          .async_op = async_op,
      });

  auto work = impl_->barrier(async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::barrier,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

// Scatter and Gather Operations
c10::intrusive_ptr<TorchWork> TorchComm::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  validateRank(root, "root");

  preHook(
      PreHookArgs{
          .name = OpName::scatter,
          .async_op = async_op,
          .output_tensor = &output_tensor,
          .root = root,
      });

  auto work =
      impl_->scatter(output_tensor, input_tensor_list, root, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::scatter,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

c10::intrusive_ptr<TorchWork> TorchComm::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  validateRank(root, "root");

  preHook(
      PreHookArgs{
          .name = OpName::gather,
          .async_op = async_op,
          .input_tensor = &input_tensor,
          .root = root,
      });

  auto work =
      impl_->gather(output_tensor_list, input_tensor, root, async_op, options);

  postHook(
      PostHookArgs{
          .name = OpName::gather,
          .work = c10::weak_intrusive_ptr<TorchWork>(work),
      });

  return work;
}

std::shared_ptr<TorchCommWindow> TorchComm::new_window(
    const std::optional<at::Tensor>& tensor) {
  preHook(
      PreHookArgs{
          .name = OpName::new_window,
      });
  auto window = impl_->new_window(tensor);
  postHook(
      PostHookArgs{
          .name = OpName::new_window,
          .new_window = std::weak_ptr<TorchCommWindow>(window),
      });
  return window;
}

// Communicator Management
std::shared_ptr<TorchComm> TorchComm::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  preHook(
      PreHookArgs{
          .name = OpName::split,
          .ranks = &ranks,
          .split_name = &name,
      });
  auto new_impl = impl_->split(ranks, name, options);
  if (new_impl == nullptr) {
    return nullptr;
  }
  auto comm =
      std::shared_ptr<TorchComm>(new TorchComm(backend_, std::move(new_impl)));
  postHook(
      PostHookArgs{
          .name = OpName::split,
          .new_comm = std::weak_ptr<TorchComm>(comm),
      });
  return comm;
}

const CommOptions& TorchComm::getOptions() const {
  return impl_->getOptions();
}

const at::Device& TorchComm::getDevice() const {
  return impl_->getDevice();
}

// Batch Operations

BatchSendRecv::BatchSendRecv(std::shared_ptr<TorchComm> parent)
    : parent_(std::move(parent)) {}

BatchSendRecv::P2POp::P2POp(OpType type, const at::Tensor& tensor, int peer) {
  this->type = type;
  this->tensor = tensor;
  this->peer = peer;
}

BatchSendRecv TorchComm::batch_op_create() {
  return BatchSendRecv(shared_from_this());
}

void BatchSendRecv::send(const at::Tensor& tensor, int dst) {
  auto op = P2POp(P2POp::OpType::SEND, tensor, dst);
  ops.push_back(op);
}

void BatchSendRecv::recv(at::Tensor& tensor, int src) {
  auto op = P2POp(P2POp::OpType::RECV, tensor, src);
  ops.push_back(op);
}

c10::intrusive_ptr<TorchWork> BatchSendRecv::issue(
    bool async_op,
    const BatchP2POptions& options) {
  return parent_->getBackendImpl()->batch_op_issue(ops, async_op, options);
}

// Global memory allocator function implementation
std::shared_ptr<c10::Allocator> get_mem_allocator(const std::string& backend) {
  return TorchCommFactory::get().get_allocator(backend);
}

RemovableHandle TorchComm::registerPreHook(TorchComm::PreHook preHook) {
  auto hookId = nextHookId_++;
  preHooks_.emplace(hookId, std::move(preHook));
  return RemovableHandle([self = weak_from_this(), hookId]() {
    if (auto selfPtr = self.lock()) {
      selfPtr->preHooks_.erase(hookId);
    }
  });
}

RemovableHandle TorchComm::registerPostHook(TorchComm::PostHook postHook) {
  auto hookId = nextHookId_++;
  postHooks_.emplace(hookId, std::move(postHook));
  return RemovableHandle([self = weak_from_this(), hookId]() {
    if (auto selfPtr = self.lock()) {
      selfPtr->postHooks_.erase(hookId);
    }
  });
}

void TorchComm::preHook(PreHookArgs&& args) {
  for (auto& hook : preHooks_) {
    hook.second(args);
  }
}

void TorchComm::postHook(PostHookArgs&& args) {
  if (!args.work) {
    return;
  }
  if (auto work = args.work->lock()) {
    work->setCallback([self = weak_from_this(), args = std::move(args)]() {
      if (auto selfPtr = self.lock()) {
        for (auto& hook : selfPtr->postHooks_) {
          hook.second(args);
        }
      }
    });
  }
}

} // namespace torch::comms
