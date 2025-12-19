// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommFactory.hpp"

namespace torch {
namespace comms {

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

int TorchComm::getRank() {
  return impl_->getRank();
}

int TorchComm::getSize() {
  return impl_->getSize();
}

std::string_view TorchComm::getCommName() const {
  return impl_->getCommName();
}

// Point-to-Point Operations
c10::intrusive_ptr<TorchWork> TorchComm::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  return impl_->send(tensor, dst, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  return impl_->recv(tensor, src, async_op, options);
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchComm::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  return impl_->broadcast(tensor, root, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  return impl_->all_reduce(tensor, op, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  return impl_->reduce(tensor, root, op, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  return impl_->all_gather(tensor_list, tensor, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  return impl_->all_gather_v(tensor_list, tensor, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  return impl_->all_gather_single(output, input, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  return impl_->reduce_scatter(output, input_list, op, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  return impl_->reduce_scatter_v(output, input_list, op, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  return impl_->reduce_scatter_single(output, input, op, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  return impl_->all_to_all_single(output, input, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  return impl_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  return impl_->all_to_all(
      output_tensor_list, input_tensor_list, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::barrier(
    bool async_op,
    const BarrierOptions& options) {
  return impl_->barrier(async_op, options);
}

// Scatter and Gather Operations
c10::intrusive_ptr<TorchWork> TorchComm::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  return impl_->scatter(
      output_tensor, input_tensor_list, root, async_op, options);
}

c10::intrusive_ptr<TorchWork> TorchComm::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  return impl_->gather(
      output_tensor_list, input_tensor, root, async_op, options);
}

std::shared_ptr<TorchCommWindow> TorchComm::new_window() {
  return impl_->new_window();
}

// Communicator Management
std::shared_ptr<TorchComm> TorchComm::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  auto new_impl = impl_->split(ranks, name, options);
  if (new_impl == nullptr) {
    return nullptr;
  }
  return std::shared_ptr<TorchComm>(
      new TorchComm(backend_, std::move(new_impl)));
}

const CommOptions& TorchComm::getOptions() const {
  return impl_->getOptions();
}

const at::Device& TorchComm::getDevice() const {
  return impl_->getDevice();
}

// Memory Management
std::shared_ptr<c10::Allocator> TorchComm::getMemAllocator() {
  return impl_->getMemAllocator();
}

// Batch Operations

BatchSendRecv::BatchSendRecv(TorchComm* parent) : parent_(parent) {}

BatchSendRecv::P2POp::P2POp(OpType type, const at::Tensor& tensor, int peer) {
  this->type = type;
  this->tensor = tensor;
  this->peer = peer;
}

BatchSendRecv TorchComm::batch_op_create() {
  return BatchSendRecv(this);
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

} // namespace comms
} // namespace torch
