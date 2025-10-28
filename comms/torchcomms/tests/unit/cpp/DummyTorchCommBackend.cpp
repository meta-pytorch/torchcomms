// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/tests/unit/cpp/DummyTorchCommBackend.hpp"
#include <comms/torchcomms/TorchWork.hpp>

namespace torch {
namespace comms {

// Dummy TorchWork implementation for testing
class DummyTorchWork : public TorchWork {
 public:
  bool isCompleted() override {
    return true;
  }

  void wait() override {}
};

class DummyTorchCommWindow : public TorchCommWindow {
 public:
  void allocate(
      const size_t window_size,
      bool cpu_buf,
      const size_t signal_size = 256) override {
    (void)cpu_buf;
    (void)signal_size;
    win_size_ = window_size;
  }
  std::shared_ptr<TorchWork> put(
      const at::Tensor& data,
      int dstRank,
      size_t targetDisp,
      bool asyncOp) override {
    (void)data;
    (void)dstRank;
    (void)targetDisp;
    (void)asyncOp;
    return std::make_shared<DummyTorchWork>();
  }
  at::Tensor getTensor(
      int rank,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      int64_t storageOffset) override {
    (void)rank;
    (void)sizes;
    (void)dtype;
    (void)storageOffset;
    return at::Tensor();
  }
  std::shared_ptr<TorchWork> signal(
      size_t signalDisp,
      uint64_t signalVal,
      int dstRank,
      bool asyncOp) override {
    (void)signalDisp;
    (void)signalVal;
    (void)dstRank;
    (void)asyncOp;
    return std::make_shared<DummyTorchWork>();
  }
  virtual std::shared_ptr<TorchWork> waitSignal(
      size_t signalDisp,
      uint64_t cmpVal,
      SignalCmpOp cmpOp,
      bool asyncOp) override {
    (void)signalDisp;
    (void)cmpVal;
    (void)cmpOp;
    (void)asyncOp;
    return std::make_shared<DummyTorchWork>();
  }
};

DummyTorchCommBackend::DummyTorchCommBackend()
    : initialized_(false), device_(at::kCPU), rank_(0), size_(1) {}

void DummyTorchCommBackend::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  device_ = device;
  options_ = options;
  initialized_ = true;
  name_ = name;
}

void DummyTorchCommBackend::finalize() {
  initialized_ = false;
}

int DummyTorchCommBackend::getRank() const {
  return rank_;
}

int DummyTorchCommBackend::getSize() const {
  return size_;
}

std::string_view DummyTorchCommBackend::getCommName() const {
  return name_;
}

std::string_view DummyTorchCommBackend::getBackendName() const {
  return kBackendName;
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::all_reduce(
    at::Tensor& tensor,
    ReduceOp op,
    bool async_op,
    const AllReduceOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::reduce(
    const at::Tensor& tensor,
    int root,
    ReduceOp op,
    bool async_op,
    const ReduceOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    ReduceOp op,
    bool async_op,
    const ReduceScatterOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    ReduceOp op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::barrier(
    bool async_op,
    const BarrierOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchWork> DummyTorchCommBackend::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  return std::make_shared<DummyTorchWork>();
}

std::shared_ptr<TorchCommWindow> DummyTorchCommBackend::window_allocate(
    const size_t window_size,
    bool cpu_buf,
    const size_t signal_size) {
  auto win = std::make_shared<DummyTorchCommWindow>();
  win->allocate(window_size, cpu_buf, signal_size);
  return win;
}

std::shared_ptr<TorchCommBackend> DummyTorchCommBackend::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  (void)ranks;
  (void)name;
  (void)options;
  return std::make_shared<DummyTorchCommBackend>();
}

const CommOptions& DummyTorchCommBackend::getOptions() const {
  return options_;
}

const at::Device& DummyTorchCommBackend::getDevice() const {
  return device_;
}

} // namespace comms
} // namespace torch

static torch::comms::TorchCommBackend* new_comm_impl() {
  return new torch::comms::DummyTorchCommBackend();
}

static void destroy_comm_impl(torch::comms::TorchCommBackend* comm) {
  delete comm;
}

static const char* get_supported_version_impl() {
  return torch::comms::TORCHCOMM_BACKEND_ABI_VERSION;
}

extern "C" torch::comms::DynamicLoaderInterface create_dynamic_loader_dummy() {
  torch::comms::DynamicLoaderInterface interface{
      .new_comm = new_comm_impl,
      .destroy_comm = destroy_comm_impl,
      .get_supported_version = get_supported_version_impl,
  };
  return interface;
}
