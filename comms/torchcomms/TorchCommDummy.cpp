// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/torchcomms/TorchCommDummy.hpp>
#include <comms/torchcomms/TorchCommFactory.hpp>
#include <comms/torchcomms/TorchWork.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu

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
  c10::intrusive_ptr<TorchWork> put(
      const at::Tensor& data,
      int dstRank,
      size_t targetDisp,
      bool asyncOp) override {
    (void)data;
    (void)dstRank;
    (void)targetDisp;
    (void)asyncOp;
    return c10::make_intrusive<DummyTorchWork>();
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
  c10::intrusive_ptr<TorchWork> signal(
      size_t signalDisp,
      uint64_t signalVal,
      int dstRank,
      bool asyncOp) override {
    (void)signalDisp;
    (void)signalVal;
    (void)dstRank;
    (void)asyncOp;
    return c10::make_intrusive<DummyTorchWork>();
  }
  virtual c10::intrusive_ptr<TorchWork> waitSignal(
      size_t signalDisp,
      uint64_t cmpVal,
      SignalCmpOp cmpOp,
      bool asyncOp) override {
    (void)signalDisp;
    (void)cmpVal;
    (void)cmpOp;
    (void)asyncOp;
    return c10::make_intrusive<DummyTorchWork>();
  }
};

TorchCommDummy::TorchCommDummy()
    : initialized_(false), device_(at::kCPU), rank_(0), size_(1) {}

void TorchCommDummy::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  device_ = device;
  options_ = options;
  initialized_ = true;
  name_ = name;
}

void TorchCommDummy::finalize() {
  initialized_ = false;
}

int TorchCommDummy::getRank() const {
  return rank_;
}

int TorchCommDummy::getSize() const {
  return size_;
}

std::string_view TorchCommDummy::getCommName() const {
  return name_;
}

std::string_view TorchCommDummy::getBackendName() const {
  return kBackendName;
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::barrier(
    bool async_op,
    const BarrierOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

c10::intrusive_ptr<TorchWork> TorchCommDummy::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  return c10::make_intrusive<DummyTorchWork>();
}

std::shared_ptr<TorchCommWindow> TorchCommDummy::window_allocate(
    const size_t window_size,
    bool cpu_buf,
    const size_t signal_size) {
  auto win = std::make_shared<DummyTorchCommWindow>();
  win->allocate(window_size, cpu_buf, signal_size);
  return win;
}

std::shared_ptr<TorchCommBackend> TorchCommDummy::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  (void)ranks;
  (void)name;
  (void)options;
  return std::make_shared<TorchCommDummy>();
}

const CommOptions& TorchCommDummy::getOptions() const {
  return options_;
}

const at::Device& TorchCommDummy::getDevice() const {
  return device_;
}

namespace {
class DummyRegistration {
 public:
  DummyRegistration() {
    TorchCommFactory::get().register_backend(
        "dummy", []() { return std::make_shared<TorchCommDummy>(); });
  }
};

static DummyRegistration registration{};
} // namespace

} // namespace comms
} // namespace torch
