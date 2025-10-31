// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

namespace torch {
namespace comms {

TorchCommWindowNCCLX::TorchCommWindowNCCLX(
    ncclComm_t ncclComm,
    std::shared_ptr<TorchCommNCCLX> torchComm,
    at::Device device)
    : nccl_comm_(ncclComm), torch_comm_(torchComm) {
  // make sure the torchComm & ncclComm are not null
  checkCommAndThrow();

  // TorchCommWindowNCCLX would ncclApi/CudaApi from TorchCommNCCLX
  nccl_api_ = torch_comm_->getNcclApi();
  cuda_api_ = torch_comm_->getCudaApi();

  // Retrieve the torchComm device, which may differ from the window device.
  // For example, a rank may allocate a window on the CPU even if its primary
  // device is GPU. When allocating a GPU window, ensure that the device matches
  // comm_device_ to maintain consistency during window_allocate()
  device_ = torch_comm_->getDevice();

  // Set CUDA device and verify it's accessible
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->setDevice(device_.index()),
      "Failed to set CUDA device to " + std::to_string(device_.index()));

  // Initialize default CUDA API implementation
  cuda_api_->eventCreate(&rma_event_);
  const char* uniqueid_force_high_env = std::getenv("TORCH_NCCL_HIGH_PRIORITY");
  bool force_high = uniqueid_force_high_env != nullptr &&
      std::string(uniqueid_force_high_env) == "1";
  cuda_api_->streamCreateWithPriority(&op_stream_, 0, force_high);
  cuda_api_->streamCreateWithPriority(&wait_stream_, 0, force_high);
  CHECK_NOTNULL(op_stream_);
  CHECK_NOTNULL(wait_stream_);
}

TorchCommWindowNCCLX::~TorchCommWindowNCCLX() noexcept {
  // free the window if it is not freed
  // Ensure all pending RMA operations have finished
  cuda_api_->streamSynchronize(op_stream_);
  cuda_api_->streamSynchronize(wait_stream_);
  win_size_ = 0;
  CHECK_EQ(nccl_api_->winFree(nccl_comm_, win_), ncclSuccess)
      << "NCCLX window free failed";
  win_ = nullptr;

  // Destroy cuda_api resources
  cuda_api_->eventDestroy(rma_event_);
  cuda_api_->streamDestroy(op_stream_);
  cuda_api_->streamDestroy(wait_stream_);
}

void TorchCommWindowNCCLX::allocate(
    const size_t window_size,
    bool cpu_buf,
    const size_t signal_size) {
  checkCommAndThrow();
  void* base_ptr;
  cpuBuf_ = cpu_buf;
  signal_size_ = signal_size;
  // TODO: add optimization to allocate window with size 0
  win_size_ = window_size;
  if (win_ != nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][Allocate]: Double allocation error: win_ != nullptr");
  }
  CHECK_EQ(
      nccl_api_->winAllocate(
          win_size_, nccl_comm_, &base_ptr, &win_, cpuBuf_, signal_size),
      ncclSuccess)
      << "[TorchCommWindowNCCLX]: NCCLX window allocation failed.";
}

std::shared_ptr<TorchWork> TorchCommWindowNCCLX::put(
    const at::Tensor& data,
    int dstRank,
    size_t targetDisp,
    bool asyncOp) {
  checkCommAndThrow();
  const auto req_size = (data.numel() + targetDisp) * data.element_size();

  checkRequestSizeAndThrow(req_size);
  CHECK_NOTNULL(win_);

  checkDeviceAndThrow(data);
  auto stream =
      asyncOp ? op_stream_ : cuda_api_->getCurrentCUDAStream(device_.index());
  if (asyncOp) {
    checkOpStreamAndThrow();
    cuda_api_->eventRecord(
        rma_event_, cuda_api_->getCurrentCUDAStream(device_.index()));
    cuda_api_->streamWaitEvent(stream, rma_event_, 0);
  }
  auto work = torch_comm_->createWork(stream, kDefaultTimeout, {data});
  work->recordStart("put");
  CHECK_EQ(
      nccl_api_->winPut(
          reinterpret_cast<void*>(data.data_ptr()),
          data.numel(),
          torch_comm_->getNcclDataType(data),
          dstRank,
          targetDisp,
          win_,
          stream),
      ncclSuccess);
  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);

  return work;
}

at::Tensor TorchCommWindowNCCLX::getTensor(
    int rank,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    int64_t storageOffset) {
  checkCommAndThrow();
  checkWindowAndThrow();
  void* base_ptr = nullptr;
  CHECK_EQ(
      nccl_api_->winSharedQuery(rank, nccl_comm_, win_, &base_ptr),
      ncclSuccess);

  CHECK_NOTNULL(base_ptr);

  const size_t numel = std::accumulate(
      sizes.begin(),
      sizes.end(),
      static_cast<size_t>(1),
      std::multiplies<size_t>());
  const auto element_size = c10::elementSize(dtype);
  const auto req_size = (numel + storageOffset) * element_size;
  checkRequestSizeAndThrow(req_size);
  auto data_ptr =
      reinterpret_cast<uint8_t*>(base_ptr) + storageOffset * element_size;
  auto options = cpuBuf_ ? at::TensorOptions().dtype(dtype).device(at::kCPU)
                         : at::TensorOptions().dtype(dtype).device(device_);
  auto t = at::for_blob(data_ptr, sizes)
               .options(options)
               .target_device(cpuBuf_ ? at::kCPU : device_)
               .make_tensor();
  return t;
}

std::shared_ptr<TorchWork> TorchCommWindowNCCLX::signal(
    size_t signalDisp,
    uint64_t signalVal,
    int dstRank,
    bool asyncOp) {
  checkWindowAndThrow();
  // make sure the signalDisp is within the signal_size_
  CHECK_LT(signalDisp, signal_size_) << "signalDisp is out of range";
  auto stream =
      asyncOp ? op_stream_ : cuda_api_->getCurrentCUDAStream(device_.index());
  if (asyncOp) {
    checkOpStreamAndThrow();
    cuda_api_->eventRecord(
        rma_event_, cuda_api_->getCurrentCUDAStream(device_.index()));
    cuda_api_->streamWaitEvent(stream, rma_event_, 0);
  }
  auto work = torch_comm_->createWork(stream, kDefaultTimeout);
  work->recordStart("signal");
  CHECK_EQ(
      nccl_api_->winSignal(signalDisp, signalVal, dstRank, win_, stream),
      ncclSuccess);
  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);
  return work;
}

std::shared_ptr<TorchWork> TorchCommWindowNCCLX::waitSignal(
    size_t signalDisp,
    uint64_t cmpVal,
    SignalCmpOp cmpOp,
    bool asyncOp) {
  checkWindowAndThrow();
  auto nccl_cmp_op = torch_comm_->getNcclSignalCmpOp(cmpOp);
  auto stream =
      asyncOp ? wait_stream_ : cuda_api_->getCurrentCUDAStream(device_.index());
  if (asyncOp) {
    checkWaitStreamAndThrow();
    cuda_api_->eventRecord(
        rma_event_, cuda_api_->getCurrentCUDAStream(device_.index()));
    cuda_api_->streamWaitEvent(stream, rma_event_, 0);
  }
  auto work = torch_comm_->createWork(stream, kDefaultTimeout);
  work->recordStart("waitSignal");
  CHECK_EQ(
      nccl_api_->winWaitSignal(signalDisp, cmpVal, nccl_cmp_op, win_, stream),
      ncclSuccess);
  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);
  return work;
}

void TorchCommWindowNCCLX::checkEventAndThrow() {
  if (rma_event_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: RMA event is not initialized, rma_event_ == nullptr");
  }
}
void TorchCommWindowNCCLX::checkOpStreamAndThrow() {
  if (op_stream_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: RMA op stream is not initialized, op_stream_ == nullptr");
  }
}

void TorchCommWindowNCCLX::checkWaitStreamAndThrow() {
  if (wait_stream_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: RMA wait stream is not initialized, wait_stream_ == nullptr");
  }
}

void TorchCommWindowNCCLX::checkRequestSizeAndThrow(size_t input_size) {
  if (input_size > win_size_) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Requested size (" +
        std::to_string(input_size) + " bytes) exceeds the window size (" +
        std::to_string(win_size_) + " bytes)");
  }
  return;
}

void TorchCommWindowNCCLX::checkDeviceAndThrow(const at::Tensor& tensor) {
  auto data_device_type = tensor.device().type();
  // if tenosr and window buffer are both on GPU, we need to check whether they
  // are on the same device, otherwise throw error
  if (!cpuBuf_ && data_device_type == at::kCUDA) {
    auto data_device_idx = tensor.device().index();
    if (device_.index() != tensor.device().index()) {
      throw std::runtime_error(
          "[TorchCommWindowNCCLX]: Device mismatch: window on device idx: " +
          std::to_string(device_.index()) +
          ", operated tensor on device idx: " +
          std::to_string(data_device_idx));
    }
  }
}

void TorchCommWindowNCCLX::checkCommAndThrow() {
  if (nccl_comm_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: NCCL communicator is not initialized, nccl_comm_ == nullptr");
  }
  if (torch_comm_.get() == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Torch communicator is not initialized, torch_comm_ == nullptr");
  }
}

void TorchCommWindowNCCLX::checkWindowAndThrow() {
  if (win_ == nullptr) {
    throw std::runtime_error("[TorchCommWindowNCCLX]: NCCLX window is null");
  }
}

} // namespace comms
} // namespace torch
