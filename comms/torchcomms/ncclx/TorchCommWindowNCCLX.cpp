// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

namespace torch {
namespace comms {

TorchCommWindowNCCLX::TorchCommWindowNCCLX(
    ncclComm_t ncclComm,
    std::shared_ptr<TorchCommNCCLX> torchComm)
    : nccl_comm_(ncclComm), torch_comm_(std::move(torchComm)) {
  // make sure the torchComm & ncclComm are not null
  checkCommAndThrow();

  // TorchCommWindowNCCLX reuse ncclApi/cudaApi/device from TorchCommNCCLX
  nccl_api_ = torch_comm_->getNcclApi();
  cuda_api_ = torch_comm_->getCudaApi();
  comm_device_ = torch_comm_->getDevice();
}

TorchCommWindowNCCLX::~TorchCommWindowNCCLX() noexcept {
  // User is responsible for waiting on work objects returned by
  // put/signal/wait_signal before destroying the window. This matches PyTorch's
  // async collective design. No synchronization needed here - proper work
  // management ensures safety.

  // check the window is deregistered
  if (win_ != nullptr) {
    auto result = nccl_api_->commWindowDeregister(nccl_comm_, win_);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "NCCLX window deregister failed";
    }
    win_ = nullptr;
    win_size_ = 0;
    buf_tensor_.reset(); // Release the tensor reference
  }
}

void TorchCommWindowNCCLX::tensor_register(const at::Tensor& tensor) {
  checkCommAndThrow();
  checkDeviceAndThrow(tensor);

  if (!tensor.defined()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register]: a valid tensor is required for window register.");
  }

  if (win_ != nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register]: Double registration error: win_ != nullptr");
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register]: contiguous tensor is required for window register.");
  }

  // Set member variables before calling winRegisterBuf
  buf_dtype_ = tensor.scalar_type();
  win_size_ = tensor.numel() * tensor.element_size();

  // Cache the buffer shape to avoid repeated calls to tensor.sizes()
  auto buf_shape = tensor.sizes();
  buf_shape_.clear();
  buf_shape_.reserve(buf_shape.size());
  for (size_t i = 0; i < buf_shape.size(); ++i) {
    buf_shape_.push_back(buf_shape[i]);
  }

  CHECK_EQ(
      nccl_api_->commWindowRegister(
          tensor.data_ptr(), win_size_, nccl_comm_, &win_),
      ncclSuccess)
      << "[TorchCommWindowNCCLX]: NCCLX window registration failed.";

  // Store a copy of the tensor to ensure its storage remains valid
  // for the lifetime of this window (via reference counting)
  buf_tensor_ = tensor;
  buf_device_ = tensor.device();
}

void TorchCommWindowNCCLX::tensor_deregister() {
  checkCommAndThrow();

  // Barrier to ensure all ranks have finished using the window
  // before anyone deregisters (prevents use-after-free)
  torch_comm_->barrier(false);

  if (win_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Double deregistration error: win_ == nullptr");
  }
  CHECK_EQ(nccl_api_->commWindowDeregister(nccl_comm_, win_), ncclSuccess)
      << "NCCLX window deregister failed";
  win_ = nullptr;
  win_size_ = 0;
  buf_tensor_.reset(); // Release the tensor reference

  // Barrier to ensure all ranks completed deregistration before cleanup
  torch_comm_->barrier(false);
}

c10::intrusive_ptr<TorchWork> TorchCommWindowNCCLX::put(
    const at::Tensor& tensor,
    int dstRank,
    size_t targetOffsetNelems,
    bool asyncOp,
    const PutOptions& options) {
  checkCommAndThrow();
  checkWindowAndThrow();
  const auto req_size =
      (tensor.numel() + targetOffsetNelems) * tensor.element_size();

  checkRequestSizeAndThrow(req_size);

  checkDeviceAndThrow(tensor);
  auto stream = torch_comm_->getOperationStream(asyncOp);
  auto work = torch_comm_->createWork(stream, options.timeout, {tensor});
  work->recordStart("put");
  CHECK_EQ(
      nccl_api_->winPut(
          tensor.data_ptr(),
          tensor.numel(),
          torch_comm_->getNcclDataType(tensor),
          dstRank,
          targetOffsetNelems,
          win_,
          stream),
      ncclSuccess);
  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);

  return work;
}

at::Tensor TorchCommWindowNCCLX::map_remote_tensor(int rank) {
  checkCommAndThrow();
  checkWindowAndThrow();
  void* base_ptr = nullptr;
  CHECK_EQ(
      nccl_api_->winSharedQuery(rank, nccl_comm_, win_, &base_ptr),
      ncclSuccess);

  CHECK_NOTNULL(base_ptr);

  // Use target_device() to bypass ATen's device validation when wrapping
  // memory pointers. This enables creating tensors from pointers where the
  // memory device differs from the caller's device.
  //
  // Memory lifetime managed by Window (freed on tensor_deregister).
  // No manual cleanup needed.
  //
  // Reference: aten/src/ATen/ops/from_blob.h:53-57
  auto options = at::TensorOptions().dtype(buf_dtype_).device(buf_device_);
  auto t = at::for_blob(base_ptr, buf_shape_)
               .options(options)
               .target_device(buf_device_)
               .make_tensor();

  return t;
}

c10::intrusive_ptr<TorchWork> TorchCommWindowNCCLX::signal(
    int peerRank,
    bool asyncOp,
    const SignalOptions& options) {
  checkWindowAndThrow();
  auto stream = torch_comm_->getOperationStream(asyncOp);
  auto work = torch_comm_->createWork(stream, options.timeout);
  work->recordStart("signal");
  CHECK_EQ(nccl_api_->winSignal(peerRank, win_, stream), ncclSuccess);
  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommWindowNCCLX::wait_signal(
    int peerRank,
    bool asyncOp,
    const WaitSignalOptions& options) {
  checkWindowAndThrow();
  auto stream = torch_comm_->getOperationStream(asyncOp);

  auto work = torch_comm_->createWork(stream, options.timeout);
  work->recordStart("wait_signal");
  CHECK_EQ(nccl_api_->winWaitSignal(peerRank, win_, stream), ncclSuccess);
  work->recordEnd();
  torch_comm_->enqueueWork(work, stream);
  return work;
}

void TorchCommWindowNCCLX::checkRequestSizeAndThrow(size_t input_size) const {
  if (input_size > win_size_) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Requested size (" +
        std::to_string(input_size) + " bytes) exceeds the window size (" +
        std::to_string(win_size_) + " bytes)");
  }
}

void TorchCommWindowNCCLX::checkDeviceAndThrow(const at::Tensor& tensor) const {
  auto data_device_type = tensor.device().type();
  // if the torchComm device is on GPU, we need to make sure the tensor is on
  // the same device as the window
  if (comm_device_.type() == at::kCUDA && data_device_type == at::kCUDA) {
    auto data_device_idx = tensor.device().index();
    if (comm_device_.index() != data_device_idx) {
      throw std::runtime_error(
          "[TorchCommWindowNCCLX]: Device mismatch: torchcomm is on device idx: " +
          std::to_string(comm_device_.index()) +
          ", operated tensor on device idx: " +
          std::to_string(data_device_idx));
    }
  }
}

void TorchCommWindowNCCLX::checkCommAndThrow() const {
  if (nccl_comm_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: NCCL communicator is not initialized, nccl_comm_ == nullptr");
  }
  if (torch_comm_.get() == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Torch communicator is not initialized, torch_comm_ == nullptr");
  }
}

void TorchCommWindowNCCLX::checkWindowAndThrow() const {
  if (win_ == nullptr) {
    throw std::runtime_error("[TorchCommWindowNCCLX]: NCCLX window is null");
  }
}

std::shared_ptr<TorchCommWindowAttr> TorchCommWindowNCCLX::get_attr(
    int peerRank) {
  checkWindowAndThrow();
  NcclxWindowAttr nccl_attr_raw = nullptr;
  CHECK_EQ(
      nccl_api_->winGetAttributes(peerRank, win_, &nccl_attr_raw), ncclSuccess)
      << "NCCLX window get_attr failed";

  CHECK_NOTNULL(nccl_attr_raw);

  // Use unique_ptr for RAII - automatic cleanup on exception or return
  std::unique_ptr<std::remove_pointer<NcclxWindowAttr>::type> nccl_attr(
      nccl_attr_raw);

  // Convert from NCCL type to TorchComm type
  auto attr = std::make_shared<TorchCommWindowAttr>();
  switch (nccl_attr->accessType) {
    case ncclWinAccessUnified:
      attr->accessType = TorchCommlWinAccessType::WIN_ACCESS_TYPE_UNIFIED;
      break;
    case ncclWinAccessSeparate:
      attr->accessType = TorchCommlWinAccessType::WIN_ACCESS_TYPE_SEPARATE;
      break;
    default:
      throw std::runtime_error("Unsupported NCCL window access type");
  }
  return attr;
}

} // namespace comms
} // namespace torch
