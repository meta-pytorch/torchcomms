// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"
#include <fmt/core.h>
#include <algorithm>
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

namespace torch::comms {

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

  // Clean up device window if allocated
  if (device_window_ != nullptr) {
    // TODO: Free device memory for device_window_
    // cuda_api_->cudaFree(device_window_);
    device_window_ = nullptr;
  }

  // Deregister all local buffers
  for (auto& buf : registered_local_buffers_) {
    if (buf.backend_window != nullptr && local_comm_ != nullptr) {
      auto result = nccl_api_->commWindowDeregister(
          local_comm_, static_cast<NcclxWindow>(buf.backend_window));
      if (result != ncclSuccess) {
        TC_LOG(ERROR) << "Failed to deregister local buffer";
      }
    }
  }
  registered_local_buffers_.clear();

  // Deregister NCCL orig window for device API
  if (nccl_orig_win_ != nullptr) {
    auto result = nccl_api_->commWindowDeregister(nccl_comm_, nccl_orig_win_);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "NCCLX orig window deregister failed";
    }
    nccl_orig_win_ = nullptr;
  }

  // Destroy local communicator
  if (local_comm_ != nullptr) {
    auto result = nccl_api_->commDestroy(local_comm_);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "Failed to destroy local communicator";
    }
    local_comm_ = nullptr;
    local_comm_initialized_ = false;
  }

  // Deregister CTRAN window for host API
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

  // 1. Register CTRAN window for host API (put/signal/wait_signal)
  // This uses the default path based on NCCL_RMA_ALGO (typically CTRAN)
  CHECK_EQ(
      nccl_api_->commWindowRegister(
          tensor.data_ptr(), win_size_, nccl_comm_, &win_),
      ncclSuccess)
      << "[TorchCommWindowNCCLX]: NCCLX window registration failed.";

  // 2. Initialize local communicator for non-collective local buffer
  // registration This is COLLECTIVE - all ranks must call tensor_register
  // together
  initLocalComm();

  // 3. Register NCCL orig window for device API (GIN support)
  // This uses NCCL_WIN_FORCE_ORIG_PATH to bypass CTRAN and ensure GIN support
  initNcclOrigWindow(tensor.data_ptr(), win_size_);

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

std::shared_ptr<TorchCommWindow> TorchCommWindowNCCLX::clone() {
  auto new_window =
      std::make_shared<TorchCommWindowNCCLX>(nccl_comm_, torch_comm_);
  if (buf_tensor_.has_value()) {
    new_window->tensor_register(buf_tensor_->clone());
  }
  return new_window;
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
        fmt::format(
            "[TorchCommWindowNCCLX]: Requested size ({} bytes) exceeds the window size ({} bytes)",
            input_size,
            win_size_));
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
          fmt::format(
              "[TorchCommWindowNCCLX]: Device mismatch: torchcomm is on device idx: {}, operated tensor on device idx: {}",
              comm_device_.index(),
              data_device_idx));
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

void TorchCommWindowNCCLX::checkLocalCommAndThrow() const {
  if (local_comm_ == nullptr || !local_comm_initialized_) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Local communicator is not initialized. "
        "Call register_local_buffer() or get_device_window() first.");
  }
}

void TorchCommWindowNCCLX::checkDeviceWindowAndThrow() const {
  if (device_window_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Device window is not initialized. "
        "Call get_device_window() first.");
  }
}

void TorchCommWindowNCCLX::initLocalComm() {
  if (local_comm_initialized_) {
    return;
  }

  checkCommAndThrow();

  // Create a 1-rank local communicator via ncclCommSplit.
  // This is COLLECTIVE - all ranks must call this.
  // Each rank gets its own 1-rank communicator where:
  //   - color = myRank (each rank in its own group)
  //   - key = 0 (only one member per group)
  // The split comm SHARES ginState with parent (same ginComms[], ginCtx[]).
  int myRank = torch_comm_->getRank();

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  auto result =
      nccl_api_->commSplit(nccl_comm_, myRank, 0, &local_comm_, &config);

  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Failed to create local communicator via ncclCommSplit. "
        "Error: " +
        std::string(nccl_api_->getErrorString(result)));
  }

  local_comm_initialized_ = true;
  TC_LOG(INFO) << "Local communicator initialized for rank " << myRank;
}

void TorchCommWindowNCCLX::initNcclOrigWindow(void* ptr, size_t size) {
  checkCommAndThrow();

  // Register window via NCCL orig path for device API (GIN support).
  // This is COLLECTIVE - all ranks must call this.
  //
  // We use NCCL_WIN_FORCE_ORIG_PATH flag to explicitly force the NCCL orig path
  // regardless of the NCCL_RMA_ALGO environment variable. This ensures device
  // API (GIN) support is always available even when CTRAN is the default RMA
  // algorithm.

  auto result = nccl_api_->commWindowRegister(
      ptr, size, nccl_comm_, &nccl_orig_win_, NCCL_WIN_FORCE_ORIG_PATH);

  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Failed to register NCCL orig window for device API. "
        "Error: " +
        std::string(nccl_api_->getErrorString(result)));
  }

  TC_LOG(INFO) << "NCCL orig window registered for device API, size=" << size;
}

device::RegisteredBuffer TorchCommWindowNCCLX::register_local_buffer(
    const at::Tensor& tensor) {
  checkCommAndThrow();
  checkDeviceAndThrow(tensor);
  checkLocalCommAndThrow(); // Must have called tensor_register() first

  if (!tensor.defined()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register_local_buffer]: A valid tensor is required.");
  }

  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register_local_buffer]: Contiguous tensor is required.");
  }

  void* ptr = tensor.data_ptr();
  size_t size = tensor.numel() * tensor.element_size();

  // Register buffer with local_comm_ (NON-COLLECTIVE because local_comm_ has
  // nranks=1) All bootstrap barriers become no-ops when nranks == 1.
  NcclxWindow local_win = nullptr;
  auto result =
      nccl_api_->commWindowRegister(ptr, size, local_comm_, &local_win);

  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register_local_buffer]: Failed to register local buffer. "
        "Error: " +
        std::string(nccl_api_->getErrorString(result)));
  }

  device::RegisteredBuffer buf;
  buf.base_ptr = ptr;
  buf.size = size;
  buf.backend_window = static_cast<void*>(local_win);

  // Track registered buffers for cleanup
  registered_local_buffers_.push_back(buf);

  TC_LOG(INFO) << "Local buffer registered: ptr=" << ptr << ", size=" << size;

  return buf;
}

void TorchCommWindowNCCLX::deregister_local_buffer(
    device::RegisteredBuffer& buf) {
  checkLocalCommAndThrow();

  if (buf.backend_window == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][deregister_local_buffer]: Invalid buffer - backend_window is null.");
  }

  // Deregister from local_comm_ (NON-COLLECTIVE)
  auto result = nccl_api_->commWindowDeregister(
      local_comm_, static_cast<NcclxWindow>(buf.backend_window));

  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][deregister_local_buffer]: Failed to deregister local buffer. "
        "Error: " +
        std::string(nccl_api_->getErrorString(result)));
  }

  // Remove from tracked buffers
  auto it = std::find_if(
      registered_local_buffers_.begin(),
      registered_local_buffers_.end(),
      [&buf](const device::RegisteredBuffer& b) {
        return b.backend_window == buf.backend_window;
      });

  if (it != registered_local_buffers_.end()) {
    registered_local_buffers_.erase(it);
  }

  // Clear the buffer
  buf.base_ptr = nullptr;
  buf.size = 0;
  buf.backend_window = nullptr;

  TC_LOG(INFO) << "Local buffer deregistered";
}

device::TorchCommDeviceWindow* TorchCommWindowNCCLX::get_device_window(
    int signal_count,
    int counter_count,
    int barrier_count) {
  checkCommAndThrow();
  checkWindowAndThrow();
  checkLocalCommAndThrow(); // Must have called tensor_register() first

  // Return existing device window if already created
  if (device_window_ != nullptr) {
    return device_window_;
  }

  // Use default counts based on communicator size if not specified
  int comm_size = torch_comm_->getSize();
  if (signal_count < 0) {
    signal_count = comm_size;
  }
  if (counter_count < 0) {
    counter_count = comm_size;
  }

  // TODO: Implement device window allocation
  // This requires:
  // 1. Allocating device memory for TorchCommDeviceWindow structure
  // 2. Allocating device memory for signals, counters, barriers
  // 3. Setting up the device comm state with GIN handles
  // 4. Copying the structure to device memory
  //
  // For now, throw an error indicating this is not yet implemented.
  throw std::runtime_error(
      "[TorchCommWindowNCCLX][get_device_window]: Device window creation not yet implemented. ");

  // Placeholder for future implementation:
  // device_window_ = allocateDeviceWindow(signal_count, counter_count,
  // barrier_count); return device_window_;
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
      attr->accessType = TorchCommWinAccessType::WIN_ACCESS_TYPE_UNIFIED;
      break;
    case ncclWinAccessSeparate:
      attr->accessType = TorchCommWinAccessType::WIN_ACCESS_TYPE_SEPARATE;
      break;
    default:
      throw std::runtime_error("Unsupported NCCL window access type");
  }
  return attr;
}

} // namespace torch::comms
