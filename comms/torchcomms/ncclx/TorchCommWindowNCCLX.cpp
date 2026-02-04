// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

#include <fmt/core.h>

#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#endif

namespace torch::comms {

// =============================================================================
// Constructor / Destructor
// =============================================================================

template <typename Backend>
TorchCommWindowNCCLX<Backend>::TorchCommWindowNCCLX(
    ncclComm_t ncclComm,
    std::shared_ptr<TorchCommNCCLX> torchComm)
    : nccl_comm_(ncclComm), torch_comm_(std::move(torchComm)) {
  checkCommAndThrow();
  nccl_api_ = torch_comm_->getNcclApi();
  comm_device_ = torch_comm_->getDevice();
}

template <typename Backend>
TorchCommWindowNCCLX<Backend>::~TorchCommWindowNCCLX() noexcept {
#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
  // Cleanup registered local buffers
  for (auto& buf : registered_local_buffers_) {
    if (buf.backend_window != nullptr && local_comm_ != nullptr) {
      nccl_api_->commWindowDeregister(
          local_comm_, static_cast<NcclxWindow>(buf.backend_window));
    }
  }
  registered_local_buffers_.clear();

  // Destroy ncclDevComm using the dev_comm stored in the deleter
  if (device_window_) {
    auto& deleter = device_window_.get_deleter();
    if (deleter.nccl_comm != nullptr && deleter.nccl_api != nullptr) {
      auto nccl_result = deleter.nccl_api->devCommDestroy(
          deleter.nccl_comm, &deleter.dev_comm);
      if (nccl_result != ncclSuccess) {
        TC_LOG(ERROR) << "Failed to destroy NCCL device communicator: "
                      << deleter.nccl_api->getErrorString(nccl_result);
      }
    }
  }

  // device_window_ unique_ptr destructor calls cudaFree via custom deleter
  device_window_.reset();

  // Cleanup NCCL orig window
  if (nccl_orig_win_ != nullptr) {
    auto result = nccl_api_->commWindowDeregister(nccl_comm_, nccl_orig_win_);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "NCCLX orig window deregister failed in destructor";
    }
    nccl_orig_win_ = nullptr;
  }

  // Cleanup local communicator
  if (local_comm_ != nullptr) {
    auto result = nccl_api_->commDestroy(local_comm_);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "NCCLX local_comm destroy failed in destructor";
    }
    local_comm_ = nullptr;
    local_comm_initialized_ = false;
  }
#endif

  // Cleanup CTRAN window (host API)
  if (win_ != nullptr) {
    auto result = nccl_api_->commWindowDeregister(nccl_comm_, win_);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "NCCLX window deregister failed";
    }
    win_ = nullptr;
    win_size_ = 0;
    buf_tensor_.reset();
  }
}

// =============================================================================
// Window Registration
// =============================================================================

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::tensor_register(const at::Tensor& tensor) {
  checkCommAndThrow();

  if (!tensor.defined()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register]: a valid tensor is required.");
  }
  checkDeviceAndThrow(tensor);
  if (win_ != nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register]: Double registration error.");
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX][register]: contiguous tensor required.");
  }

  buf_dtype_ = tensor.scalar_type();
  win_size_ = tensor.numel() * tensor.element_size();

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

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
  // Initialize local communicator and NCCL orig window for device API
  // initLocalComm();
  initNcclOrigWindow(tensor.data_ptr(), win_size_);
#endif

  buf_tensor_ = tensor;
  buf_device_ = tensor.device();
}

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::tensor_deregister() {
  checkCommAndThrow();
  torch_comm_->barrier(false);

  if (win_ == nullptr) {
    throw std::runtime_error("[TorchCommWindowNCCLX]: Double deregistration.");
  }
  auto ctran_result = nccl_api_->commWindowDeregister(nccl_comm_, win_);
  if (ctran_result != ncclSuccess) {
    TC_LOG(ERROR) << "NCCLX CTRAN window deregister failed";
  }
  win_ = nullptr;
  win_size_ = 0;

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
  if (nccl_orig_win_ != nullptr) {
    auto result = nccl_api_->commWindowDeregister(nccl_comm_, nccl_orig_win_);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "NCCLX orig window deregister failed";
    }
    nccl_orig_win_ = nullptr;
  }
#endif

  buf_tensor_.reset();
  torch_comm_->barrier(false);
}

template <typename Backend>
std::shared_ptr<TorchCommWindow> TorchCommWindowNCCLX<Backend>::clone() {
  auto new_window =
      std::make_shared<TorchCommWindowNCCLX<Backend>>(nccl_comm_, torch_comm_);
  if (buf_tensor_.has_value()) {
    new_window->tensor_register(buf_tensor_->clone());
  }
  return new_window;
}

// =============================================================================
// Host-side RMA Operations
// =============================================================================

template <typename Backend>
c10::intrusive_ptr<TorchWork> TorchCommWindowNCCLX<Backend>::put(
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

template <typename Backend>
at::Tensor TorchCommWindowNCCLX<Backend>::map_remote_tensor(int rank) {
  checkCommAndThrow();
  checkWindowAndThrow();
  void* base_ptr = nullptr;
  CHECK_EQ(
      nccl_api_->winSharedQuery(rank, nccl_comm_, win_, &base_ptr),
      ncclSuccess);

  CHECK_NOTNULL(base_ptr);

  auto options = at::TensorOptions().dtype(buf_dtype_).device(buf_device_);
  auto t = at::for_blob(base_ptr, buf_shape_)
               .options(options)
               .target_device(buf_device_)
               .make_tensor();

  return t;
}

template <typename Backend>
c10::intrusive_ptr<TorchWork> TorchCommWindowNCCLX<Backend>::signal(
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

template <typename Backend>
c10::intrusive_ptr<TorchWork> TorchCommWindowNCCLX<Backend>::wait_signal(
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

template <typename Backend>
std::shared_ptr<TorchCommWindowAttr> TorchCommWindowNCCLX<Backend>::get_attr(
    int peerRank) {
  checkWindowAndThrow();
  NcclxWindowAttr nccl_attr_raw = nullptr;
  CHECK_EQ(
      nccl_api_->winGetAttributes(peerRank, win_, &nccl_attr_raw), ncclSuccess)
      << "NCCLX window get_attr failed";

  CHECK_NOTNULL(nccl_attr_raw);

  std::unique_ptr<std::remove_pointer<NcclxWindowAttr>::type> nccl_attr(
      nccl_attr_raw);

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

// =============================================================================
// Device API Support (requires NCCLX 2.28+)
// =============================================================================

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::initLocalComm() {
  if (local_comm_initialized_) {
    return;
  }

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.splitShare = 1;

  CHECK_EQ(
      nccl_api_->commSplit(nccl_comm_, 0, 0, &local_comm_, &config),
      ncclSuccess)
      << "[TorchCommWindowNCCLX]: Failed to create local communicator";

  local_comm_initialized_ = true;
}

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::initNcclOrigWindow(void* ptr, size_t size) {
  if (nccl_orig_win_ != nullptr) {
    return;
  }

  CHECK_EQ(
      nccl_api_->commWindowRegister(
          ptr, size, nccl_comm_, &nccl_orig_win_, NCCL_WIN_DEVICE_API),
      ncclSuccess)
      << "[TorchCommWindowNCCLX]: NCCL orig window registration failed";
}

template <typename Backend>
typename TorchCommWindowNCCLX<Backend>::DeviceRegisteredBuffer
TorchCommWindowNCCLX<Backend>::register_local_buffer(const at::Tensor& tensor) {
  checkCommAndThrow();

  if (!local_comm_initialized_) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Local comm not initialized. "
        "Call tensor_register first.");
  }

  if (!tensor.defined() || !tensor.is_contiguous()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Invalid tensor for local buffer registration");
  }

  checkDeviceAndThrow(tensor);

  DeviceRegisteredBuffer buf;
  buf.base_ptr = tensor.data_ptr();
  buf.size = tensor.numel() * tensor.element_size();

  NcclxWindow local_win = nullptr;
  CHECK_EQ(
      nccl_api_->commWindowRegister(
          buf.base_ptr, buf.size, local_comm_, &local_win),
      ncclSuccess)
      << "[TorchCommWindowNCCLX]: Local buffer registration failed";

  buf.backend_window = static_cast<void*>(local_win);
  registered_local_buffers_.push_back(buf);

  return buf;
}

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::deregister_local_buffer(
    DeviceRegisteredBuffer& buf) {
  if (buf.backend_window == nullptr) {
    return;
  }

  if (!local_comm_initialized_ || local_comm_ == nullptr) {
    TC_LOG(WARNING)
        << "Local comm not initialized, skipping local buffer deregister";
    return;
  }

  auto result = nccl_api_->commWindowDeregister(
      local_comm_, static_cast<NcclxWindow>(buf.backend_window));
  if (result != ncclSuccess) {
    TC_LOG(ERROR) << "Failed to deregister local buffer";
  }

  // Remove from tracking vector
  auto it = std::find_if(
      registered_local_buffers_.begin(),
      registered_local_buffers_.end(),
      [&buf](const DeviceRegisteredBuffer& b) {
        return b.backend_window == buf.backend_window;
      });
  if (it != registered_local_buffers_.end()) {
    registered_local_buffers_.erase(it);
  }

  buf.backend_window = nullptr;
  buf.base_ptr = nullptr;
  buf.size = 0;
}

template <typename Backend>
typename TorchCommWindowNCCLX<Backend>::DeviceWindow*
TorchCommWindowNCCLX<Backend>::get_device_window(
    int signal_count,
    int counter_count,
    int barrier_count) {
  checkCommAndThrow();

  if (nccl_orig_win_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: NCCL orig window not initialized. "
        "Call tensor_register first.");
  }

  // Return existing device window pointer if already created
  if (device_window_) {
    return device_window_.get();
  }

  int commRank = 0;
  int commSize = 0;
  CHECK_EQ(nccl_api_->commUserRank(nccl_comm_, &commRank), ncclSuccess);
  CHECK_EQ(nccl_api_->commCount(nccl_comm_, &commSize), ncclSuccess);

  if (signal_count < 0) {
    signal_count = commSize;
  }
  if (counter_count < 0) {
    counter_count = commSize;
  }

  torchcomms::device::DeviceBackendConfig config;
  config.signal_count = signal_count;
  config.counter_count = counter_count;
  config.barrier_count = barrier_count;
  config.comm_rank = commRank;
  config.comm_size = commSize;

  // Create device window - the custom deleter handles all cleanup
  device_window_ = Backend::create_device_window(
      nccl_comm_,
      nccl_api_,
      config,
      nccl_orig_win_,
      buf_tensor_.has_value() ? buf_tensor_->data_ptr() : nullptr,
      win_size_);

  return device_window_.get();
}

#endif // TORCHCOMMS_HAS_NCCL_DEVICE_API

// =============================================================================
// Validation Helpers
// =============================================================================

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::checkRequestSizeAndThrow(
    size_t input_size) const {
  if (input_size > win_size_) {
    throw std::runtime_error(
        fmt::format(
            "[TorchCommWindowNCCLX]: Requested size ({} bytes) exceeds the window size ({} bytes)",
            input_size,
            win_size_));
  }
}

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::checkDeviceAndThrow(
    const at::Tensor& tensor) const {
  auto data_device_type = tensor.device().type();
  if (comm_device_.type() == at::kCUDA && data_device_type == at::kCUDA) {
    auto data_device_idx = tensor.device().index();
    if (comm_device_.index() != data_device_idx) {
      throw std::runtime_error(
          fmt::format(
              "[TorchCommWindowNCCLX]: Device mismatch: torchcomm on device {}, tensor on device {}",
              comm_device_.index(),
              data_device_idx));
    }
  }
}

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::checkCommAndThrow() const {
  if (nccl_comm_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: NCCL communicator not initialized");
  }
  if (torch_comm_.get() == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Torch communicator not initialized");
  }
}

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::checkWindowAndThrow() const {
  if (win_ == nullptr) {
    throw std::runtime_error("[TorchCommWindowNCCLX]: NCCLX window is null");
  }
}

// =============================================================================
// Explicit Template Instantiation
// =============================================================================

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
template class TorchCommWindowNCCLX<torchcomms::device::NCCLGinBackend>;
#else
template class TorchCommWindowNCCLX<HostOnlyBackend>;
#endif

} // namespace torch::comms
