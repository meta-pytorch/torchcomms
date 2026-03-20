// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

#include <fmt/core.h>

#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#endif

#if defined(ENABLE_PIPES)
#include "comms/torchcomms/device/pipes/PipesDeviceBackend.hpp"
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
  // Cleanup registered local buffers via backend-specific deregistration
  for (auto& buf : registered_local_buffers_) {
    if (nccl_comm_ != nullptr) {
      Backend::deregister_local_buffer(nccl_api_, nccl_comm_, buf);
    }
  }
  registered_local_buffers_.clear();

  // Destroy backend-specific device communicator state.
  // GIN: devCommDestroy. Pipes: no-op (cleanup via deleter).
  Backend::destroy_device_comm(device_window_);

  // device_window_ unique_ptr destructor calls cudaFree via custom deleter
  device_window_.reset();

  // Deregister extra window (GIN nccl_orig_win_; Pipes: no-op).
  Backend::deregister_extra_window(nccl_api_, nccl_comm_, &nccl_orig_win_);

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
  // GIN: register a second window with NCCL_WIN_DEVICE_API flag.
  // Pipes: no-op (device window creation deferred to get_device_window).
  Backend::register_extra_window(
      nccl_api_, nccl_comm_, &nccl_orig_win_, tensor.data_ptr(), win_size_);
#endif

  // In graph capture mode, we create a non-owned buffer: the window does not
  // hold a reference to the tensor. This relies on the caller keeping the
  // tensor alive for the lifetime of the window. The NCCL window registration
  // (commWindowRegister) independently tracks the underlying physical buffer,
  // so the window remains functional without buf_tensor_.
  //
  // IMPORTANT: In graph capture mode, the window must not outlive the graph
  // or the tensor that was registered. The user is responsible for ensuring
  // the tensor's storage remains valid for the window's entire lifetime.
  if (torch_comm_->getGraphCaptureMode()) {
    TC_LOG(WARNING)
        << "[TorchCommWindowNCCLX]: Graph capture mode active — window holds "
        << "a non-owned buffer. The registered tensor must remain alive for "
        << "the lifetime of this window. get_tensor() will return nullopt.";
  } else {
    buf_tensor_ = tensor;
  }
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
  // GIN: deregister nccl_orig_win_. Pipes: no-op.
  Backend::deregister_extra_window(nccl_api_, nccl_comm_, &nccl_orig_win_);
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
  torch_comm_->graph_event_tracker_.initOnGraphStart(stream);
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
  torch_comm_->graph_event_tracker_.initOnGraphStart(stream);
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

  torch_comm_->graph_event_tracker_.initOnGraphStart(stream);
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
typename TorchCommWindowNCCLX<Backend>::DeviceRegisteredBuffer
TorchCommWindowNCCLX<Backend>::register_local_buffer(const at::Tensor& tensor) {
  checkCommAndThrow();

  if (device_window_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Device window not initialized. "
        "Call get_device_window() first before registering local buffers.");
  }

  if (!tensor.defined() || !tensor.is_contiguous()) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Invalid tensor for local buffer registration");
  }

  checkDeviceAndThrow(tensor);

  auto buf = Backend::register_local_buffer(
      nccl_api_,
      nccl_comm_,
      tensor.data_ptr(),
      tensor.numel() * tensor.element_size());

  registered_local_buffers_.push_back(buf);
  return buf;
}

template <typename Backend>
void TorchCommWindowNCCLX<Backend>::deregister_local_buffer(
    DeviceRegisteredBuffer& buf) {
  if (buf.base_ptr == nullptr && buf.backend_window == nullptr) {
    return;
  }

  // Remove from tracking vector before deregistration
  auto it = std::find_if(
      registered_local_buffers_.begin(),
      registered_local_buffers_.end(),
      [&buf](const DeviceRegisteredBuffer& b) {
        return b.base_ptr == buf.base_ptr;
      });
  if (it != registered_local_buffers_.end()) {
    registered_local_buffers_.erase(it);
  }

  Backend::deregister_local_buffer(nccl_api_, nccl_comm_, buf);
}

template <typename Backend>
typename TorchCommWindowNCCLX<Backend>::DeviceWindow*
TorchCommWindowNCCLX<Backend>::get_device_window(
    int signal_count,
    int counter_count,
    int barrier_count) {
  checkCommAndThrow();

  if (Backend::select_device_win(win_, nccl_orig_win_) == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: Window not initialized. "
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

  // Create device window - the custom deleter handles all cleanup.
  // Both backends share the same create_device_window signature.
  device_window_ = Backend::create_device_window(
      nccl_comm_,
      nccl_api_,
      torch_comm_->getCudaApi(),
      config,
      Backend::select_device_win(win_, nccl_orig_win_),
      buf_tensor_.has_value() ? buf_tensor_->data_ptr() : nullptr,
      win_size_);

  return device_window_.get();
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
template <typename Backend>
void* TorchCommWindowNCCLX<Backend>::get_nvlink_address(
    int peer,
    size_t offset) {
  checkCommAndThrow();

  if (nccl_orig_win_ == nullptr) {
    throw std::runtime_error(
        "[TorchCommWindowNCCLX]: NCCL orig window not initialized. "
        "Call tensor_register first.");
  }

  void* outPtr = nullptr;
  CHECK_EQ(
      nccl_api_->winGetPeerDevicePointer(nccl_orig_win_, offset, peer, &outPtr),
      ncclSuccess)
      << "[TorchCommWindowNCCLX]: ncclGetPeerDevicePointer failed";

  return outPtr;
}
#endif

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
template class TorchCommWindowNCCLX<torchcomms::device::NCCLDeviceBackend>;
#if defined(ENABLE_PIPES)
template class TorchCommWindowNCCLX<torchcomms::device::PipesDeviceBackend>;
#endif
#else
template class TorchCommWindowNCCLX<HostOnlyBackend>;
#endif

} // namespace torch::comms
