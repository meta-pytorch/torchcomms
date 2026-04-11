// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "StagedRdmaTransport.h"

#include <unistd.h>

#include <cuda_runtime.h>

#include <folly/synchronization/CallOnce.h>

#include <comms/ctran/utils/CudaWrap.h>
#include <comms/utils/cvars/nccl_cvars.h>

#include <fmt/core.h>

// ibverbx wraps all libibverbs types in its own namespace
using namespace ibverbx; // NOLINT(google-build-using-namespace)

namespace {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
folly::once_flag initOnceFlag;

void initEnvironment() {
  folly::call_once(initOnceFlag, [] { ncclCvarInit(); });
}

#define CUDA_CHECK(cmd)                     \
  do {                                      \
    auto err = (cmd);                       \
    if (err != cudaSuccess) {               \
      throw std::runtime_error(             \
          fmt::format(                      \
              "CUDA error {} at {}:{}: {}", \
              static_cast<int>(err),        \
              __FILE__,                     \
              __LINE__,                     \
              cudaGetErrorString(err)));    \
    }                                       \
  } while (0)

} // namespace

namespace torch::comms {

// --- StagedBuffer ---

StagedBuffer::StagedBuffer(size_t size, int cudaDev, ibverbx::IbvPd& pd)
    : size_(size), cudaDev_(cudaDev) {
  CUDA_CHECK(cudaSetDevice(cudaDev));
  CUDA_CHECK(cudaMalloc(&buf_, size));

  // Export dmabuf fd for GDR registration
  dmabufFd_ = ctran::utils::getCuMemDmaBufFd(buf_, size);
  if (dmabufFd_ < 0) {
    // Error path cleanup — cudaFree failure is non-actionable here.
    cudaFree(buf_);
    throw std::runtime_error("Failed to get dmabuf fd for GPU buffer");
  }

  // Register with IB for RDMA access via GPUDirect
  auto maybeMr = pd.regDmabufMr(
      /*offset=*/0,
      size,
      reinterpret_cast<uintptr_t>(buf_),
      dmabufFd_,
      static_cast<ibv_access_flags>(
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ));
  if (!maybeMr) {
    close(dmabufFd_);
    // Error path cleanup — cudaFree failure is non-actionable here.
    cudaFree(buf_);
    throw std::runtime_error(
        "Failed to register dmabuf MR: " + maybeMr.error().errStr);
  }
  mr_.emplace(std::move(*maybeMr));
}

StagedBuffer::~StagedBuffer() {
  // Destruction order: MR → dmabuf fd → GPU memory
  mr_.reset();
  if (dmabufFd_ >= 0) {
    close(dmabufFd_);
  }
  if (buf_) {
    // Destructor must not throw — cudaFree failure is non-actionable.
    cudaFree(buf_);
  }
}

StagedBuffer::StagedBuffer(StagedBuffer&& other) noexcept
    : buf_(other.buf_),
      size_(other.size_),
      cudaDev_(other.cudaDev_),
      dmabufFd_(other.dmabufFd_),
      mr_(std::move(other.mr_)) {
  other.buf_ = nullptr;
  other.dmabufFd_ = -1;
}

StagedBuffer& StagedBuffer::operator=(StagedBuffer&& other) noexcept {
  if (this != &other) {
    mr_.reset();
    if (dmabufFd_ >= 0) {
      close(dmabufFd_);
    }
    if (buf_) {
      // noexcept move — cudaFree failure is non-actionable.
      cudaFree(buf_);
    }

    buf_ = other.buf_;
    size_ = other.size_;
    cudaDev_ = other.cudaDev_;
    dmabufFd_ = other.dmabufFd_;
    mr_ = std::move(other.mr_);

    other.buf_ = nullptr;
    other.dmabufFd_ = -1;
  }
  return *this;
}

// --- StagedRdmaTransportBase ---

StagedRdmaTransportBase::StagedRdmaTransportBase(
    int cudaDev,
    folly::EventBase* evb,
    StagedTransferConfig config)
    : cudaDev_(cudaDev), config_(config), evb_(evb) {
  initEnvironment();
}

StagedRdmaTransportBase::~StagedRdmaTransportBase() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
}

int32_t StagedRdmaTransportBase::getDeviceId() const {
  if (!pd_.has_value()) {
    throw std::runtime_error(
        "getDeviceId() called before setupLocalTransport()");
  }
  return pd_->getDeviceId();
}

void StagedRdmaTransportBase::initIbResources() {
  throw std::runtime_error("initIbResources() not yet implemented");
}

void StagedRdmaTransportBase::connectQp(const std::string& /*peerConnInfo*/) {
  throw std::runtime_error("connectQp() not yet implemented");
}

std::string StagedRdmaTransportBase::serializeConnInfo(
    const StagingRendezvousInfo& /*localStaging*/) {
  throw std::runtime_error("serializeConnInfo() not yet implemented");
}

// --- StagedRdmaServerTransport ---

StagedRdmaServerTransport::~StagedRdmaServerTransport() = default;

std::string StagedRdmaServerTransport::setupLocalTransport() {
  throw std::runtime_error("setupLocalTransport() not yet implemented");
}

void StagedRdmaServerTransport::connectRemoteTransport(
    const std::string& /*peerConnInfo*/) {
  throw std::runtime_error("connectRemoteTransport() not yet implemented");
}

folly::SemiFuture<commResult_t> StagedRdmaServerTransport::send(
    const ScatterGatherDescriptor& /*src*/) {
  throw std::runtime_error("send() not yet implemented");
}

// --- StagedRdmaClientTransport ---

StagedRdmaClientTransport::~StagedRdmaClientTransport() = default;

std::string StagedRdmaClientTransport::setupLocalTransport() {
  throw std::runtime_error("setupLocalTransport() not yet implemented");
}

void StagedRdmaClientTransport::connectRemoteTransport(
    const std::string& /*peerConnInfo*/) {
  throw std::runtime_error("connectRemoteTransport() not yet implemented");
}

folly::SemiFuture<commResult_t> StagedRdmaClientTransport::recv(
    const ScatterGatherDescriptor& /*dst*/) {
  throw std::runtime_error("recv() not yet implemented");
}

} // namespace torch::comms
