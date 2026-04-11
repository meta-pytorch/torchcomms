// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <optional>

#include <comms/ctran/ibverbx/IbvMr.h>
#include <comms/ctran/ibverbx/IbvPd.h>

namespace torch::comms {

// RAII wrapper for a GPU staging buffer registered for GPUDirect RDMA via
// dmabuf. Allocates GPU memory with cudaMalloc, exports a dmabuf fd, and
// registers it with the IB protection domain for zero-copy RDMA access.
//
// Destruction order: deregister MR → close dmabuf fd → cudaFree.
class StagedBuffer {
 public:
  StagedBuffer(size_t size, int cudaDev, ibverbx::IbvPd& pd);
  ~StagedBuffer();

  // Move-only
  StagedBuffer(StagedBuffer&& other) noexcept;
  StagedBuffer& operator=(StagedBuffer&& other) noexcept;
  StagedBuffer(const StagedBuffer&) = delete;
  StagedBuffer& operator=(const StagedBuffer&) = delete;

  void* data() const {
    return buf_;
  }
  size_t size() const {
    return size_;
  }
  int cudaDev() const {
    return cudaDev_;
  }
  uint32_t lkey() const {
    return mr_->mr()->lkey;
  }
  uint32_t rkey() const {
    return mr_->mr()->rkey;
  }

 private:
  void* buf_{nullptr};
  size_t size_{0};
  int cudaDev_{-1};
  int dmabufFd_{-1};
  std::optional<ibverbx::IbvMr> mr_;
};

} // namespace torch::comms
