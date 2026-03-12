// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <doca_gpunetio_host.h>
#include <glog/logging.h>
#include <unistd.h>
#include <cstddef>

namespace comms::pipes {

// Wrapper around doca_gpu_dmabuf_fd that aligns size to host page size.
// DMA-BUF exports require both address and size aligned to host page size
// (64KB on aarch64/Grace). Returns the aligned size via output parameter
// for use in subsequent operations (e.g., ibv_reg_dmabuf_mr).
inline doca_error_t
getDmaBufFdAligned(doca_gpu* gpu, void* ptr, size_t size, int* fd) {
  static const size_t pageSize = sysconf(_SC_PAGESIZE);
  const size_t alignedSize = ((size + pageSize - 1) / pageSize) * pageSize;
  if (alignedSize != size) {
    VLOG(1) << "getDmaBufFdAligned: aligning DMA-BUF size from " << size
            << " to " << alignedSize << " bytes (page size " << pageSize << ")";
  }
  return doca_gpu_dmabuf_fd(gpu, ptr, alignedSize, fd);
}

} // namespace comms::pipes
