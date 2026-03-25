// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <doca_gpunetio_host.h>
#include <glog/logging.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "comms/pipes/CudaDriverLazy.h"

namespace comms::pipes {

// Result of page-aligning a CUDA allocation for DMA-BUF export.
struct DmaBufAlignment {
  void* alignedBase;
  size_t alignedSize;
  uint64_t dmabufOffset; // offset of the user pointer within the aligned range
};

// Compute page-aligned base address and size for DMA-BUF export.
//
// cuMemGetHandleForAddressRange (inside doca_gpu_dmabuf_fd) requires both
// address and size aligned to the host page size. cudaMalloc may return
// addresses not aligned to the 64KB Grace page size. This function takes
// the allocation base/size from cuMemGetAddressRange and computes the
// aligned range that covers the full allocation.
//
// @param allocBase  Allocation base from cuMemGetAddressRange
// @param allocSize  Allocation size from cuMemGetAddressRange
// @param ptr        User buffer pointer (within the allocation)
// @param pageSize   Host page size (sysconf(_SC_PAGESIZE))
inline DmaBufAlignment compute_dmabuf_alignment(
    uintptr_t allocBase,
    size_t allocSize,
    void* ptr,
    size_t pageSize) {
  auto alignedBase = allocBase & ~(pageSize - 1);
  size_t baseOffset = allocBase - alignedBase;
  size_t alignedSize =
      ((allocSize + baseOffset + pageSize - 1) / pageSize) * pageSize;
  uint64_t dmabufOffset = reinterpret_cast<uintptr_t>(ptr) - alignedBase;
  return {reinterpret_cast<void*>(alignedBase), alignedSize, dmabufOffset};
}

// Result of exporting a GPU buffer as DMA-BUF with page alignment.
struct DmaBufExport {
  int fd; // DMA-BUF file descriptor
  DmaBufAlignment alignment; // alignment info for ibv_reg_dmabuf_mr
};

// Export a GPU buffer as DMA-BUF with proper page alignment.
//
// Handles the full flow for cudaMalloc buffers on Grace/aarch64:
//   1. cuMemGetAddressRange → find CUDA allocation base
//   2. compute_dmabuf_alignment → align base/size to host page size
//   3. doca_gpu_dmabuf_fd → export as DMA-BUF
//
// Returns std::nullopt on failure (caller can fall back to ibv_reg_mr).
// The returned DmaBufExport contains the fd and alignment info needed
// for ibv_reg_dmabuf_mr (dmabufOffset as offset, ptr as iova).
inline std::optional<DmaBufExport>
export_gpu_dmabuf_aligned(doca_gpu* gpu, void* ptr, size_t size) {
  CUdeviceptr allocBase = 0;
  size_t allocSize = 0;
  CUresult cuRes =
      pfn_cuMemGetAddressRange(&allocBase, &allocSize, (CUdeviceptr)ptr);
  if (cuRes != CUDA_SUCCESS || allocBase == 0) {
    LOG(WARNING) << "export_gpu_dmabuf_aligned: cuMemGetAddressRange failed"
                 << " err=" << cuRes << " ptr=" << ptr << " size=" << size;
    return std::nullopt;
  }

  static const size_t pageSize = sysconf(_SC_PAGESIZE);
  auto alignment =
      compute_dmabuf_alignment(allocBase, allocSize, ptr, pageSize);

  int fd = -1;
  doca_error_t err = doca_gpu_dmabuf_fd(
      gpu, alignment.alignedBase, alignment.alignedSize, &fd);
  if (err != DOCA_SUCCESS || fd < 0) {
    LOG(WARNING) << "export_gpu_dmabuf_aligned: doca_gpu_dmabuf_fd failed"
                 << " err=" << err << " ptr=" << ptr
                 << " alignedBase=" << alignment.alignedBase
                 << " alignedSize=" << alignment.alignedSize;
    return std::nullopt;
  }

  return DmaBufExport{fd, alignment};
}

} // namespace comms::pipes
