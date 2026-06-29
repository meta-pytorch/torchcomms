// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <glog/logging.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "comms/prims/platform/CudaDriverLazy.h"

namespace comms::prims {

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

// Mapping requested when exporting a GPU allocation as a DMA-BUF fd.
enum class DmaBufExportKind {
  // Standard mapping (cuMemGetHandleForAddressRange flags = 0).
  Default,
  // BAR1 PCIe mapping (CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE) so the fd
  // can back an mlx5dv_reg_dmabuf_mr with MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT
  // (NCCL's NCCL_IB_DATA_DIRECT / GDAKI path): the NIC writes straight to GPU
  // HBM over PCIe BAR1 instead of bouncing through Grace's C2C path. Requires
  // CUDA >= 12.8; on older toolkits it is unavailable (returns std::nullopt).
  Pcie,
};

// Export a GPU buffer as DMA-BUF with proper page alignment.
//
// Handles the full flow for cudaMalloc buffers on Grace/aarch64:
//   1. cuMemGetAddressRange → find CUDA allocation base
//   2. compute_dmabuf_alignment → align base/size to host page size
//   3. cuMemGetHandleForAddressRange → export as DMA-BUF fd
//
// `kind` selects the mapping: Default is the plain CUDA-driver DMA-BUF export
// (C2C) — doca_gpu_dmabuf_fd is a thin wrapper over the same
// cuMemGetHandleForAddressRange call, so no DOCA context is needed; Pcie
// requests the BAR1 PCIe mapping for mlx5 Data-Direct (see DmaBufExportKind).
// Returns std::nullopt on failure (the caller may fall back to ibv_reg_mr, or
// treat Data-Direct as mandatory and fail hard). The returned DmaBufExport
// contains the fd and alignment info needed for ibv_reg_dmabuf_mr (dmabufOffset
// as offset, ptr as iova).
inline std::optional<DmaBufExport> export_gpu_dmabuf_aligned(
    void* ptr,
    size_t size,
    DmaBufExportKind kind = DmaBufExportKind::Default) {
  // Resolve the cuMemGetHandleForAddressRange mapping flag. The PCIe
  // (Data-Direct) mapping requires CUDA >= 12.8; on older toolkits it cannot be
  // expressed, so report it as unavailable.
  unsigned long long handleFlags = 0;
  if (kind == DmaBufExportKind::Pcie) {
#if CUDA_VERSION >= 12080
    handleFlags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
#else
    (void)ptr;
    (void)size;
    return std::nullopt;
#endif
  }

  if (cuda_driver_lazy_init() != 0 || pfn_cuMemGetAddressRange == nullptr ||
      pfn_cuMemGetHandleForAddressRange == nullptr) {
    LOG(WARNING)
        << "export_gpu_dmabuf_aligned: CUDA driver API is not available";
    return std::nullopt;
  }

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
  CUresult fdRes = pfn_cuMemGetHandleForAddressRange(
      &fd,
      reinterpret_cast<CUdeviceptr>(alignment.alignedBase),
      alignment.alignedSize,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      handleFlags);
  if (fdRes != CUDA_SUCCESS || fd < 0) {
    LOG(WARNING) << "export_gpu_dmabuf_aligned: cuMemGetHandleForAddressRange"
                 << " failed err=" << fdRes << " ptr=" << ptr << " kind="
                 << (kind == DmaBufExportKind::Pcie ? "pcie" : "default")
                 << " alignedBase=" << alignment.alignedBase
                 << " alignedSize=" << alignment.alignedSize;
    return std::nullopt;
  }

  return DmaBufExport{fd, alignment};
}

} // namespace comms::prims
