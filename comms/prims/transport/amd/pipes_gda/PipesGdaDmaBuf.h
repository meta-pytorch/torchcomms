// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// PipesGdaDmaBuf - GPU DMA-BUF export for AMD/HIP builds.
//
// HSA equivalent of NVIDIA's `cuMemGetAddressRange + doca_gpu_dmabuf_fd`
// pipeline (`platform/DocaHostUtils.h`). Kept out of `PipesGdaHost.{h,cc}`
// (which mirror `doca_*` 1:1) because it is a Meta helper, not part of the
// `pipes_gda_*` / DOCA host API surface.

#pragma once

#ifdef __HIP_PLATFORM_AMD__

#include <cstddef>
#include <cstdint>
#include <optional>

namespace comms::prims {

struct DmaBufAlignment {
  void* alignedBase{nullptr};
  std::size_t alignedSize{0};
  uint64_t dmabufOffset{0};
};

struct DmaBufExport {
  int fd{-1};
  DmaBufAlignment alignment;
};

// Mapping requested when exporting a GPU allocation as a DMA-BUF fd. mlx5
// Data-Direct (Pcie / BAR1 PCIe-mapped DMA-BUF) is NVIDIA-only; on AMD the
// Pcie kind is unsupported and export returns std::nullopt. (Data-Direct is
// also disabled in NIC discovery on AMD, so Pcie is never requested at runtime;
// the enum exists so the shared registerBuffer compiles under HIP.)
enum class DmaBufExportKind {
  Default,
  Pcie,
};

std::optional<DmaBufExport> export_gpu_dmabuf_aligned(
    void* ptr,
    std::size_t size,
    DmaBufExportKind kind = DmaBufExportKind::Default);

} // namespace comms::prims

#endif // __HIP_PLATFORM_AMD__
