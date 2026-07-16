// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "pipes_gda/PipesGdaDmaBuf.h" // @manual

#ifdef __HIP_PLATFORM_AMD__

#include <dlfcn.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

#include <hip/hip_runtime.h>

namespace comms::prims {

std::optional<DmaBufExport>
export_gpu_dmabuf_aligned(void* ptr, std::size_t size, DmaBufExportKind kind) {
  // mlx5 Data-Direct (BAR1 PCIe mapping) is NVIDIA-only; the AMD HSA export has
  // no equivalent, so report Pcie as unavailable. (Data-Direct is also disabled
  // in AMD NIC discovery, so this is never requested at runtime.)
  if (kind == DmaBufExportKind::Pcie) {
    return std::nullopt;
  }
  // Skip non-GPU pointers: HSA also exports host-pinned (`hipHostMalloc`)
  // memory, but that fd points at host pages — feeding it to ibv_reg_dmabuf_mr
  // silently corrupts RDMA or SIGSEGVs inside libmlx5.
  hipPointerAttribute_t attrs{};
  if (hipPointerGetAttributes(&attrs, ptr) != hipSuccess ||
      attrs.type != hipMemoryTypeDevice) {
    return std::nullopt;
  }

  using ExportDmabufFn = int (*)(const void*, size_t, int*, uint64_t*);
  static ExportDmabufFn exportDmabuf = nullptr;
  static std::once_flag dlOpenOnce;
  std::call_once(dlOpenOnce, []() {
    void* lib = dlopen("libhsa-runtime64.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) {
      lib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (lib) {
      exportDmabuf = reinterpret_cast<ExportDmabufFn>(
          dlsym(lib, "hsa_amd_portable_export_dmabuf"));
    }
  });

  if (!exportDmabuf) {
    return std::nullopt;
  }

  int fd = -1;
  uint64_t dmabufOffset = 0;
  int hsaStatus = exportDmabuf(ptr, size, &fd, &dmabufOffset);
  if (hsaStatus != 0 || fd < 0) {
    return std::nullopt;
  }

  // Skip dmabuf if HSA returned a non-page-aligned offset — `ibv_reg_dmabuf_mr`
  // would SIGSEGV inside libmlx5 on AMD MI300X.
  if (dmabufOffset % static_cast<uint64_t>(sysconf(_SC_PAGESIZE)) != 0) {
    close(fd);
    return std::nullopt;
  }

  uintptr_t alignedBase = reinterpret_cast<uintptr_t>(ptr) - dmabufOffset;
  DmaBufExport ex;
  ex.fd = fd;
  ex.alignment.alignedBase = reinterpret_cast<void*>(alignedBase);
  ex.alignment.alignedSize = size + dmabufOffset;
  ex.alignment.dmabufOffset = dmabufOffset;
  return ex;
}

} // namespace comms::prims

#endif // __HIP_PLATFORM_AMD__
