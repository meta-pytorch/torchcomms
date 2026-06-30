// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// `<cuda.h>` (driver API) and `<cuda_runtime.h>` are NVIDIA-only. Multimem is
// NVIDIA-only; on AMD impl bodies throw and the stub `CUdevice` /
// `CUmemGenericAllocationHandle` / `CUdeviceptr` typedefs come from
// `CuMemAllocation.h`.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/memory/CuMemMapping.h"
#include "comms/prims/memory/CuMulticastAllocation.h"
#include "comms/prims/memory/NvlMemExchange.h"

namespace comms::prims {

/**
 * MultimemHandler - the NVSwitch multicast overlay over a shared physical
 * allocation.
 *
 * Addressing model:
 *
 * A CUDA multicast object replicates writes into every team rank's local
 * backing allocation at the same object offset. This handler owns that
 * multicast overlay and a single device pointer:
 *
 * - multimem pointer: multicast VA mapped to the shared multicast object.
 *   Device code writes this pointer with `multimem.st.*` or `multimem.red.*`
 *   instructions when it wants NVSwitch to replicate the write across all team
 *   ranks. A multimem store to `getMultimemDeviceMemPtr() + x` becomes a local
 *   write visible at offset `x` in every rank's backing allocation.
 *
 * This handler does NOT own or serve a unicast (local) pointer. The local
 * backing is a shared `CuMemAllocation` provided by the caller (typically a
 * GpuMemHandler, which owns the unicast VA and is responsible for zeroing it);
 * this handler only binds that physical allocation into the multicast object
 * and maps the multicast VA. The `CuMemAllocation` is co-owned via shared_ptr,
 * so it stays alive as long as either the caller or this handler is alive.
 *
 * Pointer values are process-local CUDA virtual addresses. Do not compare the
 * numeric multimem pointer value across MPI ranks or encode protocols that
 * require it to match. The portable invariant is that the ranks map the same
 * multicast object and interpret offsets within that object identically.
 *
 * `bootstrap` is the global communicator bootstrap. `commRank` is this
 * process's rank in that global communicator. `nvlRankToCommRank` maps each
 * NVLink-local rank to its global communicator rank. The handler derives its
 * NVLink-local rank from the map, then exchange() uses the bootstrap's
 * NVLink-domain collectives. Callers do not need to pre-wrap the bootstrap in
 * an NVLink-local adapter.
 *
 * The multicast object handle is shared with fabric handles when the driver
 * supports them, and falls back to POSIX file descriptors for single-host H100
 * NVLS. POSIX FD sharing is intra-host only; the descriptor number itself is
 * process-local, so peers duplicate rank 0's FD before importing it.
 *
 * The caller must set the current CUDA device to `cudaDevice` before using this
 * handler. MultimemHandler uses `cudaDevice` to query driver properties and
 * describe allocations, but it does not call cudaSetDevice().
 */
class MultimemHandler {
 public:
  /**
   * Binds an existing physical VMM allocation into a new multicast object.
   *
   * `backing` is the shared physical allocation (typically a GpuMemHandler's,
   * obtained via GpuMemHandler::allocation()) that this handler binds into the
   * multicast object. `backing->size()` becomes this handler's allocated size,
   * so the caller must have sized `backing` to a multiple of max(local
   * allocation granularity, multicast granularity) -- see backingGranularity().
   *
   * Throws std::runtime_error if `backing` is null.
   */
  MultimemHandler(
      std::shared_ptr<CuMemAllocation> backing,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int32_t commRank,
      std::vector<int> nvlRankToCommRank,
      int cudaDevice);

  ~MultimemHandler();

  MultimemHandler(const MultimemHandler&) = delete;
  MultimemHandler& operator=(const MultimemHandler&) = delete;
  MultimemHandler(MultimemHandler&&) = delete;
  MultimemHandler& operator=(MultimemHandler&&) = delete;

  /**
   * Collective operation over all ranks in the multicast team.
   *
   * Creates the multicast object (team rank 0), exchanges its handle, binds
   * each rank's shared backing allocation to the multicast object, and maps the
   * multicast VA. Returns only after every rank has completed setup, so the
   * multicast pointer is ready for device use.
   */
  void exchange();

  /**
   * Returns this rank's multicast VA. Device code uses this pointer with
   * `multimem.*` instructions to broadcast writes to every rank in the
   * multicast team.
   */
  void* getMultimemDeviceMemPtr() const;

  std::size_t getAllocatedSize() const;

  /**
   * Returns this handler's shared physical backing allocation.
   */
  std::shared_ptr<CuMemAllocation> backing() const {
    return backing_;
  }

  /**
   * Returns whether the current process can create CUDA multicast allocations
   * on `cudaDevice`. The caller is expected to have already made `cudaDevice`
   * current; this helper does not switch devices.
   */
  static bool isMultimemSupported(int cudaDevice);

  /**
   * Returns max(local allocation granularity, multicast granularity) for the
   * handle type that would be selected on `cudaDevice` for a multicast team of
   * `nvlRanks` devices, both queried with the RECOMMENDED granularity. A
   * caller can use this as the `alignFloor` when allocating its physical
   * backing so that the resulting allocation is sized to be bindable into a
   * multicast object.
   *
   * Returns 0 if multimem is not supported on `cudaDevice`.
   */
  static std::size_t backingGranularity(int cudaDevice, int nvlRanks);

  // The shareable-handle type used to share the multicast object handle.
  // Aliased to ShareableHandleType so the shared export/import helpers can be
  // reused.
  using HandleType = ShareableHandleType;

 private:
  struct AllocationLayout {
    std::size_t allocatedSize{0};
    std::size_t granularity{0};
  };

  // Throws std::runtime_error if `backing` is null, otherwise returns it. Used
  // in the constructor's initializer list so the throw happens before const
  // members are initialized.
  static std::shared_ptr<CuMemAllocation> requireBacking(
      std::shared_ptr<CuMemAllocation> backing);

  CUdevice initializeDevice() const;
  AllocationLayout computeAllocationLayout() const;
  void createMulticastHandle(std::size_t allocatedSize);
  ShareableHandle exportMulticastHandle();
  ShareableHandle exchangeMulticastHandle(ShareableHandle handle);
  void importMulticastHandle(const ShareableHandle& handle);
  void addLocalDeviceToMulticast(CUdevice cuDev);
  void bindLocalMemoryToMulticast(std::size_t allocatedSize);
  void mapMulticastMemory(const AllocationLayout& layout);
  void synchronizeRanks(const char* phase);
  // AllGather handleType_ across the NVL team and throw if not every rank
  // selected the same one. Called from exchange() before rank 0 creates the
  // multicast object so a per-rank selection drift (e.g. one rank without IMEX
  // chose POSIX FD while peers chose fabric) is reported as a clear error
  // instead of a confusing export/import failure later.
  void agreeOnHandleType();
  std::string describeState(const char* failedPhase, const char* completedPhase)
      const;
  void cleanup();

  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  const int32_t commRank_{-1};
  const int32_t nvlRank_{-1};
  const int32_t nvlRanks_{-1};
  const std::vector<int> nvlRankToCommRank_;
  const int cudaDevice_{-1};
  const std::size_t requestedSize_{0};
  HandleType handleType_{HandleType::kUnsupported};

  bool exchanged_{false};
  // Set true by the catch arms in exchange() before cleanup() runs. Once
  // tripped, every subsequent exchange() call throws immediately rather
  // than re-entering the multi-phase setup. The handler is one-shot: a
  // failed exchange tears down `backing_` / `overlay_` / `multicastMapping_`
  // via cleanup(), so a naive retry would crash on `backing_->size()` /
  // `backing_->handle()` deref. To retry, the caller must construct a new
  // MultimemHandler with the same (still caller-owned) backing.
  bool failed_{false};
  bool deviceInitialized_{false};
  // Rank 0 keeps the exported POSIX FD open until peers have duplicated it.
  int multicastExportedFd_{-1};

  CUdevice cuDev_{0};
  std::size_t allocatedSize_{0};

  // The shared physical backing (a GpuMemHandler's, or any CuMemAllocation),
  // co-owned so it outlives this handler's multicast binding. The multicast
  // object (overlay_) binds backing_->handle() and replicates writes into it;
  // the multicast VA mapping (multicastMapping_) co-owns overlay_.
  std::shared_ptr<CuMemAllocation> backing_;
  std::shared_ptr<CuMulticastAllocation> overlay_;
  std::unique_ptr<CuMemMapping> multicastMapping_;
};

} // namespace comms::prims
