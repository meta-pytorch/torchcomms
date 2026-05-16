// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/GpuMemHandler.h"

namespace comms::pipes {

/**
 * MultimemHandler - Manages one NVSwitch multicast allocation.
 *
 * Addressing model:
 *
 * Each rank owns a private local physical allocation. All ranks also join one
 * CUDA multicast object whose offsets describe the shared multicast window.
 * The handler exposes two device pointers on each rank:
 *
 * - local pointer: normal unicast VA mapped to this rank's local backing
 *   allocation. Loads from this pointer read the data delivered to the local
 *   rank, and ordinary stores update only the local rank.
 * - multimem pointer: multicast VA mapped to the shared multicast object.
 *   Device code writes this pointer with `multimem.st.*` or `multimem.red.*`
 *   instructions when it wants NVSwitch to replicate the write.
 *
 * A store to the multimem pointer is replicated by NVSwitch into every rank's
 * local backing allocation at the same object offset. For example, a multimem
 * store to `getMultimemDeviceMemPtr() + x` becomes a local write visible at
 * `getLocalDeviceMemPtr() + x` on every rank in the multicast team.
 *
 * Store path:
 *
 *   writer rank
 *   multimemPtr + x
 *        |
 *        |  multimem.st value
 *        v
 *   CUDA multicast object offset x
 *        |
 *        +--> rank 0 localPtr + x = value
 *        +--> rank 1 localPtr + x = value
 *        +--> rank 2 localPtr + x = value
 *        +--> ...
 *
 * A normal store to `localPtr + x` does not enter the multicast object and
 * only updates the writer's local backing allocation.
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
 * The caller must set the current CUDA device to `cudaDevice` before using this
 * handler. MultimemHandler uses `cudaDevice` to query driver properties and
 * describe allocations, but it does not call cudaSetDevice().
 */
class MultimemHandler {
 public:
  MultimemHandler(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int32_t commRank,
      std::vector<int> nvlRankToCommRank,
      int cudaDevice,
      std::size_t size);

  ~MultimemHandler();

  MultimemHandler(const MultimemHandler&) = delete;
  MultimemHandler& operator=(const MultimemHandler&) = delete;
  MultimemHandler(MultimemHandler&&) = delete;
  MultimemHandler& operator=(MultimemHandler&&) = delete;

  /**
   * Collective operation over all ranks in the multicast team.
   *
   * Allocates local backing memory, exchanges the multicast object handle,
   * binds each rank's local allocation to the multicast object, and maps the
   * multicast VA. Returns only after every rank has completed setup, so the
   * local and multicast pointers are ready for device use.
   */
  void exchange();

  /**
   * Returns this rank's local backing allocation. Loads from this pointer read
   * data delivered by multicast stores, and ordinary stores only update this
   * rank's local backing memory.
   */
  void* getLocalDeviceMemPtr() const;

  /**
   * Returns this rank's multicast VA. Device code uses this pointer with
   * `multimem.*` instructions to broadcast writes to every rank in the
   * multicast team.
   */
  void* getMultimemDeviceMemPtr() const;

  std::size_t getAllocatedSize() const {
    return allocatedSize_;
  }

  /**
   * Returns whether the current process can create CUDA multicast allocations
   * on `cudaDevice`. The caller is expected to have already made `cudaDevice`
   * current; this helper does not switch devices.
   */
  static bool isMultimemSupported(int cudaDevice);

 private:
  struct DeviceContext {
    CUdevice cuDev{0};
    CUmemAccessDesc accessDesc{};
  };

  struct AllocationLayout {
    std::size_t allocatedSize{0};
    std::size_t localGranularity{0};
    std::size_t multicastGranularity{0};
  };

  DeviceContext initializeDevice() const;
  AllocationLayout computeAllocationLayout(const DeviceContext& device) const;
  void createMulticastHandle(std::size_t allocatedSize);
  FabricHandle exportMulticastHandle();
  FabricHandle exchangeMulticastHandle(FabricHandle fabricHandle);
  void importMulticastHandle(const FabricHandle& fabricHandle);
  void allocateLocalMemory(
      const DeviceContext& device,
      const AllocationLayout& layout);
  void addLocalDeviceToMulticast(CUdevice cuDev);
  void bindLocalMemoryToMulticast(std::size_t allocatedSize);
  void mapMulticastMemory(
      const DeviceContext& device,
      const AllocationLayout& layout);
  void synchronizeRanks(const char* phase);
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

  bool exchanged_{false};
  bool deviceInitialized_{false};
  bool multicastHandleValid_{false};
  bool localHandleValid_{false};
  bool localMapped_{false};
  bool multicastBound_{false};
  bool multicastMapped_{false};

  CUdevice cuDev_{0};
  CUdeviceptr localPtr_{0};
  CUdeviceptr multicastPtr_{0};
  CUmemGenericAllocationHandle localAllocHandle_{0};
  CUmemGenericAllocationHandle multicastHandle_{0};

  std::size_t allocatedSize_{0};
};

} // namespace comms::pipes
