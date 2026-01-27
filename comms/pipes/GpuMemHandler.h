// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "comms/ctran/interfaces/IBootstrap.h"

namespace comms::pipes {

#if CUDART_VERSION >= 12030
using FabricHandle = CUmemFabricHandle;
#else
struct FabricHandle {
  unsigned char data[64]; // CU_IPC_HANDLE_SIZE
};
#endif

/**
 * Memory sharing mode - determines which IPC mechanism to use.
 */
enum class MemSharingMode {
  // Use CUDA fabric handles (CU_MEM_HANDLE_TYPE_FABRIC)
  // Requires: Hopper+ GPU, CUDA 12.3+
  // Supports: Multi-node NVLink (GB200 NVL72)
  kFabric,

  // Use cudaIpcMemHandle_t
  // Works on: All CUDA GPUs
  // Limitation: Intra-node only
  kCudaIpc,
};

/**
 * Union to hold either fabric handle or cudaIpc handle for exchange.
 */
union IpcHandle {
  FabricHandle fabric;
  cudaIpcMemHandle_t cudaIpc;
};

/**
 * GpuMemHandler - Manages GPU memory sharing across processes.
 *
 * This class provides GPU memory sharing with automatic fallback:
 * 1. CUDA Fabric handles (preferred) - for H100+, CUDA 12.3+, enables GB200
 * MNNVL
 * 2. cudaIpcMemHandle_t (fallback) - for older GPUs/CUDA, intra-node only
 *
 * The mode is automatically detected at construction time based on hardware
 * and CUDA version capabilities.
 *
 * DESIGN NOTE: This class allocates memory internally rather than accepting
 * an external buffer. This is intentional because fabric handles require
 * memory to be allocated with specific flags (CU_MEM_HANDLE_TYPE_FABRIC) at
 * allocation time - you cannot create a fabric handle from an arbitrary
 * cudaMalloc'd buffer. By owning the allocation, GpuMemHandler ensures the
 * memory is properly allocated for the chosen sharing mode.
 *
 * This class is NOT thread-safe. Only one thread per process should use it.
 *
 * USAGE:
 *   GpuMemHandler handler(bootstrap, selfRank, nRanks, size);
 *   handler.exchangeMemPtrs();
 *   void* localPtr = handler.getLocalDeviceMemPtr();  // get allocated buffer
 *   void* peerPtr = handler.getPeerDeviceMemPtr(peerRank);
 */
class GpuMemHandler {
 public:
  /**
   * Constructor - Allocates shareable GPU memory.
   *
   * Automatically selects the best available sharing mode:
   * - Fabric handles on Hopper+ with CUDA 12.3+
   * - cudaIpcMemHandle on older systems
   *
   * @param bootstrap Bootstrap interface for collective operations
   * @param selfRank This rank's ID (0 to nRanks-1)
   * @param nRanks Total number of ranks
   * @param size Size of memory to allocate
   *
   * @throws std::runtime_error if memory allocation fails
   */
  GpuMemHandler(
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      int32_t selfRank,
      int32_t nRanks,
      size_t size);

  /**
   * Constructor with explicit mode selection.
   *
   * @param bootstrap Bootstrap interface for collective operations
   * @param selfRank This rank's ID (0 to nRanks-1)
   * @param nRanks Total number of ranks
   * @param size Size of memory to allocate
   * @param mode Explicitly select fabric or cudaIpc mode
   *
   * @throws std::runtime_error if requested mode is not supported or allocation
   * fails
   */
  GpuMemHandler(
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      int32_t selfRank,
      int32_t nRanks,
      size_t size,
      MemSharingMode mode);

  ~GpuMemHandler();

  // Non-copyable, non-movable
  GpuMemHandler(const GpuMemHandler&) = delete;
  GpuMemHandler& operator=(const GpuMemHandler&) = delete;
  GpuMemHandler(GpuMemHandler&&) = delete;
  GpuMemHandler& operator=(GpuMemHandler&&) = delete;

  /**
   * Exchange memory handles across all ranks.
   *
   * COLLECTIVE OPERATION: All ranks must call this.
   *
   * After this call, getPeerDeviceMemPtr() can be used to access
   * any peer's memory.
   *
   * @throws std::runtime_error if exchange fails
   */
  void exchangeMemPtrs();

  /**
   * Get pointer to local memory (this rank's allocation).
   *
   * Can be called before or after exchangeMemPtrs().
   */
  void* getLocalDeviceMemPtr() const;

  /**
   * Get pointer to peer's memory.
   *
   * PRECONDITION: exchangeMemPtrs() must have been called.
   *
   * @param rank Peer rank to access (can be selfRank for local ptr)
   * @return Pointer usable in local CUDA kernels to access peer's memory
   *
   * @throws std::runtime_error if exchange hasn't happened yet
   */
  void* getPeerDeviceMemPtr(int32_t rank) const;

  /**
   * Get the actual allocated size (may be larger than requested due to
   * alignment).
   */
  size_t getAllocatedSize() const {
    return allocatedSize_;
  }

  /**
   * Get the memory sharing mode being used.
   */
  MemSharingMode getMode() const {
    return mode_;
  }

  /**
   * Check if the current GPU supports fabric handles.
   *
   * @return true if Hopper+ GPU with CUDA 12.3+ and fabric support enabled
   */
  static bool isFabricHandleSupported();

  /**
   * Get the best available sharing mode for the current system.
   */
  static MemSharingMode detectBestMode();

 private:
  void init(size_t size);

  // Fabric mode methods
  void allocateFabricMemory(size_t size);
  void exchangeFabricHandles();
  void importFabricPeerMemory(
      int32_t rank,
      const FabricHandle& handle,
      size_t peerAllocatedSize);
  void cleanupFabric();

  // CudaIpc mode methods
  void allocateCudaIpcMemory(size_t size);
  void exchangeCudaIpcHandles();
  void cleanupCudaIpc();

  std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap_;
  const int32_t selfRank_{-1};
  const int32_t nRanks_{-1};
  const MemSharingMode mode_{MemSharingMode::kCudaIpc};

  // Common state
  size_t allocatedSize_{0};
  bool exchanged_{false};

  // ---- Fabric mode state ----
  CUdeviceptr fabricLocalPtr_{0};
  CUmemGenericAllocationHandle fabricLocalAllocHandle_{0};
  FabricHandle fabricLocalHandle_{};
  std::vector<CUdeviceptr> fabricPeerPtrs_;
  std::vector<CUmemGenericAllocationHandle> fabricPeerAllocHandles_;
  std::vector<size_t> fabricPeerAllocatedSizes_;

  // ---- CudaIpc mode state ----
  void* cudaIpcLocalPtr_{nullptr};
  cudaIpcMemHandle_t cudaIpcLocalHandle_{};
  std::vector<void*> cudaIpcPeerPtrs_;
};

// Backwards compatibility alias
using FabricMemHandler = GpuMemHandler;

} // namespace comms::pipes
