// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <vector>

// `<cuda.h>` (driver API) and `<cuda_runtime.h>` are NVIDIA-only. On AMD,
// fabric-handle support is unavailable; the class falls back to the cudaIpc
// path. HIPify rewrites the `cudaIpc*` / `cuMem*` / `CU*` symbol references
// in this header (and the corresponding `.cc`) to their `hip*` equivalents
// before compilation, and `<hip/hip_runtime.h>` provides the real HIP types
// (`hipIpcMemHandle_t`, `hipMemGenericAllocationHandle_t`, `hipDeviceptr_t`).
// The fabric-mode methods themselves are guarded by `#ifndef
// __HIP_PLATFORM_AMD__` in `GpuMemHandler.cc` so the unsupported driver-API
// calls aren't emitted on AMD.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <memory>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/memory/CuMemMapping.h"
#include "comms/prims/memory/NvlMemExchange.h"

namespace comms::prims {

// Forward-declared so GpuMemHandler.h stays lean for its many users; the
// composed multicast overlay is an implementation detail (out-of-line dtor in
// the .cc, which includes MultimemHandler.h).
class MultimemHandler;

/**
 * Memory sharing mode - determines which IPC mechanism to use.
 */
enum class MemSharingMode {
  // Use CUDA fabric handles (CU_MEM_HANDLE_TYPE_FABRIC)
  // Requires: Hopper+ GPU, CUDA 12.3+
  // Supports: Multi-node NVLink (GB200 NVL72)
  kFabric,

  // Use CUDA VMM allocations shared via POSIX file descriptors
  // (CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR).
  // Requires: CUDA 12.3+ VMM support.
  // Limitation: Intra-node only (descriptors are duplicated via pidfd_getfd).
  // Used for single-host H100 NVLink where fabric handles are unavailable.
  kPosixFd,

  // Use cudaIpcMemHandle_t
  // Works on: All CUDA GPUs
  // Limitation: Intra-node only
  kCudaIpc,

  // Like kCudaIpc, but on AMD the buffer is allocated GPU-uncached / HSA
  // fine-grained (hipExtMallocWithFlags + hipDeviceMallocUncached) so BNXT NICs
  // can DMA into it via dma-buf (see transport/amd/docs/design.md). On NVIDIA
  // it falls back to cudaMalloc; the IPC contract is identical.
  kCudaIpcUncached,
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
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
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
   * @param alignFloor Optional minimum allocation alignment/size floor. Must be
   * 0 (no floor) or a power of two. In a VMM mode it raises the physical
   * allocation's granularity/size floor so the allocation can satisfy a larger
   * granularity requirement (e.g. so it can later be bound into a multicast
   * object - see MultimemHandler::backingGranularity()). Ignored in cudaIpc
   * mode.
   *
   * @throws std::runtime_error if requested mode is not supported or allocation
   * fails
   */
  GpuMemHandler(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int32_t selfRank,
      int32_t nRanks,
      size_t size,
      MemSharingMode mode,
      std::size_t alignFloor = 0);

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
   * Get the local IPC handle (cudaIpc / hipIpc). Only valid in `kCudaIpc` /
   * `kCudaIpcUncached` mode.
   *
   * @throws std::runtime_error when called in a VMM (fabric / posix-fd) mode.
   */
  const cudaIpcMemHandle_t& getLocalIpcHandle() const;

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
   * Returns this handler's shared physical VMM allocation in a VMM-backed mode
   * (kFabric or kPosixFd), or nullptr in cudaIpc mode (cudaMalloc has no VMM
   * handle). The allocation is co-owned via shared_ptr; the multicast overlay
   * (exchangeMulticast) binds the same allocation so the unicast and multicast
   * views share one physical backing. Valid after construction.
   */
  std::shared_ptr<CuMemAllocation> allocation() const {
    return allocation_;
  }

  /**
   * Adds an NVSwitch multicast overlay over this handler's physical allocation.
   *
   * COLLECTIVE OPERATION over the NVLink-domain team described by
   * `nvlRankToCommRank`. Creates a multicast object, binds this rank's physical
   * allocation into it, and maps the multicast VA. After this call,
   * getMultimemDeviceMemPtr() returns a pointer usable with `multimem.*`
   * instructions, while getLocalDeviceMemPtr() continues to return the unicast
   * VA onto the same physical backing.
   *
   * Zeroing the local backing before peers read through the multicast VA is the
   * CALLER'S responsibility. This method intentionally issues no default-stream
   * operations (no implicit `cudaMemset` / `cudaStreamSynchronize`) so it can't
   * race with caller-managed compute streams; instead, the caller should
   * `cudaMemsetAsync` the backing on the same stream it will use for the
   * subsequent collective and synchronize there.
   *
   * The allocation must have been sized to backingGranularity() (pass it as the
   * constructor's `alignFloor`). Throws in cudaIpc mode (no VMM handle).
   *
   * @param commRank This process's rank in the global communicator
   * @param nvlRankToCommRank Maps each NVLink-local rank to its global rank
   * @param cudaDevice CUDA device ordinal (already current)
   */
  void exchangeMulticast(
      int32_t commRank,
      std::vector<int> nvlRankToCommRank,
      int cudaDevice);

  /**
   * Returns this rank's multicast VA. PRECONDITION: exchangeMulticast() done.
   */
  void* getMultimemDeviceMemPtr() const;

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

  /**
   * Returns whether multicast/multimem is supported on `cudaDevice`. Delegates
   * to MultimemHandler::isMultimemSupported.
   */
  static bool isMultimemSupported(int cudaDevice);

  /**
   * Returns the allocation alignment/size floor (alignFloor) required for a
   * physical allocation to be bindable into a multicast object for a team of
   * `nvlRanks` devices on `cudaDevice`. Delegates to
   * MultimemHandler::backingGranularity. Returns 0 if multimem is unsupported.
   */
  static std::size_t backingGranularity(int cudaDevice, int nvlRanks);

 private:
  void init(size_t size, std::size_t alignFloor);

  // VMM mode methods (shared by kFabric and kPosixFd). The physical allocation
  // requests both handle types when the device allows it; the actual exported
  // shareable-handle type is chosen at exchange time via supportsFabric().
  void allocateVmmMemory(size_t size, std::size_t alignFloor);
  void exchangeVmmHandles();
  void cleanupVmm();

  // CudaIpc mode methods
  void allocateCudaIpcMemory(size_t size);
  void exchangeCudaIpcHandles();
  void cleanupCudaIpc();

  // True for the VMM-backed modes (kFabric / kPosixFd), which share the same
  // NvlMemExchange-based code path. False for kCudaIpc.
  bool isVmmMode() const {
    return mode_ == MemSharingMode::kFabric ||
        mode_ == MemSharingMode::kPosixFd;
  }

  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  const int32_t selfRank_{-1};
  const int32_t nRanks_{-1};
  const MemSharingMode mode_{MemSharingMode::kCudaIpc};

  // Common state
  size_t allocatedSize_{0};
  bool exchanged_{false};

  // ---- VMM mode state (kFabric / kPosixFd) ----
  // The local physical allocation (co-owned via shared_ptr so a multicast
  // overlay can share it) and its unicast VA mapping (which also co-owns the
  // allocation via keepAlive). The actual exported shareable-handle type is
  // chosen at exchange time via allocation_->supportsFabric().
  std::shared_ptr<CuMemAllocation> allocation_;
  std::unique_ptr<CuMemMapping> unicastMapping_;
  // Peer mappings + pointers, produced by nvlMemExchange during
  // exchangeMemPtrs(). For VMM modes the peer mappings co-own their imported
  // allocations; for cudaIpc only the peer pointers (owned by the IPC runtime).
  NvlPeerMem peers_;
#ifndef __HIP_PLATFORM_AMD__
  // Optional NVSwitch multicast overlay over allocation_, created lazily by
  // exchangeMulticast(). NVIDIA-only; forward-declared, destroyed via the
  // out-of-line dtor (which includes MultimemHandler.h).
  std::unique_ptr<MultimemHandler> multimem_;
  // Topology args recorded by the first successful exchangeMulticast() so a
  // re-entry with mismatched args can be rejected instead of silently no-op'd.
  int32_t multimemCommRank_{-1};
  std::vector<int> multimemNvlRankToCommRank_;
  int multimemCudaDevice_{-1};
#endif

  // ---- CudaIpc mode state ----
  void* cudaIpcLocalPtr_{nullptr};
  cudaIpcMemHandle_t cudaIpcLocalHandle_{};
};

// Backwards compatibility alias
using FabricMemHandler = GpuMemHandler;

} // namespace comms::prims
