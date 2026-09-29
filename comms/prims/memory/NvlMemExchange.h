// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// `<cuda.h>` (driver API) and `<cuda_runtime.h>` are NVIDIA-only. On AMD the
// concrete VMM driver-API calls are unavailable and the impl bodies throw;
// `CuMemAllocation.h` (included transitively via `CuMemMapping.h`) declares the
// matching AMD stub typedefs for `CUdevice` / `CUmemGenericAllocationHandle` /
// `CUdeviceptr`, and the `CUmemAllocationHandleType` stub below covers the one
// type unique to this header. Mirrors CuMemAllocation.h / MultimemHandler.h.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <sys/types.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/CuMemMapping.h"

namespace comms::prims {

/**
 * Fixed-width identity for a protocol layered over a multicast allocation.
 *
 * Every rank in an NVLink team must supply the same value. The owning protocol
 * assigns `protocol`, versions its parameter schema with `version`, and owns
 * the interpretation of `parameters`. The all-zero default identifies raw
 * multicast users with no additional protocol contract; it is still compared
 * across ranks.
 */
struct MulticastExchangeContract {
  uint64_t protocol{0};
  uint64_t version{0};
  std::array<uint64_t, 6> parameters{};

  bool operator==(const MulticastExchangeContract& other) const {
    return protocol == other.protocol && version == other.version &&
        parameters == other.parameters;
  }
};

#if defined(__HIP_PLATFORM_AMD__)
// Stub the one CUDA driver-API type unique to this header (the shared
// `CUdevice` / `CUmemGenericAllocationHandle` stubs come from CuMemAllocation.h
// via CuMemMapping.h). Concrete VMM/multicast driver-API calls are NVIDIA-only
// and live behind `#if !defined(__HIP_PLATFORM_AMD__)` in the .cc. The real
// CUDA type is an enum (int-backed), so an unsigned-int alias matches its ABI.
using CUmemAllocationHandleType = unsigned int;
#endif

#if CUDART_VERSION >= 12030
using FabricHandle = CUmemFabricHandle;
#else
struct FabricHandle {
  unsigned char data[64]; // CU_IPC_HANDLE_SIZE
};
#endif

/**
 * NvlMemExchange - NVLink-domain shared-memory building blocks.
 *
 * Bundles the two pieces needed to share a GPU buffer across NVLink-local
 * ranks:
 *
 *  1. Shareable-handle export/import for CUDA VMM allocation handles. The same
 *     export/import logic is needed in two places:
 *       - GpuMemHandler exports its unicast backing handle so peers can map the
 *         same physical allocation.
 *       - MultimemHandler exports the multicast object handle so peers can join
 *         the same multicast team.
 *  2. The consolidated peer exchange (export -> all-gather -> import -> map)
 *     used by GpuMemHandler to give every rank a VA onto every peer's backing
 *     allocation, for both VMM (fabric / POSIX FD) and cudaIpc modes.
 *
 * Two shareable-handle mechanisms are supported, selected by capability:
 *  - kFabric: CU_MEM_HANDLE_TYPE_FABRIC, works across hosts (e.g. GB200 MNNVL).
 *  - kPosixFd: CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, intra-host only (e.g.
 *    single-host H100 without IMEX). The descriptor number is process-local, so
 *    peers duplicate the exporter's fd via pidfd_open/pidfd_getfd before
 * import.
 */
enum class ShareableHandleType : uint8_t {
  kUnsupported,
  kFabric,
  kPosixFd,
};

/**
 * A shareable handle ready to be exchanged across ranks.
 *
 * For kFabric, `fabric` is the exported fabric handle and can be copied between
 * processes verbatim. For kPosixFd, `pid` and `fd` identify an open file
 * descriptor in the exporter's process; peers duplicate it before importing.
 */
struct ShareableHandle {
  ShareableHandleType type{ShareableHandleType::kUnsupported};
  FabricHandle fabric{}; // valid iff type == kFabric
  int32_t pid{-1}; // valid iff type == kPosixFd
  int32_t fd{-1}; // exporter-local fd; valid iff type == kPosixFd
};

/**
 * Maps a ShareableHandleType to the corresponding CUDA handle type. Throws
 * std::runtime_error for kUnsupported.
 */
CUmemAllocationHandleType toCudaHandleType(ShareableHandleType type);

/**
 * Selects the best shareable-handle type for `cudaDevice`: fabric if supported,
 * else POSIX FD, else kUnsupported. Initializes the CUDA driver lazily and
 * resolves the CUdevice internally.
 */
ShareableHandleType selectShareableHandleType(int cudaDevice);

/**
 * Exports `handle` as the given shareable-handle `type`.
 *
 * For kFabric the returned handle is self-contained. For kPosixFd the returned
 * ShareableHandle carries {pid=getpid(), fd=<newly exported fd>}; the CALLER
 * owns that fd and must keep it open until all peers have imported it, then
 * close it.
 *
 * Throws std::runtime_error on kUnsupported or any CUDA driver error.
 */
ShareableHandle exportShareableHandle(
    CUmemGenericAllocationHandle handle,
    ShareableHandleType type);

/**
 * Imports the physical handle described by `h`.
 *
 * For kFabric the fabric handle is imported directly. For kPosixFd the
 * exporter's fd is duplicated into this process via duplicateRemoteFd(), the
 * handle is imported, and the duplicated fd is closed before returning.
 *
 * Returns the imported physical handle (the caller owns the reference and must
 * eventually cuMemRelease it). Throws std::runtime_error on any failure.
 */
CUmemGenericAllocationHandle importShareableHandle(const ShareableHandle& h);

/**
 * The per-rank peer memory state produced by an NVLink-domain exchange.
 *
 * `peerPtrs[rank]` is a VA usable in local kernels to access rank `rank`'s
 * backing allocation; the self slot holds the local pointer. For VMM modes,
 * `vmmMappings` holds the RAII peer VAs; each mapping co-owns the imported peer
 * CuMemAllocation (via CuMemMapping's keepAlive), so releasing a mapping also
 * releases the imported physical handle -- there is no separate handle vector
 * to track. For cudaIpc mode `vmmMappings` is empty and the peer pointers are
 * owned by the CUDA IPC runtime (closed via cudaIpcCloseMemHandle by the
 * caller).
 */
struct NvlPeerMem {
  std::vector<void*> peerPtrs;
  std::vector<CuMemMapping> vmmMappings;
};

/**
 * Scratch storage allocated before a failure-safe NVLink memory exchange.
 *
 * Construct this during the caller's local preparation phase, before the
 * communicator-wide readiness agreement. The prepared exchange entry points
 * below then perform no host-vector allocation before their first collective.
 * A workspace is one-shot: either prepared helper consumes it on entry, and a
 * later call fails locally without touching bootstrap. The helper arguments
 * must also match the rank, rank count, and local pointer supplied here.
 * The workspace does not own `localPtr`; its allocation must remain alive
 * through the prepared collective. GpuMemHandler provides that ownership in
 * production.
 */
class NvlMemExchangeWorkspace {
 public:
  NvlMemExchangeWorkspace(int32_t rank, int32_t nRanks, void* localPtr);
  ~NvlMemExchangeWorkspace();

  NvlMemExchangeWorkspace(const NvlMemExchangeWorkspace&) = delete;
  NvlMemExchangeWorkspace& operator=(const NvlMemExchangeWorkspace&) = delete;
  NvlMemExchangeWorkspace(NvlMemExchangeWorkspace&&) = delete;
  NvlMemExchangeWorkspace& operator=(NvlMemExchangeWorkspace&&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  void consume(int32_t rank, int32_t nRanks, void* localPtr);

  friend NvlPeerMem nvlMemExchangeVmmPrepared(
      meta::comms::IBootstrap&,
      int32_t,
      int32_t,
      CUdevice,
      CUmemGenericAllocationHandle,
      void*,
      std::size_t,
      bool,
      NvlMemExchangeWorkspace&);
  friend NvlPeerMem nvlMemExchangeCudaIpcPrepared(
      meta::comms::IBootstrap&,
      int32_t,
      int32_t,
      void*,
      const cudaIpcMemHandle_t&,
      NvlMemExchangeWorkspace&);
};

/**
 * VMM (fabric / POSIX FD) peer exchange.
 *
 * Exports `localHandle` (as fabric when `preferFabric`, else POSIX FD) and
 * all-gathers export status with every rank's shareable handle + allocated
 * size. It then imports and maps each peer's backing allocation and all-gathers
 * import status. These agreements propagate rank-local failures and keep every
 * exported POSIX FD open until all peers have finished their import attempts.
 *
 * `peerPtrs[rank]` is null for self; the caller fills the self slot with
 * `localPtr`. Throws std::runtime_error on any failure. Requires CUDA 12.3+.
 * When provided, `localHandlePossiblyExposed` becomes true after local export
 * succeeds and immediately before the handle exchange begins.
 */
NvlPeerMem nvlMemExchangeVmm(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    CUdevice cuDev,
    CUmemGenericAllocationHandle localHandle,
    void* localPtr,
    std::size_t allocatedSize,
    bool preferFabric,
    bool* localHandlePossiblyExposed = nullptr);

NvlPeerMem nvlMemExchangeVmmPrepared(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    CUdevice cuDev,
    CUmemGenericAllocationHandle localHandle,
    void* localPtr,
    std::size_t allocatedSize,
    bool preferFabric,
    NvlMemExchangeWorkspace& workspace);

/**
 * cudaIpc peer exchange.
 *
 * Exports `localPtr`'s cudaIpc handle, all-gathers handles, and opens each
 * peer's handle. The self slot of `peerPtrs` is filled with `localPtr`; peer
 * slots are owned by the CUDA IPC runtime (the caller closes them via
 * cudaIpcCloseMemHandle). `vmmMappings` is empty. Throws
 * std::runtime_error on any failure.
 * When provided, `localHandlePossiblyExposed` becomes true after local export
 * succeeds and immediately before the handle exchange begins.
 */
NvlPeerMem nvlMemExchangeCudaIpc(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    void* localPtr,
    bool* localHandlePossiblyExposed = nullptr);

/**
 * Failure-safe cudaIpc exchange using storage and the local handle prepared
 * before the caller's communicator-wide readiness agreement.
 *
 * Handle exchange and peer import each end in a team status agreement. Any
 * failure closes peer mappings opened by this attempt before it is propagated.
 */
NvlPeerMem nvlMemExchangeCudaIpcPrepared(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    void* localPtr,
    const cudaIpcMemHandle_t& localHandle,
    NvlMemExchangeWorkspace& workspace);

} // namespace comms::prims
