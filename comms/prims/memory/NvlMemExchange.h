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
#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/CuMemMapping.h"

namespace comms::prims {

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
 * VMM (fabric / POSIX FD) peer exchange.
 *
 * Exports `localHandle` (as fabric when `preferFabric`, else POSIX FD),
 * all-gathers every rank's shareable handle + allocated size, imports and maps
 * each peer's backing allocation, holds a post-import barrier so no rank closes
 * its exported POSIX FD before peers have duplicated it, then closes the local
 * exported fd.
 *
 * `peerPtrs[rank]` is null for self; the caller fills the self slot with
 * `localPtr`. Throws std::runtime_error on any failure. Requires CUDA 12.3+.
 */
NvlPeerMem nvlMemExchangeVmm(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    CUdevice cuDev,
    CUmemGenericAllocationHandle localHandle,
    void* localPtr,
    std::size_t allocatedSize,
    bool preferFabric);

/**
 * cudaIpc peer exchange.
 *
 * Exports `localPtr`'s cudaIpc handle, all-gathers handles, and opens each
 * peer's handle. The self slot of `peerPtrs` is filled with `localPtr`; peer
 * slots are owned by the CUDA IPC runtime (the caller closes them via
 * cudaIpcCloseMemHandle). `vmmMappings` is empty. Throws
 * std::runtime_error on any failure.
 */
NvlPeerMem nvlMemExchangeCudaIpc(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    void* localPtr);

} // namespace comms::prims
