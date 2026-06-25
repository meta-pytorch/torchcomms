// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/CuMemMapping.h"

namespace comms::prims {

#if CUDART_VERSION >= 12030
using FabricHandle = CUmemFabricHandle;
#else
struct FabricHandle {
  unsigned char data[64]; // CU_IPC_HANDLE_SIZE
};
#endif

/**
 * NvlMemExchange - NVLink-domain peer-exchange building blocks.
 *
 * The consolidated peer exchange (export -> all-gather -> import -> map) that
 * gives every rank a VA onto every peer's backing allocation, for both VMM
 * (fabric) and cudaIpc modes. Used by GpuMemHandler. The `cuMem*` fabric
 * driver-API paths are guarded by `#if CUDART_VERSION >= 12030`; on AMD only
 * the cudaIpc path (HIPify-rewritten) is used. Provides the FabricHandle
 * typedef.
 */

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
 * VMM (fabric) peer exchange.
 *
 * Exports `localHandle` as a fabric handle, all-gathers every rank's handle +
 * allocated size, then imports and maps each peer's backing allocation into a
 * fresh peer VA. `peerPtrs[rank]` is null for self; the caller fills the self
 * slot with `localPtr`. Throws std::runtime_error on any failure. Requires CUDA
 * 12.3+.
 */
NvlPeerMem nvlMemExchangeVmm(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    CUdevice cuDev,
    CUmemGenericAllocationHandle localHandle,
    void* localPtr,
    std::size_t allocatedSize);

/**
 * cudaIpc peer exchange.
 *
 * Exports `localPtr`'s cudaIpc handle, all-gathers handles, and opens each
 * peer's handle. The self slot of `peerPtrs` is filled with `localPtr`; peer
 * slots are owned by the CUDA IPC runtime (the caller closes them via
 * cudaIpcCloseMemHandle). `vmmMappings` is empty. Throws std::runtime_error on
 * any failure.
 */
NvlPeerMem nvlMemExchangeCudaIpc(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    void* localPtr);

} // namespace comms::prims
