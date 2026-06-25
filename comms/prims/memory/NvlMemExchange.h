// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// `<cuda.h>` (driver API) and `<cuda_runtime.h>` are NVIDIA-only. On AMD,
// fabric-handle support is unavailable; only the cudaIpc path is used, and its
// `cudaIpcMemHandle_t` parameter is HIPify-rewritten to `hipIpcMemHandle_t`
// (provided by `<hip/hip_runtime.h>`).
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"

namespace comms::prims {

#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
using FabricHandle = CUmemFabricHandle;
#else
struct FabricHandle {
  unsigned char data[64]; // CU_IPC_HANDLE_SIZE
};
// Stub the CUDA driver-API typedefs referenced by the always-declared VMM
// signature so the header (and downstream consumers like GpuMemHandler)
// compiles on AMD / pre-CUDA-12.3. The concrete VMM driver-API code lives
// behind `#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030` in
// the .cc; on these platforms `nvlMemExchangeVmm` is a throwing stub. Types
// match the real CUDA driver-API typedefs (`unsigned long long`).
#if defined(__HIP_PLATFORM_AMD__)
using CUdeviceptr = unsigned long long;
using CUmemGenericAllocationHandle = unsigned long long;
#endif
#endif

/**
 * NvlMemExchange - NVLink-domain peer-exchange building blocks.
 *
 * The peer exchange (export -> all-gather -> import -> map) that gives every
 * rank a VA onto every peer's backing allocation, for both VMM (fabric) and
 * cudaIpc modes. Used by GpuMemHandler. The VMM/fabric path is NVIDIA-only
 * (CUDA driver API + CUDA 12.3+); on AMD or pre-12.3, `nvlMemExchangeVmm` is
 * declared but its body throws. The cudaIpc path is HIPify-rewritten to
 * hipIpc on AMD.
 */

/**
 * VMM (fabric) peer exchange. all-gathers this rank's exported fabric handle
 * and allocation size, then imports + maps every peer's physical memory into a
 * fresh local virtual address with RW access. Results are written into the
 * per-rank output vectors (indexed by rank); the self slot of `peerPtrs` is
 * left untouched (the caller stores its own local VA there).
 *
 * COLLECTIVE OPERATION: all ranks must call. Throws on AMD or pre-CUDA-12.3.
 *
 * Output vectors must each have size `>= nRanks`.
 *
 * @param bootstrap Bootstrap interface for the allGather
 * @param selfRank This rank's ID (0..nRanks-1)
 * @param nRanks Total number of ranks
 * @param localHandle This rank's exported fabric handle
 * @param allocatedSize This rank's allocation size
 * @param peerPtrs [out] per-rank mapped peer pointers (size nRanks)
 * @param peerAllocHandles [out] per-rank imported allocation handles (size
 * nRanks)
 * @param peerAllocatedSizes [out] per-rank allocation sizes (size nRanks)
 */
void nvlMemExchangeVmm(
    meta::comms::IBootstrap& bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    const FabricHandle& localHandle,
    std::size_t allocatedSize,
    std::vector<CUdeviceptr>& peerPtrs,
    std::vector<CUmemGenericAllocationHandle>& peerAllocHandles,
    std::vector<std::size_t>& peerAllocatedSizes);

/**
 * cudaIpc peer exchange. all-gathers this rank's cudaIpc handle, then opens
 * every peer's handle into `peerPtrs` (indexed by rank). Intra-node only.
 *
 * COLLECTIVE OPERATION: all ranks must call.
 *
 * `peerPtrs` must have size `>= nRanks`.
 *
 * @param bootstrap Bootstrap interface for the allGather
 * @param selfRank This rank's ID (0..nRanks-1)
 * @param nRanks Total number of ranks
 * @param localHandle This rank's cudaIpc handle
 * @param peerPtrs [out] per-rank opened peer pointers (size nRanks)
 */
void nvlMemExchangeCudaIpc(
    meta::comms::IBootstrap& bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    const cudaIpcMemHandle_t& localHandle,
    std::vector<void*>& peerPtrs);

} // namespace comms::prims
