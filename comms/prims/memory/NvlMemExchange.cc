// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/memory/NvlMemExchange.h"

#include "comms/prims/core/Checks.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/platform/CudaDriverLazy.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace comms::prims {

namespace {

// Import one peer's fabric handle, wrap it in a co-ownable CuMemAllocation, map
// it into a fresh peer VA, and record the mapping + pointer in `result`. The
// imported handle is owned by the CuMemAllocation that the peer VA mapping
// co-owns, so dropping the mapping releases the handle -- no separate handle
// bookkeeping.
void importAndMapPeerMemory(
    int32_t rank,
    const FabricHandle& handle,
    std::size_t peerAllocatedSize,
    CUdevice cuDev,
    NvlPeerMem& result) {
#if CUDART_VERSION < 12030
  (void)rank;
  (void)handle;
  (void)peerAllocatedSize;
  (void)cuDev;
  (void)result;
  throw std::runtime_error("Fabric handles require CUDA 12.3+");
#else
  CUmemGenericAllocationHandle peerHandle = 0;
  checkCuError(
      pfn_cuMemImportFromShareableHandle(
          &peerHandle,
          const_cast<void*>(static_cast<const void*>(&handle)),
          CU_MEM_HANDLE_TYPE_FABRIC),
      "cuMemImportFromShareableHandle failed");

  // CuMemAllocation::adopt() takes ownership of `peerHandle` on entry: on
  // internal failure (CUDA query, allocator bad_alloc) it releases the handle
  // and rethrows, so there is no raw-handle window for the caller to guard.
  // The returned unique_ptr promotes implicitly to shared_ptr for
  // CuMemMapping::overAllocation's keep-alive contract.
  std::shared_ptr<CuMemAllocation> peerAlloc =
      CuMemAllocation::adopt(peerHandle, cuDev, peerAllocatedSize);
  const std::size_t granularity = peerAlloc->granularity();

  result.vmmMappings.push_back(
      CuMemMapping::overAllocation(
          std::move(peerAlloc), peerAllocatedSize, granularity));
  result.peerPtrs.at(static_cast<std::size_t>(rank)) =
      // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer
      reinterpret_cast<void*>(result.vmmMappings.back().devicePtr());
#endif
}

} // namespace

NvlPeerMem nvlMemExchangeVmm(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    CUdevice cuDev,
    CUmemGenericAllocationHandle localHandle,
    void* localPtr,
    std::size_t allocatedSize) {
  NvlPeerMem result;
  result.peerPtrs.assign(static_cast<std::size_t>(nRanks), nullptr);
  result.peerPtrs[static_cast<std::size_t>(rank)] = localPtr;

#if CUDART_VERSION < 12030
  (void)bootstrap;
  (void)cuDev;
  (void)localHandle;
  (void)allocatedSize;
  throw std::runtime_error("nvlMemExchangeVmm requires CUDA 12.3+");
#else
  // Export the local handle once as a fabric handle.
  FabricHandle localFabric{};
  checkCuError(
      pfn_cuMemExportToShareableHandle(
          &localFabric, localHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0),
      "cuMemExportToShareableHandle for fabric handle failed");

  struct ExchangeData {
    FabricHandle handle{};
    std::size_t allocatedSize{};
  };

  std::vector<ExchangeData> allData(static_cast<std::size_t>(nRanks));
  allData[static_cast<std::size_t>(rank)].handle = localFabric;
  allData[static_cast<std::size_t>(rank)].allocatedSize = allocatedSize;

  auto gatherResult =
      bootstrap.allGather(allData.data(), sizeof(ExchangeData), rank, nRanks)
          .get();
  if (gatherResult != 0) {
    throw std::runtime_error("nvlMemExchangeVmm allGather failed");
  }

  for (int32_t peer = 0; peer < nRanks; ++peer) {
    if (peer == rank) {
      continue;
    }
    const auto peerIdx = static_cast<std::size_t>(peer);
    importAndMapPeerMemory(
        peer,
        allData[peerIdx].handle,
        allData[peerIdx].allocatedSize,
        cuDev,
        result);
  }

  return result;
#endif
}

NvlPeerMem nvlMemExchangeCudaIpc(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    void* localPtr) {
  NvlPeerMem result;
  result.peerPtrs.assign(static_cast<std::size_t>(nRanks), nullptr);
  result.peerPtrs[static_cast<std::size_t>(rank)] = localPtr;

  cudaIpcMemHandle_t localHandle{};
  checkCudaError(
      cudaIpcGetMemHandle(&localHandle, localPtr),
      "cudaIpcGetMemHandle failed");

  std::vector<cudaIpcMemHandle_t> allHandles(static_cast<std::size_t>(nRanks));
  allHandles[static_cast<std::size_t>(rank)] = localHandle;

  auto gatherResult =
      bootstrap
          .allGather(
              allHandles.data(), sizeof(cudaIpcMemHandle_t), rank, nRanks)
          .get();
  if (gatherResult != 0) {
    throw std::runtime_error("nvlMemExchangeCudaIpc allGather failed");
  }

  for (int32_t peer = 0; peer < nRanks; ++peer) {
    if (peer == rank) {
      continue;
    }
    const auto peerIdx = static_cast<std::size_t>(peer);
    checkCudaError(
        cudaIpcOpenMemHandle(
            &result.peerPtrs[peerIdx],
            allHandles[peerIdx],
            cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle failed");
  }

  return result;
}

} // namespace comms::prims
