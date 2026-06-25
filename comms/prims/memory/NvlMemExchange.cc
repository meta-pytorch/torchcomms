// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/memory/NvlMemExchange.h"

#include "comms/prims/core/Checks.h"

#if !defined(__HIP_PLATFORM_AMD__)
#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#include <stdexcept>
#include <vector>

namespace comms::prims {

namespace {

// Import + map one peer's fabric memory into a fresh local virtual address with
// RW access. Writes the imported handle and mapped pointer into the per-rank
// output vectors at index `rank`. Caller (nvlMemExchangeVmm) pre-sizes both
// output vectors to `nRanks` and bounds-checks at the public entry point;
// `.at()` is used at the write sites as a second proof of bounds for the
// clang-tidy ParameterUncheckedArrayBounds checker.
void importFabricPeerMemory(
    int32_t rank,
    const FabricHandle& handle,
    std::size_t peerAllocatedSize,
    std::vector<CUdeviceptr>& peerPtrs,
    std::vector<CUmemGenericAllocationHandle>& peerAllocHandles) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)rank;
  (void)handle;
  (void)peerAllocatedSize;
  (void)peerPtrs;
  (void)peerAllocHandles;
  throw std::runtime_error("Fabric handles require CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error("CUDA driver not available");
  }

  int cudaDev = 0;
  CUdevice cuDev;

  checkCudaError(cudaGetDevice(&cudaDev), "cudaGetDevice failed");
  checkCuError(pfn_cuDeviceGet(&cuDev, cudaDev), "cuDeviceGet failed");

  // Import the fabric handle to get allocation handle
  checkCuError(
      pfn_cuMemImportFromShareableHandle(
          &peerAllocHandles.at(rank),
          const_cast<void*>(static_cast<const void*>(&handle)),
          CU_MEM_HANDLE_TYPE_FABRIC),
      "cuMemImportFromShareableHandle failed");

  // Get allocation properties for granularity
  CUmemAllocationProp prop = {};
  checkCuError(
      pfn_cuMemGetAllocationPropertiesFromHandle(
          &prop, peerAllocHandles.at(rank)),
      "cuMemGetAllocationPropertiesFromHandle failed");

  std::size_t granularity = 0;
  checkCuError(
      pfn_cuMemGetAllocationGranularity(
          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      "cuMemGetAllocationGranularity failed");

  // Reserve virtual address space for peer memory
  checkCuError(
      pfn_cuMemAddressReserve(
          &peerPtrs.at(rank), peerAllocatedSize, granularity, 0, 0),
      "cuMemAddressReserve for peer failed");

  // Map peer's physical memory to our virtual address
  checkCuError(
      pfn_cuMemMap(
          peerPtrs.at(rank),
          peerAllocatedSize,
          0,
          peerAllocHandles.at(rank),
          0),
      "cuMemMap for peer failed");

  // Set access permissions for peer memory
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  checkCuError(
      pfn_cuMemSetAccess(peerPtrs.at(rank), peerAllocatedSize, &accessDesc, 1),
      "cuMemSetAccess for peer failed");
#endif
}

} // namespace

void nvlMemExchangeVmm(
    meta::comms::IBootstrap& bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    const FabricHandle& localHandle,
    std::size_t allocatedSize,
    std::vector<CUdeviceptr>& peerPtrs,
    std::vector<CUmemGenericAllocationHandle>& peerAllocHandles,
    std::vector<std::size_t>& peerAllocatedSizes) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)bootstrap;
  (void)selfRank;
  (void)nRanks;
  (void)localHandle;
  (void)allocatedSize;
  (void)peerPtrs;
  (void)peerAllocHandles;
  (void)peerAllocatedSizes;
  throw std::runtime_error("Fabric handles require CUDA 12.3+");
#else
  if (static_cast<std::size_t>(nRanks) > peerPtrs.size() ||
      static_cast<std::size_t>(nRanks) > peerAllocHandles.size() ||
      static_cast<std::size_t>(nRanks) > peerAllocatedSizes.size()) {
    throw std::runtime_error(
        "nvlMemExchangeVmm: output vectors must each have size >= nRanks");
  }

  // Prepare data for allGather: fabric handle + allocation size
  struct ExchangeData {
    FabricHandle handle;
    std::size_t allocatedSize;
  };

  std::vector<ExchangeData> allData(nRanks);
  allData[selfRank].handle = localHandle;
  allData[selfRank].allocatedSize = allocatedSize;

  // Exchange fabric handles with all ranks
  auto result =
      bootstrap
          .allGather(allData.data(), sizeof(ExchangeData), selfRank, nRanks)
          .get();
  if (result != 0) {
    throw std::runtime_error("nvlMemExchangeVmm allGather failed");
  }

  // Import peer memory from received fabric handles
  for (int32_t rank = 0; rank < nRanks; ++rank) {
    if (rank == selfRank) {
      continue;
    }
    peerAllocatedSizes.at(rank) = allData[rank].allocatedSize;
    importFabricPeerMemory(
        rank,
        allData[rank].handle,
        allData[rank].allocatedSize,
        peerPtrs,
        peerAllocHandles);
  }
#endif
}

void nvlMemExchangeCudaIpc(
    meta::comms::IBootstrap& bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    const cudaIpcMemHandle_t& localHandle,
    std::vector<void*>& peerPtrs) {
  if (static_cast<std::size_t>(nRanks) > peerPtrs.size()) {
    throw std::runtime_error(
        "nvlMemExchangeCudaIpc: peerPtrs must have size >= nRanks");
  }

  // Exchange IPC handles with all ranks
  std::vector<cudaIpcMemHandle_t> allHandles(nRanks);
  allHandles[selfRank] = localHandle;

  auto result =
      bootstrap
          .allGather(
              allHandles.data(), sizeof(cudaIpcMemHandle_t), selfRank, nRanks)
          .get();
  if (result != 0) {
    throw std::runtime_error("nvlMemExchangeCudaIpc allGather failed");
  }

  // Open peer memory handles
  for (int32_t rank = 0; rank < nRanks; ++rank) {
    if (rank == selfRank) {
      continue;
    }
    checkCudaError(
        cudaIpcOpenMemHandle(
            &peerPtrs.at(rank),
            allHandles[rank],
            cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle failed");
  }
}

} // namespace comms::prims
