// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/platform/CudaDriverLazy.h"

#include <cstdio>
#include <mutex>

namespace comms::prims {

// Function pointer globals (initially nullptr).
PFN_cuDeviceGet_v2000 pfn_cuDeviceGet = nullptr;
PFN_cuDeviceGetAttribute_v2000 pfn_cuDeviceGetAttribute = nullptr;
PFN_cuCtxGetCurrent_v4000 pfn_cuCtxGetCurrent = nullptr;
PFN_cuGetErrorString_v6000 pfn_cuGetErrorString = nullptr;
PFN_cuMemCreate_v10020 pfn_cuMemCreate = nullptr;
PFN_cuMemRelease_v10020 pfn_cuMemRelease = nullptr;
PFN_cuMemAddressReserve_v10020 pfn_cuMemAddressReserve = nullptr;
PFN_cuMemAddressFree_v10020 pfn_cuMemAddressFree = nullptr;
PFN_cuMemMap_v10020 pfn_cuMemMap = nullptr;
PFN_cuMemUnmap_v10020 pfn_cuMemUnmap = nullptr;
PFN_cuMemSetAccess_v10020 pfn_cuMemSetAccess = nullptr;
PFN_cuMemGetAllocationGranularity_v10020 pfn_cuMemGetAllocationGranularity =
    nullptr;
PFN_cuMemExportToShareableHandle_v10020 pfn_cuMemExportToShareableHandle =
    nullptr;
PFN_cuMemImportFromShareableHandle_v10020 pfn_cuMemImportFromShareableHandle =
    nullptr;
PFN_cuMemGetAllocationPropertiesFromHandle_v10020
    pfn_cuMemGetAllocationPropertiesFromHandle = nullptr;
PFN_cuMemRetainAllocationHandle_v11000 pfn_cuMemRetainAllocationHandle =
    nullptr;
PFN_cuMemGetAddressRange_v3020 pfn_cuMemGetAddressRange = nullptr;

namespace {

std::once_flag init_flag;
int init_result = -1;

int load_sym(const char* name, void** ptr, int version) {
  cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 13000
  auto res = cudaGetDriverEntryPointByVersion(
      name, ptr, version, cudaEnableDefault, &status);
#else
  auto res = cudaGetDriverEntryPoint(name, ptr, cudaEnableDefault, &status);
#endif
  if (res != cudaSuccess || status != cudaDriverEntryPointSuccess) {
    fprintf(
        stderr,
        "prims: failed to resolve CUDA driver symbol %s version %d "
        "(cudaError=%d, status=%d)\n",
        name,
        version,
        static_cast<int>(res),
        static_cast<int>(status));
    return -1;
  }
  return 0;
}

void do_init() {
#define LOAD(symbol, version)                                                \
  if (load_sym(#symbol, reinterpret_cast<void**>(&pfn_##symbol), version) != \
      0) {                                                                   \
    init_result = -1;                                                        \
    return;                                                                  \
  }

  LOAD(cuDeviceGet, 2000);
  LOAD(cuDeviceGetAttribute, 2000);
  LOAD(cuCtxGetCurrent, 4000);
  LOAD(cuGetErrorString, 6000);
  LOAD(cuMemCreate, 10020);
  LOAD(cuMemRelease, 10020);
  LOAD(cuMemAddressReserve, 10020);
  LOAD(cuMemAddressFree, 10020);
  LOAD(cuMemMap, 10020);
  LOAD(cuMemUnmap, 10020);
  LOAD(cuMemSetAccess, 10020);
  LOAD(cuMemGetAllocationGranularity, 10020);
  LOAD(cuMemExportToShareableHandle, 10020);
  LOAD(cuMemImportFromShareableHandle, 10020);
  LOAD(cuMemGetAllocationPropertiesFromHandle, 10020);
  LOAD(cuMemRetainAllocationHandle, 11000);
  LOAD(cuMemGetAddressRange, 3020);

#undef LOAD

  init_result = 0;
}

} // namespace

int cuda_driver_lazy_init() {
  std::call_once(init_flag, do_init);
  return init_result;
}

} // namespace comms::prims
