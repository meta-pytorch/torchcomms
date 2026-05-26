// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/CudaDriverLazy.h"

#include <cstdio>
#include <mutex>

namespace comms::pipes {

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
#if CUDART_VERSION >= 12010
PFN_cuMulticastCreate_v12010 pfn_cuMulticastCreate = nullptr;
PFN_cuMulticastAddDevice_v12010 pfn_cuMulticastAddDevice = nullptr;
PFN_cuMulticastBindMem_v12010 pfn_cuMulticastBindMem = nullptr;
PFN_cuMulticastBindAddr_v12010 pfn_cuMulticastBindAddr = nullptr;
PFN_cuMulticastUnbind_v12010 pfn_cuMulticastUnbind = nullptr;
PFN_cuMulticastGetGranularity_v12010 pfn_cuMulticastGetGranularity = nullptr;
#endif

namespace {

std::once_flag init_flag;
int init_result = -1;

int load_sym(const char* name, void** ptr) {
  cudaDriverEntryPointQueryResult status;
  auto res = cudaGetDriverEntryPoint(name, ptr, cudaEnableDefault, &status);
  if (res != cudaSuccess || status != cudaDriverEntryPointSuccess) {
    fprintf(
        stderr,
        "pipes: failed to resolve CUDA driver symbol %s "
        "(cudaError=%d, status=%d)\n",
        name,
        static_cast<int>(res),
        static_cast<int>(status));
    return -1;
  }
  return 0;
}

void load_optional_sym(const char* name, void** ptr) {
  cudaDriverEntryPointQueryResult status;
  auto res = cudaGetDriverEntryPoint(name, ptr, cudaEnableDefault, &status);
  if (res != cudaSuccess || status != cudaDriverEntryPointSuccess) {
    *ptr = nullptr;
  }
}

void do_init() {
#define LOAD(symbol)                                                     \
  if (load_sym(#symbol, reinterpret_cast<void**>(&pfn_##symbol)) != 0) { \
    init_result = -1;                                                    \
    return;                                                              \
  }

  LOAD(cuDeviceGet);
  LOAD(cuDeviceGetAttribute);
  LOAD(cuCtxGetCurrent);
  LOAD(cuGetErrorString);
  LOAD(cuMemCreate);
  LOAD(cuMemRelease);
  LOAD(cuMemAddressReserve);
  LOAD(cuMemAddressFree);
  LOAD(cuMemMap);
  LOAD(cuMemUnmap);
  LOAD(cuMemSetAccess);
  LOAD(cuMemGetAllocationGranularity);
  LOAD(cuMemExportToShareableHandle);
  LOAD(cuMemImportFromShareableHandle);
  LOAD(cuMemGetAllocationPropertiesFromHandle);
  LOAD(cuMemRetainAllocationHandle);
  LOAD(cuMemGetAddressRange);

#if CUDART_VERSION >= 12010
#define LOAD_OPTIONAL(symbol) \
  load_optional_sym(#symbol, reinterpret_cast<void**>(&pfn_##symbol))

  LOAD_OPTIONAL(cuMulticastCreate);
  LOAD_OPTIONAL(cuMulticastAddDevice);
  LOAD_OPTIONAL(cuMulticastBindMem);
  LOAD_OPTIONAL(cuMulticastBindAddr);
  LOAD_OPTIONAL(cuMulticastUnbind);
  LOAD_OPTIONAL(cuMulticastGetGranularity);

#undef LOAD_OPTIONAL
#endif

#undef LOAD

  init_result = 0;
}

} // namespace

int cuda_driver_lazy_init() {
  std::call_once(init_flag, do_init);
  return init_result;
}

} // namespace comms::pipes
