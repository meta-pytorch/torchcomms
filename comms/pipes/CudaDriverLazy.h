// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once
// Lazy-loaded CUDA driver API function pointers.
//
// Uses cudaGetDriverEntryPoint (from cudart) to resolve CUDA driver API
// symbols at runtime, avoiding a link-time dependency on libcuda.so.1.
// This is the same mechanism ncclx uses in cudawrap.cc.
//
// Usage:
//   if (comms::pipes::cuda_driver_lazy_init() != 0) {
//     // CUDA driver not available
//   }
//   CUresult err = comms::pipes::pfn_cuMemCreate(&handle, size, &prop, 0);

#include <cuda.h>

#include <cudaTypedefs.h>
#include <cuda_runtime.h>

namespace comms::pipes {

/// Initialize CUDA driver function pointers via cudaGetDriverEntryPoint.
/// Thread-safe (uses std::call_once). Returns 0 on success, non-zero if
/// the CUDA driver is unavailable (e.g., on CPU-only machines).
int cuda_driver_lazy_init();

// Device queries
extern PFN_cuDeviceGet_v2000 pfn_cuDeviceGet;
extern PFN_cuDeviceGetAttribute_v2000 pfn_cuDeviceGetAttribute;

// Context
extern PFN_cuCtxGetCurrent_v4000 pfn_cuCtxGetCurrent;

// Error handling
extern PFN_cuGetErrorString_v6000 pfn_cuGetErrorString;

// VMM allocation
extern PFN_cuMemCreate_v10020 pfn_cuMemCreate;
extern PFN_cuMemRelease_v10020 pfn_cuMemRelease;

// VMM address management
extern PFN_cuMemAddressReserve_v10020 pfn_cuMemAddressReserve;
extern PFN_cuMemAddressFree_v10020 pfn_cuMemAddressFree;

// VMM mapping
extern PFN_cuMemMap_v10020 pfn_cuMemMap;
extern PFN_cuMemUnmap_v10020 pfn_cuMemUnmap;
extern PFN_cuMemSetAccess_v10020 pfn_cuMemSetAccess;
extern PFN_cuMemGetAllocationGranularity_v10020
    pfn_cuMemGetAllocationGranularity;

// Fabric handle sharing
extern PFN_cuMemExportToShareableHandle_v10020 pfn_cuMemExportToShareableHandle;
extern PFN_cuMemImportFromShareableHandle_v10020
    pfn_cuMemImportFromShareableHandle;
extern PFN_cuMemGetAllocationPropertiesFromHandle_v10020
    pfn_cuMemGetAllocationPropertiesFromHandle;

// Allocation queries
extern PFN_cuMemRetainAllocationHandle_v11000 pfn_cuMemRetainAllocationHandle;
extern PFN_cuMemGetAddressRange_v3020 pfn_cuMemGetAddressRange;

} // namespace comms::pipes
