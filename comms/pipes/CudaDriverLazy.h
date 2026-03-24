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

#include <cstddef>

#if defined(__has_include)
#if __has_include(<cudaTypedefs.h>)
#define TORCHCOMMS_HAVE_CUDA_TYPEDEFS 1
#endif
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#if defined(TORCHCOMMS_HAVE_CUDA_TYPEDEFS)
#include <cudaTypedefs.h>
#endif

namespace comms::pipes {

/// Initialize CUDA driver function pointers via cudaGetDriverEntryPoint.
/// Thread-safe (uses std::call_once). Returns 0 on success, non-zero if
/// the CUDA driver is unavailable (e.g., on CPU-only machines).
int cuda_driver_lazy_init();

#if defined(TORCHCOMMS_HAVE_CUDA_TYPEDEFS)
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
extern PFN_cuMemExportToShareableHandle_v10020
    pfn_cuMemExportToShareableHandle;
extern PFN_cuMemImportFromShareableHandle_v10020
    pfn_cuMemImportFromShareableHandle;
extern PFN_cuMemGetAllocationPropertiesFromHandle_v10020
    pfn_cuMemGetAllocationPropertiesFromHandle;

// Allocation queries
extern PFN_cuMemRetainAllocationHandle_v11000 pfn_cuMemRetainAllocationHandle;
extern PFN_cuMemGetAddressRange_v3020 pfn_cuMemGetAddressRange;

#else
// -----------------------------
// ROCm/without cudaTypedefs stubs
// -----------------------------

// Provide minimal types used by this module so it can compile without
// cudaTypedefs.h. The fabric path is gated elsewhere.

#ifndef CUresult
using CUresult = int;
#endif
#ifndef CUdevice
using CUdevice = int;
#endif
#ifndef CUcontext
using CUcontext = void*;
#endif

// Keep signatures permissive; only compilation matters for ROCm.
using PFN_cuDeviceGet_v2000 = CUresult (*)(CUdevice*, int);
using PFN_cuDeviceGetAttribute_v2000 = CUresult (*)(int*, int, CUdevice);
using PFN_cuCtxGetCurrent_v4000 = CUresult (*)(CUcontext*);
// Matches cuGetErrorString(CUresult err, const char** pStr)
using PFN_cuGetErrorString_v6000 = CUresult (*)(CUresult, const char**);

using PFN_cuMemCreate_v10020 = CUresult (*)(void**, size_t, void*, unsigned int);
using PFN_cuMemRelease_v10020 = CUresult (*)(void*);

using PFN_cuMemAddressReserve_v10020 =
    CUresult (*)(unsigned long long*, size_t, size_t, int, int);
using PFN_cuMemAddressFree_v10020 =
    CUresult (*)(unsigned long long, size_t);

using PFN_cuMemMap_v10020 =
    CUresult (*)(unsigned long long, size_t, size_t, void*, int);
using PFN_cuMemUnmap_v10020 = CUresult (*)(unsigned long long, size_t);
using PFN_cuMemSetAccess_v10020 =
    CUresult (*)(unsigned long long, size_t, void*, unsigned int);
using PFN_cuMemGetAllocationGranularity_v10020 =
    CUresult (*)(size_t*, void*, int);

using PFN_cuMemExportToShareableHandle_v10020 =
    CUresult (*)(void*, void*, int, unsigned int);
using PFN_cuMemImportFromShareableHandle_v10020 =
    CUresult (*)(void*, void*, int);
using PFN_cuMemGetAllocationPropertiesFromHandle_v10020 =
    CUresult (*)(void*, void*);

using PFN_cuMemRetainAllocationHandle_v11000 = CUresult (*)(void*);
using PFN_cuMemGetAddressRange_v3020 = CUresult (*)(size_t*, unsigned long long*, void*);

extern PFN_cuDeviceGet_v2000 pfn_cuDeviceGet;
extern PFN_cuDeviceGetAttribute_v2000 pfn_cuDeviceGetAttribute;
extern PFN_cuCtxGetCurrent_v4000 pfn_cuCtxGetCurrent;
extern PFN_cuGetErrorString_v6000 pfn_cuGetErrorString;

extern PFN_cuMemCreate_v10020 pfn_cuMemCreate;
extern PFN_cuMemRelease_v10020 pfn_cuMemRelease;
extern PFN_cuMemAddressReserve_v10020 pfn_cuMemAddressReserve;
extern PFN_cuMemAddressFree_v10020 pfn_cuMemAddressFree;
extern PFN_cuMemMap_v10020 pfn_cuMemMap;
extern PFN_cuMemUnmap_v10020 pfn_cuMemUnmap;
extern PFN_cuMemSetAccess_v10020 pfn_cuMemSetAccess;
extern PFN_cuMemGetAllocationGranularity_v10020 pfn_cuMemGetAllocationGranularity;

extern PFN_cuMemExportToShareableHandle_v10020
    pfn_cuMemExportToShareableHandle;
extern PFN_cuMemImportFromShareableHandle_v10020
    pfn_cuMemImportFromShareableHandle;
extern PFN_cuMemGetAllocationPropertiesFromHandle_v10020
    pfn_cuMemGetAllocationPropertiesFromHandle;

extern PFN_cuMemRetainAllocationHandle_v11000 pfn_cuMemRetainAllocationHandle;
extern PFN_cuMemGetAddressRange_v3020 pfn_cuMemGetAddressRange;

#endif

} // namespace comms::pipes
