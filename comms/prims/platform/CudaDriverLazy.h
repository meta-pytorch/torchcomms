// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// Lazy-loaded CUDA driver API function pointers.
//
// Uses cudaGetDriverEntryPoint (from cudart) to resolve CUDA driver API
// symbols at runtime, avoiding a link-time dependency on libcuda.so.1.
// This is the same mechanism ncclx uses in cudawrap.cc.
//
// Usage:
//   if (comms::prims::cuda_driver_lazy_init() != 0) {
//     // CUDA driver not available
//   }
//   CUresult err = comms::prims::pfn_cuMemCreate(&handle, size, &prop, 0);

// NVIDIA-only: the lazy CUDA driver API has no HIP equivalent, and every caller
// guards its use behind `#if CUDART_VERSION >= 12030` / `#ifndef
// __HIP_PLATFORM_AMD__` (the VMM / fabric paths, compiled out on AMD). Compile
// to nothing under HIP so a hipified TU that transitively includes this header
// (e.g. via Checks.h) doesn't pull real <cuda_runtime.h> alongside
// <hip/hip_runtime.h> (the ROCm vector-type alias vs CUDA struct clash).
#if !defined(__HIP_PLATFORM_AMD__)

#include <cuda.h>

#include <cudaTypedefs.h>
#include <cuda_runtime.h>

namespace comms::prims {

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

// DMA-BUF export. doca_gpu_dmabuf_fd() is a thin wrapper over this driver call,
// so the generic GPU dmabuf helper uses it directly (no DOCA context needed).
extern PFN_cuMemGetHandleForAddressRange_v11070
    pfn_cuMemGetHandleForAddressRange;

// NVSwitch multicast / multimem. The PFN typedefs were introduced in CUDA
// 12.1 (hence the `_v12010` suffix), but the multimem feature is only
// usable on CUDA 12.3+ (it needs `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`
// + functioning driver-side multicast). Gate the lazy-loaded symbols on
// 12.3 so the header / loader can't end up half-populated on an
// unsupported toolkit. The runtime feature gate in
// `MultimemHandler::selectMultimemHandleTypeImpl` uses the same threshold,
// keeping the two layers aligned. NOLINTs match the convention for the
// other pfn_cu* globals above: these are runtime-resolved driver function
// pointers that the lazy loader writes via void**, so they cannot be const.
#if CUDART_VERSION >= 12030
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
extern PFN_cuMulticastCreate_v12010 pfn_cuMulticastCreate;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
extern PFN_cuMulticastAddDevice_v12010 pfn_cuMulticastAddDevice;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
extern PFN_cuMulticastBindMem_v12010 pfn_cuMulticastBindMem;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
extern PFN_cuMulticastBindAddr_v12010 pfn_cuMulticastBindAddr;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
extern PFN_cuMulticastUnbind_v12010 pfn_cuMulticastUnbind;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
extern PFN_cuMulticastGetGranularity_v12010 pfn_cuMulticastGetGranularity;
#endif

} // namespace comms::prims

#endif // !defined(__HIP_PLATFORM_AMD__)
