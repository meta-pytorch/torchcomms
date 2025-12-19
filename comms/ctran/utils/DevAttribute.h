#pragma once

#ifdef __CUDA_ARCH__
#define DEVICE_ATTRIBUTE __device__ __forceinline__
#else
#if defined(__HIPCC__)
// For HIP compilation: mark functions as both __host__ and __device__
// so they can be called from both host (.cc) and device (.cu) code
#define DEVICE_ATTRIBUTE __host__ __device__ __forceinline__
#else
// For regular C++ compilation (host-only)
#define DEVICE_ATTRIBUTE inline
#endif
#endif
