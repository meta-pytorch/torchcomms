#pragma once

#ifdef __CUDA_ARCH__
#define DEVICE_ATTRIBUTE __device__ __forceinline__
#else

#if defined(__HIP_PLATFORM_AMD__)
#define DEVICE_ATTRIBUTE __device__ __forceinline__
#else
#define DEVICE_ATTRIBUTE inline
#endif

#endif
