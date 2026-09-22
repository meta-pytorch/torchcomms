#include <iostream>
#include <hip/hip_runtime.h>

extern "C" int launch_kernel(int device_id);

#define HIP_CALL(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            std::cerr << hipGetErrorString(err) << std::endl;                                      \
            return(-1);                                                                            \
        }                                                                                          \
    } while(0)

__global__ void
kernelA(int x, int y)
{
    volatile int i, t;
    for(i=0; i<1000000; i++){
        t = 173/x;
        x = t + y;
    }
}

template <typename T>
__global__ void
kernelC(T* C_d, const T* A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = offset; i < N; i += stride)
    {
        C_d[i] = A_d[i] * A_d[i];
    }
}

int launch_kernel(int device_id) {
    const int NUM_LAUNCH = 1;

    HIP_CALL(hipSetDevice(device_id));

    for(int i = 0; i < NUM_LAUNCH; i++)
    {
        hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0, 1, 2);
    }

    HIP_CALL(hipDeviceSynchronize());

    return 0;
}
