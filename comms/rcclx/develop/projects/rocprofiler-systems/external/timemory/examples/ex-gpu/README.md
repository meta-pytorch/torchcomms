# ex-gpu

## Build

See [examples](../README.md##Build). Additionally requires `-DTIMEMORY_USE_CUDA=ON` or `-DTIMEMORY_USE_HIP=ON` option
enabled in cmake.

## ex_gpu_event

This example executes a set of kernels using either CUDA or HIP and measures the kernel execution time via either
`cudaEvent_t` or `hipEvent_t`. Aliases to the different backends are defined as follow:

```cpp
#if defined(TIMEMORY_USE_CUDA)
using gpu_marker = tim::component::nvtx_marker;
using gpu_event  = tim::component::cuda_event;
#else
using gpu_marker = tim::component::roctx_marker;
using gpu_event  = tim::component::hip_event;
#endif
```

## Kernel Instrumentation

This example contains two example implementations of using timemory to instrument GPU kernels.
In the each example, the kernels have an extra parameter of type `device_data` which is defined
within the `gpu_device_timer` component. Device timing is initialized via calling
`gpu_device_timer::start(device::gpu{}, size_t threads_per_block)`. As usual, you can pass these
parameters through the component bundle start member function. The `gpu_device_timer` start member function
allocates memory on the GPU for the `device_data` object to update. Only when `stop()` is called
on `gpu_device_timer` (or it's bundle) will the data be copied back over to the host and
updated.

### Overview

```cpp
namespace comp = tim::component;

using bundle_t = tim::component_tuple<comp::wall_clock, comp::gpu_device_timer>;

__global__
void bar(...,
         comp::gpu_device_timer::device_data _timer)
{
    _timer.start();
    // ... etc. ...
    _timer.stop();
}

// "n" is the number of blocks. Threads per block is 1024 here but could also
// be variable
void foo(size_t n)
{
    bundle_t _obj{ __FUNCTION__ };

    _obj.start(tim::device::gpu{}, 1024);

    // get_reference<T>() is used here to guarantee that gpu_device_timer is available
    bar<<<n, 1024>>>(...,
                     _obj.get_reference<comp::gpu_device_timer>().get_device_data());

    _obj.stop();
}
```

`device_data` uses `clock64()` on the device to collect timestamps.
In both of the kernel instrumentation examples, it's definition is similar to the following:

```cpp
struct device_data
{
    __device__ void start()
    {
        if(!m_data)
            return;
        m_buff = clock64();
    }

    __device__ void stop()
    {
        if(!m_data)
            return;
        auto _idx = (blockDim.x * blockDim.y * blockIdx.z) +
                    (blockDim.x * blockIdx.y) +
                    threadIdx.x;
        atomicAdd(&(m_data[_idx]), static_cast<int>(clock64() - m_buff));
    }

    long long int m_buff = 0;
    unsigned int* m_data = nullptr;
};
```

### ex_kernel_instrument

This is one theoretical implementation of using timemory to instrument GPU kernels.
In the `gpu_device_timer` component implementation, the data is directly stored in the
`gpu_device_timer` instance.

### ex_kernel_instrument_v2

This is another theoretical implementation of using timemory to instrument GPU kernels.
The primary difference between this instance and [ex_kernel_instrument](#ex_kernel_instrument)
is that the data is not stored directly in `gpu_device_timer` but instead, it uses the
`data_tracker` to improve the data storage and output capabilities.

### ex_kernel_instrument_v3

This is the same as [ex_kernel_instrument_v2](#ex_kernel_instrument_v2) but does not required
modifying the kernel parameters.

Version 3 ([ex_kernel_instrument_v3](#ex_kernel_instrument_v3)) is the recommended style.
