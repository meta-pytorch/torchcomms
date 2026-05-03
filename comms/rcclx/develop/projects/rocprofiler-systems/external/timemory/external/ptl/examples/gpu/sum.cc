//
// MIT License
// Copyright (c) 2019 Jonathan R. Madsen
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ---------------------------------------------------------------
//
//   PTL implementation file
//
//  Description:
//
//      Here we copy the memory to GPU and call the GPU kernels
//

//============================================================================//
// C++

// includes all C, CUDA, and C++ header files

#include "sum.hh"
#include "ThreadPool.hh"
#include "profiler.hh"

typedef std::vector<float>   farray_t;
typedef std::vector<int64_t> iarray_t;

#define PRINT_HERE(extra)                                                                \
    printf("[%lu]> %s@'%s':%i %s\n", ThreadPool::get_this_thread_id(), __FUNCTION__,     \
           __FILE__, __LINE__, extra)

//============================================================================//

#if defined(PTL_USE_NVTX)

static nvtxEventAttributes_t nvtx_thrust_sum;
static nvtxEventAttributes_t nvtx_cuda_sum;

//----------------------------------------------------------------------------//

void
init_nvtx()
{
    static bool first = true;
    if(!first)
        return;
    first = false;

    nvtx_thrust_sum.version       = NVTX_VERSION;
    nvtx_thrust_sum.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_thrust_sum.colorType     = NVTX_COLOR_ARGB;
    nvtx_thrust_sum.color         = 0xff0000ff; /* blue? */
    nvtx_thrust_sum.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_thrust_sum.message.ascii = "calc_coords";

    nvtx_cuda_sum.version       = NVTX_VERSION;
    nvtx_cuda_sum.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    nvtx_cuda_sum.colorType     = NVTX_COLOR_ARGB;
    nvtx_cuda_sum.color         = 0xffff0000; /* red */
    nvtx_cuda_sum.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    nvtx_cuda_sum.message.ascii = "sort_intersections";
}

#endif

//============================================================================//

int
cuda_device_count()
{
    int         deviceCount = 0;
    cudaError_t error_id    = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess)
        return 0;

    return deviceCount;
}

//============================================================================//

void
cuda_device_query()
{
    static bool first = true;
    if(first)
        first = false;
    else
        return;

    int         deviceCount    = 0;
    int         driverVersion  = 0;
    int         runtimeVersion = 0;
    cudaError_t error_id       = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned error code %d\n--> %s\n",
               static_cast<int>(error_id), cudaGetErrorString(error_id));

        if(deviceCount > 0)
        {
            cudaSetDevice(0);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            printf("\nDevice %d: \"%s\"\n", 0, deviceProp.name);

            // Console log
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            printf("  CUDA Driver Version / Runtime Version          %d.%d / "
                   "%d.%d\n",
                   driverVersion / 1000, (driverVersion % 100) / 10,
                   runtimeVersion / 1000, (runtimeVersion % 100) / 10);
            printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
                   deviceProp.major, deviceProp.minor);
        }

        return;
    }

    if(deviceCount == 0)
        printf("No available CUDA device(s) detected\n");
    else
        printf("Detected %d CUDA capable devices\n", deviceCount);

    for(int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        // This only available in CUDA 4.0-4.2 (but these were only exposed in
        // the CUDA Driver API)
        int memoryClock;
        int memBusWidth;
        int L2CacheSize;

        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);

        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);

        char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(msg, sizeof(msg),
                  "  Total amount of global memory:                 %.0f MBytes "
                  "(%llu bytes)\n",
                  static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                  (unsigned long long) deviceProp.totalGlobalMem);
#else
        snprintf(msg, sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes "
                 "(%llu bytes)\n",
                 static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                 (unsigned long long) deviceProp.totalGlobalMem);
#endif
        printf("%s", msg);

        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f "
               "GHz)\n",
               deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",
               deviceProp.memoryBusWidth);

        if(deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n",
                   deviceProp.l2CacheSize);
        }

#else
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               memoryClock * 1e-3f);
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                              dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if(L2CacheSize)
            printf("  L2 Cache Size:                                 %d bytes\n",
                   L2CacheSize);
#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
               "%d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
               deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
               deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d "
               "layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
               "layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %lu bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n",
               deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n",
               deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy "
               "engine(s)\n",
               (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n",
               deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
               deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
               deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
               deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
               deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                    : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n",
               deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               deviceProp.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               deviceProp.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char* sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with "
            "device "
            "simultaneously)",
            "Exclusive (only one host thread in one process is able to use "
            "::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this "
            "device)",
            "Exclusive Process (many threads in one process is able to use "
            "::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    printf("\n\n");
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
}

//============================================================================//

int
cuda_set_device(int device)
{
    int deviceCount = cuda_device_count();
    if(deviceCount == 0)
        return -1;

    // don't set to higher than number of devices
    device = device % deviceCount;
    // update thread-static variable
    this_thread_device() = device;
    // actually set the device
    cudaSetDevice(device);
    // return the modulus
    return device;
}

//============================================================================//

int
cuda_multi_processor_count()
{
    if(cuda_device_count() == 0)
        return 0;

    // keep from querying device
    static thread_local cuda_device_info<int>* _instance = new cuda_device_info<int>();
    // use the thread assigned devices
    int device = this_thread_device();

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return ((*_instance)[device] = deviceProp.multiProcessorCount);
}

//============================================================================//

int
cuda_max_threads_per_block()
{
    if(cuda_device_count() == 0)
        return 0;

    // keep from querying device
    static thread_local cuda_device_info<int>* _instance = new cuda_device_info<int>();
    // use the thread assigned devices
    int device = this_thread_device();

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return ((*_instance)[device] = deviceProp.maxThreadsPerBlock);
}

//============================================================================//

int
cuda_warp_size()
{
    if(cuda_device_count() == 0)
        return 0;

    // keep from querying device
    static thread_local cuda_device_info<int>* _instance = new cuda_device_info<int>();
    // use the thread assigned devices
    int device = this_thread_device();

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return ((*_instance)[device] = deviceProp.warpSize);
}

//============================================================================//

int
cuda_shared_memory_per_block()
{
    if(cuda_device_count() == 0)
        return 0;

    // keep from querying device
    static thread_local cuda_device_info<int>* _instance = new cuda_device_info<int>();
    // use the thread assigned devices
    int device = this_thread_device();

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return ((*_instance)[device] = deviceProp.sharedMemPerBlock);
}

//============================================================================//

float
compute_sum(farray_t& cpu_data)
{
    NVTX_RANGE_PUSH(&nvtx_cuda_sum);

    static thread_local uint64_t      tid    = ThreadPool::get_this_thread_id();
    static thread_local cudaStream_t& stream = cuda_streams::instance()->get(tid);

    TIMEMORY_AUTO_TIMER("[cuda]");

    float _sum = 0.0f;

    uintmax_t grainsize = static_cast<uintmax_t>(cuda_max_threads_per_block());
    float*    buffer    = gpu_malloc<float>(grainsize);

    if(cpu_data.size() > grainsize)
    {
        uintmax_t nitr = cpu_data.size() / grainsize;
        uintmax_t nmod = cpu_data.size() % grainsize;
        if(nmod > 0)
            ++nitr;
        uintmax_t nrem = (nmod > 0) ? nmod : grainsize;

        aligned_ptr<float> gpu_data = aligned_gpu_malloc<float, 512>(grainsize);

        auto offset = 0;
        for(uintmax_t i = 0; i < nitr; ++i)
        {
            uintmax_t size = grainsize;
            if(i + 1 == nitr)
                size = nrem;

            async_gpu_memcpy(gpu_data.ptr, cpu_data.data() + offset, size, stream);
            gpu_data.size = size;

            float _tmp_sum = compute_sum_host(gpu_data, stream, false, buffer);

            _sum += _tmp_sum;
            offset += size;
        }

        gpu_data.free();
    }
    else
    {
        aligned_ptr<float> gpu_data = aligned_async_malloc_and_memcpy<float, 512>(
            cpu_data.data(), cpu_data.size(), stream);
        float _tmp_sum = compute_sum_host(gpu_data, stream, false, buffer);

        _sum += _tmp_sum;

        gpu_data.free();
    }

    cudaFree(buffer);

    NVTX_RANGE_POP(stream);

    return _sum;
}

//============================================================================//

float
compute_sum(thrust::host_vector<float>& cpu_data)
{
    NVTX_RANGE_PUSH(&nvtx_thrust_sum);

    static thread_local uint64_t      tid    = ThreadPool::get_this_thread_id();
    static thread_local cudaStream_t& stream = cuda_streams::instance()->get(tid);

    TIMEMORY_AUTO_TIMER("[thrust]");

    float*             buffer   = nullptr;
    aligned_ptr<float> gpu_data = aligned_async_malloc_and_memcpy<float, 512>(
        cpu_data.data(), cpu_data.size(), stream);
    float _sum = compute_sum_host(gpu_data, stream, true, buffer);

    NVTX_RANGE_POP(stream);

    return _sum;
}

//============================================================================//

uint64_t
run_gpu(uint64_t n)
{
    cuda_device_query();
    set_this_thread_device();

    // constants
    const float    factor  = 1.0f;
    const float    epsilon = std::numeric_limits<float>::epsilon();
    const uint64_t scale   = 1;
    const uint64_t size    = scale * n;
    // const solution
    const float real_sum = factor * size;

    auto check = [&](const float& calc_sum) {
        uint64_t _ret = (abs(real_sum - calc_sum) < epsilon) ? 1 : 0;
        if(_ret == 0)
            printf("[%lu] > incorrect GPU summation. real = %g, calculated = %g\n",
                   ThreadPool::get_this_thread_id(), real_sum, calc_sum);
        return _ret;
    };

    uint64_t ret = 0;

    {
        // data
        // farray_t data(size, factor);
        // computed results
        // float calc_sum = compute_sum(data);
        // check if same
        // ret += check(calc_sum);
        ret += 1;
        // PRINT_HERE(std::string(std::string("calc : ") +
        // std::to_string(calc_sum) +
        //                       std::string(", ") +
        //                       std::string("real : ") +
        //                       std::to_string(real_sum)).c_str());
    }

    {
        // data
        thrust::host_vector<float> thrust_data(size, factor);
        // solution
        float real_sum = factor * size;
        // computed results
        float calc_sum = compute_sum(thrust_data);
        // PRINT_HERE(std::string(std::string("calc : ") +
        // std::to_string(calc_sum) +
        //                       std::string(", ") +
        //                       std::string("real : ") +
        //                       std::to_string(real_sum)).c_str());

        // check if same
        ret += check(calc_sum);
    }

    return (ret < 2) ? 0 : 1;
}
//============================================================================//
