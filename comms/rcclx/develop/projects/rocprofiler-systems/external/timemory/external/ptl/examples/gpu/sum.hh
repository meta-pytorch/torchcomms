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
//  ---------------------------------------------------------------
//   PTL header
//
//

#pragma once

#ifdef __cplusplus
#    ifndef BEGIN_EXTERN_C
#        define BEGIN_EXTERN_C                                                           \
            extern "C"                                                                   \
            {
#    endif
#    ifndef END_EXTERN_C
#        define END_EXTERN_C }
#    endif
#else
#    ifndef BEGIN_EXTERN_C
#        define BEGIN_EXTERN_C
#    endif
#    ifndef END_EXTERN_C
#        define END_EXTERN_C
#    endif
#endif

//============================================================================//
//  C headers

BEGIN_EXTERN_C
#include "sum.h"
END_EXTERN_C

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <ctime>

//============================================================================//
//  C++ headers

#include <atomic>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "PTL/ThreadData.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/Threading.hh"
#include "PTL/Utility.hh"

#if defined(PTL_USE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <thrust/device_vector.h>
#    include <thrust/execution_policy.h>
#    include <thrust/functional.h>
#    include <thrust/host_vector.h>
#    include <thrust/reduce.h>
#    include <thrust/system/cpp/execution_policy.h>
#    include <thrust/system/cuda/execution_policy.h>
#    include <thrust/system/omp/execution_policy.h>
#    include <thrust/system/tbb/execution_policy.h>
#    include <thrust/transform.h>
#    include <vector_types.h>
#endif

//============================================================================//

template <typename _Tp>
using cuda_device_info = std::unordered_map<int, _Tp>;

//============================================================================//

inline int&
this_thread_device()
{
#if defined(PTL_USE_CUDA)
    static std::atomic<int> _ntid(0);
    static thread_local int _instance =
        (cuda_device_count() > 0) ? ((_ntid++) % cuda_device_count()) : 0;
    return _instance;
#else
    static thread_local int _instance = 0;
    return _instance;
#endif
}

//============================================================================//

inline void
set_this_thread_device()
{
#if defined(PTL_USE_CUDA)
    cuda_set_device(this_thread_device());
#endif
}

//============================================================================//
//  CUDA only
#if defined(PTL_USE_CUDA)

//============================================================================//
//  CUDA headers

#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <vector_types.h>

//============================================================================//
//
//      CUDA streams
//
//============================================================================//

inline cudaStream_t*
create_streams(const uint64_t nstreams)
{
    cudaStream_t* streams = new cudaStream_t[nstreams];
    for(uint64_t i = 0; i < nstreams; ++i)
        cudaStreamCreate(&streams[i]);
    return streams;
}

//============================================================================//

inline void
destroy_streams(cudaStream_t* streams, const uint64_t nstreams)
{
    for(uint64_t i = 0; i < nstreams; ++i)
        cudaStreamDestroy(streams[i]);
    delete[] streams;
}

//============================================================================//
//
//      struct for aligned pointers
//
//============================================================================//

template <typename _Tp, uintmax_t _AlignWidth = 512>
struct aligned_pointer
{
    typedef aligned_pointer<_Tp, _AlignWidth> this_type;
    typedef _Tp                               value_type;
    typedef _Tp*                              pointer_type;

    _Tp*      ptr;
    uintmax_t size;
    uintmax_t padding;
    uintmax_t storage_size;

    aligned_pointer(const uintmax_t _size)
    :  // pointer to data
        ptr(nullptr)
    ,
        // size of data
        size(_size)
    ,
        // extra padding aligning to "_AlignWidth" byte width
        padding((_size % _AlignWidth == 0) ? 0 : (_AlignWidth - (size % _AlignWidth)))
    ,
        // size of allocation
        storage_size(size + padding)
    {}

    this_type& allocate()
    {
        // allocate array aligned to "_AlignWidth"
        cudaMalloc((void**) &ptr, storage_size * sizeof(_Tp));
        return *this;
    }

    void free() { cudaFree((void*) ptr); }
};

template <typename _Tp>
using aligned_ptr = aligned_pointer<_Tp, 512>;

//============================================================================//
//
//      Non-Asynchronous Routines
//
//============================================================================//

template <typename _Tp>
_Tp*
gpu_malloc(uintmax_t _size)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, _size * sizeof(_Tp));
    return _gpu;
}

//----------------------------------------------------------------------------//

template <typename _Tp, uintmax_t _Align = 512>
aligned_pointer<_Tp, _Align>
aligned_gpu_malloc(uintmax_t _size)
{
    return aligned_pointer<_Tp, _Align>(_size).allocate();
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
gpu_memcpy(_Tp* _gpu, const _Tp* _cpu, uintmax_t _size)
{
    cudaMemcpy(_gpu, _cpu, _size * sizeof(_Tp), cudaMemcpyHostToDevice);
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
cpu_memcpy(const _Tp* _gpu, _Tp* _cpu, uintmax_t _size)
{
    cudaMemcpy(_cpu, _gpu, _size * sizeof(_Tp), cudaMemcpyDeviceToHost);
}

//----------------------------------------------------------------------------//

template <typename _Tp>
_Tp*
malloc_and_memcpy(const _Tp* _cpu, uintmax_t _size)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, _size * sizeof(_Tp));
    cudaMemcpy(_gpu, _cpu, _size * sizeof(_Tp), cudaMemcpyHostToDevice);
    return _gpu;
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
memcpy_and_free(_Tp* _cpu, _Tp* _gpu, uintmax_t _size)
{
    cudaMemcpy(_cpu, _gpu, _size * sizeof(_Tp), cudaMemcpyDeviceToHost);
    cudaFree(_gpu);
}

//============================================================================//
//
//      Asynchronous Routines
//
//============================================================================//

template <typename _Tp>
void
async_gpu_memcpy(_Tp* _gpu, const _Tp* _cpu, uintmax_t _size, cudaStream_t _stream)
{
    cudaMemcpyAsync(_gpu, _cpu, _size * sizeof(_Tp), cudaMemcpyHostToDevice, _stream);
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
async_cpu_memcpy(const _Tp* _gpu, _Tp* _cpu, uintmax_t _size, cudaStream_t _stream)
{
    cudaMemcpyAsync(_cpu, _gpu, _size * sizeof(_Tp), cudaMemcpyDeviceToHost, _stream);
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
async_gpu_memset(_Tp* _gpu, uintmax_t _size, cudaStream_t _stream)
{
    cudaMemsetAsync(_gpu, 0, _size * sizeof(_Tp), _stream);
}

//----------------------------------------------------------------------------//

template <typename _Tp>
_Tp*
async_malloc_and_memset(uintmax_t _size, cudaStream_t _stream)
{
    _Tp* _gpu = gpu_malloc<_Tp>(_size);
    async_gpu_memset(_gpu, _size, _stream);
    return _gpu;
}

//----------------------------------------------------------------------------//

template <typename _Tp>
_Tp*
async_malloc_and_memcpy(const _Tp* _cpu, uintmax_t _size, cudaStream_t _stream)
{
    _Tp* _gpu;
    cudaMalloc((void**) &_gpu, _size * sizeof(_Tp));
    cudaMemcpyAsync(_gpu, _cpu, _size * sizeof(_Tp), cudaMemcpyHostToDevice, _stream);
    return _gpu;
}

//----------------------------------------------------------------------------//

template <typename _Tp, uintmax_t _Align = 512>
aligned_pointer<_Tp, _Align>
aligned_async_malloc_and_memcpy(const _Tp* _cpu, uintmax_t _size, cudaStream_t _stream)
{
    aligned_pointer<_Tp, _Align> _gpu(_size);

    // run cudaMalloc
    _gpu.allocate();

    // copy "_size" values from CPU to GPU
    cudaMemcpyAsync(_gpu.ptr, _cpu, _size * sizeof(_Tp), cudaMemcpyHostToDevice, _stream);

    // zero initialize extra padding
    cudaMemsetAsync(_gpu.ptr + _size, 0, _gpu.padding * sizeof(_Tp), _stream);

    // return pointer
    return _gpu;
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
async_memcpy_and_free(_Tp* _cpu, _Tp* _gpu, uintmax_t size, cudaStream_t stream)
{
    cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
    cudaFree(_gpu);
}

//============================================================================//

#else  // not defined(PTL_USE_CUDA)

#    if !defined(cudaStream_t)
#        define cudaStream_t int
#    endif

//============================================================================//
template <typename _Tp>
_Tp*
gpu_malloc(uintmax_t size)
{
    return nullptr;
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
gpu_memcpy(_Tp*, const _Tp*, uintmax_t, cudaStream_t)
{}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
cpu_memcpy(const _Tp*, _Tp*, uintmax_t, cudaStream_t)
{}
//----------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_memcpy(const _Tp*, uintmax_t)
{
    return nullptr;
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
memcpy_and_free(_Tp*, _Tp*, uintmax_t)
{}
//----------------------------------------------------------------------------//
template <typename _Tp>
_Tp*
malloc_and_async_memcpy(const _Tp*, uintmax_t, cudaStream_t)
{
    return nullptr;
}
//----------------------------------------------------------------------------//
template <typename _Tp>
void
async_memcpy_and_free(_Tp*, _Tp*, uintmax_t, cudaStream_t)
{}
//----------------------------------------------------------------------------//
inline cudaStream_t*
create_streams(const int)
{
    return nullptr;
}
//----------------------------------------------------------------------------//
inline void
destroy_streams(cudaStream_t*, const int)
{}
//============================================================================//

#endif  // if defined(PTL_USE_CUDA)

//============================================================================//

class cuda_streams
{
public:
    cuda_streams(uint64_t nstreams = 64)
    : m_nstreams(nstreams)
    , m_streams(create_streams(nstreams))
    {}

    ~cuda_streams() { destroy_streams(m_streams, m_nstreams); }

    static cuda_streams*& instance()
    {
        static cuda_streams* _instance = new cuda_streams();
        return _instance;
    }

    // operator cudaStream_t*() { return m_streams; }

    cudaStream_t& get(uint64_t i) { return m_streams[i]; }

    uint64_t size() const { return m_nstreams; }
    uint64_t num_streams() const { return m_nstreams; }

private:
    uint64_t      m_nstreams;
    cudaStream_t* m_streams;
};

//============================================================================//

template <typename _Tp>
using aligned_ptr = aligned_pointer<_Tp, 512>;
typedef std::vector<float> farray_t;

//----------------------------------------------------------------------------//

float
compute_sum(farray_t& data);

//----------------------------------------------------------------------------//

float
compute_sum_host(aligned_ptr<float>& data, cudaStream_t stream, bool with_thrust,
                 float* buffer);

//----------------------------------------------------------------------------//

uint64_t
run_gpu(uint64_t n);

//============================================================================//
