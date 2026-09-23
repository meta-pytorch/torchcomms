// MIT License
//
// Copyright (c) 2019 Jonathan R. Madsen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#pragma once

//======================================================================================//

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

//--------------------------------------------------------------------------------------//

#ifndef DLL
#    ifdef WIN32
#        define DLL __declspec(dllexport)
#    else
#        define DLL
#    endif
#endif

//--------------------------------------------------------------------------------------//

#ifdef __cplusplus
#    include <cstdio>
#    include <cstring>
#else
#    include <stdio.h>
#    include <string.h>
#endif

//======================================================================================//

// Define C++11
#ifndef CXX11
#    if __cplusplus > 199711L  // C++11
#        define CXX11
#    endif
#endif

//======================================================================================//

// Define C++14
#ifndef CXX14
#    if __cplusplus > 201103L  // C++14
#        define CXX14
#    endif
#endif

//======================================================================================//

// Define C++17
#ifndef CXX17
#    if __cplusplus > 201402L  // C++17
#        define CXX17
#    endif
#endif

//======================================================================================//

#if !defined(PRAGMA_SIMD)
#    define PRAGMA_SIMD _Pragma("omp simd")
#endif

//======================================================================================//

#if !defined(PRAGMA_SIMD_REDUCTION)
#    define PRAGMA(statement) _Pragma(statement)
#endif

//======================================================================================//

#define _USE_MATH_DEFINES
#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327
#endif

//======================================================================================//
//  C headers

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

//======================================================================================//
//  C++ headers

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "PTL/AutoLock.hh"
#include "PTL/TBBTaskGroup.hh"
#include "PTL/Task.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
#include "PTL/ThreadData.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/Threading.hh"
#include "PTL/TiMemory.hh"
#include "PTL/Types.hh"
#include "PTL/Utility.hh"

using namespace PTL;

//--------------------------------------------------------------------------------------//

#if defined(PTL_USE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <npp.h>
#    include <nppi.h>
#    include <vector_types.h>
#else
#    if !defined(cudaStream_t)
#        define cudaStream_t int
#    endif
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

//======================================================================================//
// this function is used by a macro -- returns a unique identifier to the thread
inline uintmax_t
get_this_thread_id()
{
    return ThreadPool::get_this_thread_id();
}

//======================================================================================//
// short hand for static_cast
#if !defined(scast)
#    define scast static_cast
#endif

//======================================================================================//
// get the number of hardware threads
#if !defined(HW_CONCURRENCY)
#    define HW_CONCURRENCY std::thread::hardware_concurrency()
#endif

//======================================================================================//
// debugging
#if !defined(PRINT_HERE)
#    define PRINT_HERE(extra)                                                            \
        printf("[%lu]> %s@'%s':%i %s\n", get_this_thread_id(), __FUNCTION__, __FILE__,   \
               __LINE__, extra)
#endif

//======================================================================================//
// start a timer
#if !defined(START_TIMER)
#    define START_TIMER(var) auto var = std::chrono::system_clock::now()
#endif

//======================================================================================//
// report a timer
#if !defined(REPORT_TIMER)
#    define REPORT_TIMER(start_time, note, counter, total_count)                         \
        {                                                                                \
            auto                          end_time = std::chrono::system_clock::now();   \
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;       \
            printf("[%li]> %-16s :: %3i of %3i... %5.2f seconds\n",                      \
                   get_this_thread_id(), note, counter, total_count,                     \
                   elapsed_seconds.count());                                             \
        }
#endif

//======================================================================================//

#if !defined(PTL_USE_CUDA)
#    if !defined(__global__)
#        define __global__
#    endif
#    if !defined(__device__)
#        define __device__
#    endif
#endif

#if defined(__NVCC__) && defined(PTL_USE_CUDA)

//--------------------------------------------------------------------------------------//
// this is always defined, even in release mode
#    if !defined(CUDA_CHECK_CALL)
#        define CUDA_CHECK_CALL(err)                                                     \
            {                                                                            \
                if(cudaSuccess != err)                                                   \
                {                                                                        \
                    std::stringstream ss;                                                \
                    ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'"          \
                       << __FILE__ << "':" << __LINE__ << " : "                          \
                       << cudaGetErrorString(err);                                       \
                    fprintf(stderr, "%s\n", ss.str().c_str());                           \
                    printf("%s\n", ss.str().c_str());                                    \
                    throw std::runtime_error(ss.str().c_str());                          \
                }                                                                        \
            }
#    endif
// this is only defined in debug mode
#    if !defined(CUDA_CHECK_LAST_ERROR)
#        if defined(DEBUG)
#            define CUDA_CHECK_LAST_ERROR()                                              \
                {                                                                        \
                    cudaStreamSynchronize(0);                                            \
                    cudaError err = cudaGetLastError();                                  \
                    if(cudaSuccess != err)                                               \
                    {                                                                    \
                        fprintf(stderr, "cudaCheckError() failed at %s@'%s':%i : %s\n",  \
                                __FUNCTION__, __FILE__, __LINE__,                        \
                                cudaGetErrorString(err));                                \
                        printf("cudaCheckError() failed at %s@'%s':%i : %s\n",           \
                               __FUNCTION__, __FILE__, __LINE__,                         \
                               cudaGetErrorString(err));                                 \
                        exit(1);                                                         \
                    }                                                                    \
                }
#        else
#            define CUDA_CHECK_LAST_ERROR()                                              \
                {                                                                        \
                    ;                                                                    \
                }
#        endif
#    endif

#endif  // NVCC and PTL_USE_CUDA

//======================================================================================//
