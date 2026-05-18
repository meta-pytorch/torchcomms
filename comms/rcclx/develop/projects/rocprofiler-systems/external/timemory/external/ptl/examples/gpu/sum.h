//
// MIT License
// Copyright (c) 2018 Jonathan R. Madsen
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
//   PTL file
//
//  Description:
//
//      Here we declare the GPU interface
//

#ifndef sum_h_
#define sum_h_

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

#ifndef DLL
#    if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#        define DLL __declspec(dllexport)
#    else
#        define DLL
#    endif
#endif

#ifdef __cplusplus
#    include <cstdio>
#    include <cstring>
#else
#    include <stdio.h>
#    include <string.h>
#endif

#if defined(PTL_USE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <vector_types.h>

#    if !defined(CUDA_CHECK_LAST_ERROR)
#        if defined(DEBUG)
#            define CUDA_CHECK_LAST_ERROR()                                              \
                {                                                                        \
                    cudaError err = cudaGetLastError();                                  \
                    if(cudaSuccess != err)                                               \
                    {                                                                    \
                        fprintf(stderr, "cudaCheckError() failed at %s@'%s':%i : %s\n",  \
                                __FUNCTION__, __FILE__, __LINE__,                        \
                                cudaGetErrorString(err));                                \
                        exit(-1);                                                        \
                    }                                                                    \
                }
#        else
#            define CUDA_CHECK_LAST_ERROR()                                              \
                {                                                                        \
                    ;                                                                    \
                }
#        endif
#    endif
#else
#    if !defined(cudaStream_t)
#        define cudaStream_t int
#    endif
#    if !defined(CUDA_CHECK_LAST_ERROR)
#        define CUDA_CHECK_LAST_ERROR()                                                  \
            {                                                                            \
                ;                                                                        \
            }
#    endif
#endif

//============================================================================//
//
//      NVTX macros
//
//============================================================================//

#if defined(PTL_USE_NVTX)
#    include <nvToolsExt.h>
#    ifndef NVTX_RANGE_PUSH
#        define NVTX_RANGE_PUSH(obj) nvtxRangePushEx(obj)
#    endif
#    ifndef NVTX_RANGE_POP
#        define NVTX_RANGE_POP(obj)                                                      \
            cudaStreamSynchronize(obj);                                                  \
            nvtxRangePop()
#    endif

void
init_nvtx();

#else
#    ifndef NVTX_RANGE_PUSH
#        define NVTX_RANGE_PUSH(obj)
#    endif
#    ifndef NVTX_RANGE_POP
#        define NVTX_RANGE_POP(obj)
#    endif

void
init_nvtx()
{}

#endif

//============================================================================//

BEGIN_EXTERN_C  // begin extern "C"

    //----------------------------------------------------------------------------//
    //  global info
    //----------------------------------------------------------------------------//

    void DLL
         cuda_device_query();
int      DLL
         cuda_device_count();

//----------------------------------------------------------------------------//
//  device-specific info
//----------------------------------------------------------------------------//

// this functions sets "thread_device()" value to device number
int DLL
    cuda_set_device(int device);

// the functions below use "thread_device()" function to get device number
int DLL
    cuda_multi_processor_count();
int DLL
    cuda_max_threads_per_block();
int DLL
    cuda_warp_size();
int DLL
    cuda_shared_memory_per_block();

//============================================================================//

END_EXTERN_C  // end extern "C"

//----------------------------------------------------------------------------//

#endif
