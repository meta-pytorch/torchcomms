// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#if defined(TIMEMORY_USE_HIP)
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

#if defined(TIMEMORY_HIPCC)
#    include <hip/amd_detail/amd_device_functions.h>
#endif

namespace tim
{
namespace hip
{
//
#if defined(TIMEMORY_USE_HIP)
// define some types for when HIP is enabled
using stream_t = hipStream_t;
using event_t  = hipEvent_t;
using error_t  = hipError_t;
using memcpy_t = hipMemcpyKind;
// define some values for when HIP is enabled
static constexpr stream_t               default_stream_v   = nullptr;
static const decltype(hipSuccess)       success_v          = hipSuccess;
static const decltype(hipErrorNotReady) err_not_ready_v    = hipErrorNotReady;
static const hipMemcpyKind              host_to_host_v     = hipMemcpyHostToHost;
static const hipMemcpyKind              host_to_device_v   = hipMemcpyHostToDevice;
static const hipMemcpyKind              device_to_host_v   = hipMemcpyDeviceToHost;
static const hipMemcpyKind              device_to_device_v = hipMemcpyDeviceToDevice;
//
#else
//
// define some types for when HIP is disabled
//
using stream_t = void*;
using event_t  = int;
using error_t  = int;
using memcpy_t = int;
//
// define some values for when HIP is disabled
static constexpr stream_t default_stream_v   = nullptr;
static const int          success_v          = 0;
static const int          err_not_ready_v    = 0;
static const int          host_to_host_v     = 0;
static const int          host_to_device_v   = 1;
static const int          device_to_host_v   = 2;
static const int          device_to_device_v = 3;
#endif
//
}  // namespace hip
}  // namespace tim
