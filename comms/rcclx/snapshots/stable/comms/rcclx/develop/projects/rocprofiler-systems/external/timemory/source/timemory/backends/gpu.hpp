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

#if defined(TIMEMORY_USE_CUDA)
#    include "timemory/backends/cuda.hpp"
#else
#    include "timemory/backends/hip.hpp"
#endif

#include "timemory/macros/attributes.hpp"
#include "timemory/macros/language.hpp"

namespace tim
{
namespace gpu
{
//
#if defined(TIMEMORY_USE_HIP)
//
using namespace ::tim::hip;
//
#elif defined(TIMEMORY_USE_CUDA)
//
using namespace ::tim::cuda;
//
#else
//
using namespace ::tim::hip;
//
#endif
//
TIMEMORY_DEVICE_INLINE void
memory_fence()
{
#if defined(TIMEMORY_GPU_DEVICE_COMPILE) && TIMEMORY_GPU_DEVICE_COMPILE > 0
    __threadfence();
#endif
}
//
TIMEMORY_DEVICE_INLINE void
store_fence()
{
    memory_fence();
}
//
TIMEMORY_DEVICE_INLINE void
load_fence()
{
    memory_fence();
}
//
TIMEMORY_DEVICE_INLINE void
fence()
{
    memory_fence();
}
//
}  // namespace gpu
}  // namespace tim
