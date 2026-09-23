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

#if !defined(TIMEMORY_ERT_SOURCE)
#    define TIMEMORY_ERT_SOURCE 1
#endif

#if !defined(TIMEMORY_ERT_SOURCE_GPU)
#    define TIMEMORY_ERT_SOURCE_GPU 1
#endif

#if !defined(TIMEMORY_USE_HIP)
#    error "File should not be compiled without TIMEMORY_USE_HIP defined"
#else
#    define TIMEMORY_ERT_SOURCE_HIP 1
#endif

#include "timemory/ert/extern.hpp"
#include "timemory/ert/macros.hpp"

namespace tim
{
namespace ert
{
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    class counter<device::gpu, float, component::ert_timer>)
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    class counter<device::gpu, double, component::ert_timer>)
//
#if defined(TIMEMORY_USE_CUDA_HALF)
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    class counter<device::gpu, cuda::fp16_t, component::ert_timer>)
#endif
//
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    struct configuration<device::gpu, float, component::ert_timer>)
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    struct configuration<device::gpu, double, component::ert_timer>)
//
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    struct executor<device::gpu, float, component::ert_timer>)
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    struct executor<device::gpu, double, component::ert_timer>)
//
#if defined(TIMEMORY_USE_CUDA_HALF)
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    struct configuration<device::gpu, cuda::fp16_t, component::ert_timer>)
//
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_GPU(
    struct executor<device::gpu, cuda::fp16_t, component::ert_timer>)
#endif
}  // namespace ert
}  // namespace tim
