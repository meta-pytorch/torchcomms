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

#include "timemory/components/roofline/extern.hpp"

// disable warning #177-D: variable "<unnamed>::timemory_storage_initializer_<N>" was
// declared but never referenced
#pragma nv_diag_suppress 177

#if defined(TIMEMORY_USE_CUPTI)
#    if defined(TIMEMORY_USE_CUDA_HALF)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    gpu_roofline_hp_flops, true,
    std::tuple<typename ::tim::component::cupti_activity::value_type,
               typename ::tim::component::cupti_counters::value_type>)
#    endif
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    gpu_roofline_sp_flops, true,
    std::tuple<typename ::tim::component::cupti_activity::value_type,
               typename ::tim::component::cupti_counters::value_type>)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    gpu_roofline_dp_flops, true,
    std::tuple<typename ::tim::component::cupti_activity::value_type,
               typename ::tim::component::cupti_counters::value_type>)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    gpu_roofline_flops, true,
    std::tuple<typename ::tim::component::cupti_activity::value_type,
               typename ::tim::component::cupti_counters::value_type>)
#endif

#if defined(TIMEMORY_USE_CUPTI)
template struct tim::component::gpu_roofline<float>;
template struct tim::component::gpu_roofline<double>;
#    if defined(TIMEMORY_USE_CUDA_HALF)
template struct tim::component::gpu_roofline<tim::cuda::fp16_t>;
template struct tim::component::gpu_roofline<tim::cuda::fp16_t, float, double>;
#    else
template struct tim::component::gpu_roofline<float, double>;
#    endif
#endif
