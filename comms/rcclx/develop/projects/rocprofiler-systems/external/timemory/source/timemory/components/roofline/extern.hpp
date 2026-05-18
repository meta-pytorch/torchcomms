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

/**
 * \file timemory/components/roofline/extern.hpp
 * \brief Include the extern declarations for roofline components
 */

#pragma once

#include "timemory/components/extern/common.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/roofline/components.hpp"

#if defined(TIMEMORY_USE_PAPI)
#    include "timemory/components/papi/components.hpp"
#    include "timemory/components/papi/extern.hpp"

TIMEMORY_DECLARE_EXTERN_COMPONENT(cpu_roofline_sp_flops, true,
                                  std::pair<std::vector<long long>, double>)
TIMEMORY_DECLARE_EXTERN_COMPONENT(cpu_roofline_dp_flops, true,
                                  std::pair<std::vector<long long>, double>)
TIMEMORY_DECLARE_EXTERN_COMPONENT(cpu_roofline_flops, true,
                                  std::pair<std::vector<long long>, double>)
extern template struct tim::component::cpu_roofline<float>;
extern template struct tim::component::cpu_roofline<double>;
extern template struct tim::component::cpu_roofline<float, double>;
#endif

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/components/cupti/components.hpp"
#    include "timemory/components/cupti/extern.hpp"

#    if defined(TIMEMORY_USE_CUDA_HALF)
TIMEMORY_DECLARE_EXTERN_COMPONENT(
    gpu_roofline_hp_flops, true,
    std::tuple<typename ::tim::component::cupti_activity::value_type,
               typename ::tim::component::cupti_counters::value_type>)
#    endif
TIMEMORY_DECLARE_EXTERN_COMPONENT(
    gpu_roofline_sp_flops, true,
    std::tuple<typename ::tim::component::cupti_activity::value_type,
               typename ::tim::component::cupti_counters::value_type>)
TIMEMORY_DECLARE_EXTERN_COMPONENT(
    gpu_roofline_dp_flops, true,
    std::tuple<typename ::tim::component::cupti_activity::value_type,
               typename ::tim::component::cupti_counters::value_type>)
TIMEMORY_DECLARE_EXTERN_COMPONENT(
    gpu_roofline_flops, true,
    std::tuple<typename ::tim::component::cupti_activity::value_type,
               typename ::tim::component::cupti_counters::value_type>)
#endif

#if defined(TIMEMORY_USE_CUPTI)
extern template struct tim::component::gpu_roofline<float>;
extern template struct tim::component::gpu_roofline<double>;
#    if defined(TIMEMORY_USE_CUDA_HALF)
extern template struct tim::component::gpu_roofline<tim::cuda::fp16_t>;
extern template struct tim::component::gpu_roofline<tim::cuda::fp16_t, float, double>;
#    else
extern template struct tim::component::gpu_roofline<float, double>;
#    endif
#endif
