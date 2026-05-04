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

#if !defined(TIMEMORY_CUPTI_SOURCE) && !defined(TIMEMORY_USE_CUPTI_EXTERN) &&            \
    !defined(TIMEMORY_CUPTI_COMPONENT_HEADER_MODE)
#    define TIMEMORY_CUPTI_COMPONENT_HEADER_MODE 1
#endif

#if !defined(TIMEMORY_CUPTI_INLINE)
#    if defined(TIMEMORY_CUPTI_COMPONENT_HEADER_MODE)
#        define TIMEMORY_CUPTI_INLINE inline
#    else
#        define TIMEMORY_CUPTI_INLINE
#    endif
#endif

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

#if defined(TIMEMORY_USE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#endif

TIMEMORY_DECLARE_COMPONENT(cupti_activity)
TIMEMORY_DECLARE_COMPONENT(cupti_counters)
TIMEMORY_DECLARE_COMPONENT(cupti_profiler)
TIMEMORY_DECLARE_COMPONENT(cupti_pcsampling)

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SET_COMPONENT_API(component::cupti_activity, tpls::nvidia, device::gpu,
                           category::external, category::timing, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::cupti_counters, tpls::nvidia, device::gpu,
                           category::external, category::hardware_counter, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::cupti_profiler, tpls::nvidia, device::gpu,
                           category::external, category::hardware_counter, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::cupti_pcsampling, tpls::nvidia, device::gpu,
                           category::external, os::agnostic)
//
//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::cupti_activity, double)
TIMEMORY_STATISTICS_TYPE(component::cupti_counters, std::vector<double>)
TIMEMORY_STATISTICS_TYPE(component::cupti_profiler, std::vector<double>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_statistics, component::cupti_pcsampling, false_type)
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_CUPTI) || !defined(TIMEMORY_USE_CUDA) ||                       \
    defined(TIMEMORY_COMPILER_INSTRUMENTATION)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_counters, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_activity, false_type)
#endif
//
#if !defined(TIMEMORY_USE_CUPTI) || !defined(TIMEMORY_USE_CUDA) ||                       \
    !defined(TIMEMORY_USE_CUPTI_NVPERF) || defined(TIMEMORY_COMPILER_INSTRUMENTATION)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_profiler, false_type)
#endif
//
#if !defined(TIMEMORY_USE_CUPTI) || !defined(TIMEMORY_USE_CUDA) ||                       \
    !defined(TIMEMORY_USE_CUPTI_PCSAMPLING) ||                                           \
    defined(TIMEMORY_COMPILER_INSTRUMENTATION)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cupti_pcsampling, false_type)
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              IS TIMING CATEGORY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::cupti_activity, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              USES TIMING UNITS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::cupti_activity, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              SECONDARY DATA
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::cupti_activity, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(secondary_data, component::cupti_counters, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::cupti_counters, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::cupti_profiler, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LABEL PRINTING
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::cupti_counters,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_label_printing, component::cupti_profiler,
                               true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::cupti_counters, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::cupti_profiler, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM SERIALIZATION
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_serialization, component::cupti_counters, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_serialization, component::cupti_profiler, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                                BASE HAS ACCUM
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::cupti_pcsampling, false_type)
//
//--------------------------------------------------------------------------------------//
//
//                                  REPORTING
//
//--------------------------------------------------------------------------------------//
//
// collection method has no use for separating value from accum
// mean values have no meaning here
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_mean, component::cupti_pcsampling, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_self, component::cupti_pcsampling, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::cupti_pcsampling, false_type)
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(cupti_activity, TIMEMORY_CUPTI_ACTIVITY,
                                 "cupti_activity", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(cupti_counters, TIMEMORY_CUPTI_COUNTERS,
                                 "cupti_counters", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(cupti_pcsampling, TIMEMORY_CUPTI_PCSAMPLING,
                                 "cupti_pcsampling", "cuda_pcsampling")
//
// TIMEMORY_PROPERTY_SPECIALIZATION(cupti_profiler, TIMEMORY_CUPTI_PROFILER,
// "cupti_profiler", "")
