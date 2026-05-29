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
 * \file timemory/components/gperftools/types.hpp
 * \brief Declare the gperftools component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_COMPONENT(gperftools_cpu_profiler)
TIMEMORY_DECLARE_COMPONENT(gperftools_heap_profiler)
//
// for backwards-compatibility -- name change due to confusion with "gperf" (GNU perf)
//
TIMEMORY_COMPONENT_ALIAS(gperf_cpu_profiler, gperftools_cpu_profiler)
TIMEMORY_COMPONENT_ALIAS(gperf_heap_profiler, gperftools_heap_profiler)

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SET_COMPONENT_API(component::gperftools_cpu_profiler, tpls::gperftools,
                           category::external, category::timing, os::supports_unix)
TIMEMORY_SET_COMPONENT_API(component::gperftools_heap_profiler, tpls::gperftools,
                           category::external, category::memory, os::supports_unix)
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_COMPILER_INSTRUMENTATION)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::gperftools, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gperftools_heap_profiler,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gperftools_cpu_profiler,
                               false_type)
//
#else
#    if !defined(TIMEMORY_USE_GPERFTOOLS)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::gperftools, false_type)
#    endif
//
//      GPERF and gperftools_heap_profiler
//
#    if !defined(TIMEMORY_USE_GPERFTOOLS) && !defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gperftools_heap_profiler,
                               false_type)
#    endif
//
//
//      GPERF AND gperftools_cpu_profiler
//
#    if !defined(TIMEMORY_USE_GPERFTOOLS) && !defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gperftools_cpu_profiler,
                               false_type)
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::gperftools_heap_profiler,
                               true_type)
//
//--------------------------------------------------------------------------------------//
//
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(gperftools_cpu_profiler,
                                 TIMEMORY_GPERFTOOLS_CPU_PROFILER,
                                 "gperftools_cpu_profiler", "gperftools_cpu",
                                 "gperftools-cpu", "gperf_cpu_profiler", "gperf_cpu")
//
TIMEMORY_PROPERTY_SPECIALIZATION(gperftools_heap_profiler,
                                 TIMEMORY_GPERFTOOLS_HEAP_PROFILER,
                                 "gperftools_heap_profiler", "gperftools_heap",
                                 "gperftools-heap", "gperf_heap_profiler", "gperf_heap")
