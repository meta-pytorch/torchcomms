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
 * \file timemory/components/craypat/types.hpp
 * \brief Declare the craypat component types
 */

#pragma once

#include "timemory/components/craypat/backends.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_COMPONENT(craypat_record)
TIMEMORY_DECLARE_COMPONENT(craypat_region)
TIMEMORY_DECLARE_COMPONENT(craypat_counters)
TIMEMORY_DECLARE_COMPONENT(craypat_heap_stats)
TIMEMORY_DECLARE_COMPONENT(craypat_flush_buffer)

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SET_COMPONENT_API(component::craypat_record, tpls::craypat, category::external,
                           os::supports_linux)
TIMEMORY_SET_COMPONENT_API(component::craypat_region, tpls::craypat, category::external,
                           category::decorator, os::supports_linux)
TIMEMORY_SET_COMPONENT_API(component::craypat_counters, tpls::craypat, category::external,
                           os::supports_linux)
TIMEMORY_SET_COMPONENT_API(component::craypat_heap_stats, tpls::craypat,
                           category::external, os::supports_linux)
TIMEMORY_SET_COMPONENT_API(component::craypat_flush_buffer, tpls::craypat,
                           category::external, os::supports_linux)
//
//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::craypat_flush_buffer, double)
TIMEMORY_STATISTICS_TYPE(component::craypat_counters, std::vector<unsigned long>)
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_CRAYPAT)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::craypat, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::craypat_record, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::craypat_region, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::craypat_counters, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::craypat_heap_stats, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::craypat_flush_buffer, false_type)
//
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              PROPERTIES
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(craypat_record, TIMEMORY_CRAYPAT_RECORD,
                                 "craypat_record", "")
TIMEMORY_PROPERTY_SPECIALIZATION(craypat_region, TIMEMORY_CRAYPAT_REGION,
                                 "craypat_region", "")
TIMEMORY_PROPERTY_SPECIALIZATION(craypat_counters, TIMEMORY_CRAYPAT_COUNTERS,
                                 "craypat_counters", "")
TIMEMORY_PROPERTY_SPECIALIZATION(craypat_heap_stats, TIMEMORY_CRAYPAT_HEAP_STATS,
                                 "craypat_heap_stats", "")
TIMEMORY_PROPERTY_SPECIALIZATION(craypat_flush_buffer, TIMEMORY_CRAYPAT_FLUSH_BUFFER,
                                 "craypat_flush_buffer", "")
//
//--------------------------------------------------------------------------------------//
//
//                              START PRIORITY
//
//--------------------------------------------------------------------------------------//
//
// start early
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, component::craypat_record,
                               priority_constant<-4>)
// start early
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, component::craypat_region,
                               priority_constant<-2>)
// start late
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, component::craypat_counters,
                               priority_constant<64>)
//
//--------------------------------------------------------------------------------------//
//
//                              STOP PRIORITY
//
//--------------------------------------------------------------------------------------//
//
// stop early
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, component::craypat_counters,
                               priority_constant<-64>)
// stop late
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, component::craypat_flush_buffer,
                               priority_constant<4>)
//
//======================================================================================//
