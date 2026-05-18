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
 * \file timemory/components/hip/types.hpp
 * \brief Declare the hip component types
 */

#pragma once

#include "timemory/backends/hip.hpp"
#include "timemory/backends/roctx.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

#if defined(TIMEMORY_PYBIND11_SOURCE)
namespace pybind11
{
class object;
}
#endif

TIMEMORY_DECLARE_COMPONENT(hip_event)
TIMEMORY_DECLARE_COMPONENT(roctx_marker)
TIMEMORY_COMPONENT_ALIAS(hip_roctx, roctx_marker)

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SET_COMPONENT_API(component::hip_event, tpls::rocm, device::gpu,
                           category::external, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::roctx_marker, tpls::rocm, category::external,
                           category::decorator, os::agnostic)
//
//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::hip_event, float)
//
//--------------------------------------------------------------------------------------//
//
//                              IS TIMING CATEGORY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::hip_event, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              USES TIMING UNITS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::hip_event, true_type)
//
//--------------------------------------------------------------------------------------//
//
//                              START PRIORITY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, component::hip_event,
                               priority_constant<128>)
//
//--------------------------------------------------------------------------------------//
//
//                              STOP PRIORITY
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, component::hip_event,
                               priority_constant<-128>)
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_HIP)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::rocm, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::hip_event, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::roctx_marker, false_type)
#endif
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(hip_event, TIMEMORY_HIP_EVENT, "hip_event", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(roctx_marker, TIMEMORY_ROCTX_MARKER, "roctx_marker",
                                 "roctx")
