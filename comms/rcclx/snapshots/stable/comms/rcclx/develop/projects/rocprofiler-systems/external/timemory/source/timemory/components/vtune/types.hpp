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

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_COMPONENT(vtune_event)
TIMEMORY_DECLARE_COMPONENT(vtune_frame)
TIMEMORY_DECLARE_COMPONENT(vtune_profiler)

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//

TIMEMORY_SET_COMPONENT_API(component::vtune_event, category::logger, category::external,
                           tpls::intel)

TIMEMORY_SET_COMPONENT_API(component::vtune_frame, category::decorator,
                           category::external, tpls::intel)

TIMEMORY_SET_COMPONENT_API(component::vtune_profiler, category::external, tpls::intel)

//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_USE_VTUNE)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::intel, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::vtune_event, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::vtune_frame, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::vtune_profiler, false_type)
#endif

//--------------------------------------------------------------------------------------//
//
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::vtune_event, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::vtune_frame, true_type)

//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(vtune_event, TIMEMORY_VTUNE_EVENT, "vtune_event", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(vtune_frame, TIMEMORY_VTUNE_FRAME, "vtune_frame", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(vtune_profiler, TIMEMORY_VTUNE_PROFILER,
                                 "vtune_profiler", "")
