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
#include "timemory/mpl/type_traits.hpp"  // for type-traits

#if !defined(TIMEMORY_TIMESTAMP_SOURCE) && !defined(TIMEMORY_USE_TIMESTAMP_EXTERN)
#    if !defined(TIMEMORY_TIMESTAMP_COMPONENT_HEADER_MODE)
#        define TIMEMORY_TIMESTAMP_COMPONENT_HEADER_MODE 1
#    endif
#endif

TIMEMORY_DECLARE_COMPONENT(timestamp)

TIMEMORY_SET_COMPONENT_API(component::timestamp, project::timemory, category::timing,
                           os::agnostic)

TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::timestamp, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::timestamp, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(timeline_storage, component::timestamp, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::timestamp, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_mean, component::timestamp, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_count, component::timestamp, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_self, component::timestamp, false_type)

namespace tim
{
namespace component
{
using timestamp_entry_t = std::chrono::time_point<std::chrono::system_clock>;
}  // namespace component
}  // namespace tim

TIMEMORY_PROPERTY_SPECIALIZATION(timestamp, TIMEMORY_TIMESTAMP, "timestamp", "")
