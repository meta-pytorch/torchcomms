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
#include "timemory/components/perfetto/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"

TIMEMORY_DECLARE_COMPONENT(perfetto_trace)

TIMEMORY_SET_COMPONENT_API(component::perfetto_trace, project::timemory, tpls::perfetto,
                           category::external, category::timing, category::visualization,
                           os::agnostic)

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::perfetto_trace, false_type)

#if !defined(TIMEMORY_USE_PERFETTO)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::perfetto_trace, false_type)
#endif

#if defined(TIMEMORY_PYBIND11_SOURCE)
//
namespace tim
{
namespace trait
{
//
template <>
struct python_args<TIMEMORY_STORE, component::perfetto_trace>
{
    using type = type_list<type_list<size_t>, type_list<const char*, size_t>>;
};
//
template <>
struct python_args<TIMEMORY_START, component::perfetto_trace>
{
    using type = type_list<type_list<const char*>>;
};
}  // namespace trait
}  // namespace tim
#endif

TIMEMORY_PROPERTY_SPECIALIZATION(perfetto_trace, TIMEMORY_PERFETTO_TRACE,
                                 "perfetto_trace", "perfetto")
