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
#include "timemory/data/types.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_TEMPLATE_COMPONENT(data_tracker, typename InpT,
                                    typename Tag = TIMEMORY_API)

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace component
{
/// \typedef tim::component::data_tracker_integer
/// \brief Specialization of \ref tim::component::data_tracker for storing signed integer
/// data
using data_tracker_integer = data_tracker<intmax_t, TIMEMORY_API>;

/// \typedef tim::component::data_tracker_unsigned
/// \brief Specialization of \ref tim::component::data_tracker for storing unsigned
/// integer data
using data_tracker_unsigned = data_tracker<size_t, TIMEMORY_API>;

/// \typedef tim::component::data_tracker_floating
/// \brief Specialization of \ref tim::component::data_tracker for storing floating point
/// data
using data_tracker_floating = data_tracker<double, TIMEMORY_API>;
}  // namespace component
}  // namespace tim
//
namespace tim
{
namespace trait
{
template <typename InpT, typename Tag>
struct component_apis<component::data_tracker<InpT, Tag>>
{
    using type = type_list<TIMEMORY_API, category::logger, os::agnostic>;
};
//
template <typename InpT, typename Tag>
struct assignable_units<component::data_tracker<InpT, Tag>> : true_type
{};
//
#if defined(TIMEMORY_COMPILER_INSTRUMENTATION)
template <typename InpT, typename Tag>
struct is_available<component::data_tracker<InpT, Tag>> : std::false_type
{};
#endif
//
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::data_tracker_integer, intmax_t)
TIMEMORY_STATISTICS_TYPE(component::data_tracker_unsigned, size_t)
TIMEMORY_STATISTICS_TYPE(component::data_tracker_floating, double)
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace trait
{
//
template <typename InpT, typename Tag>
struct base_has_accum<component::data_tracker<InpT, Tag>> : false_type
{};
//
template <typename InpT, typename Tag>
struct is_component<component::data_tracker<InpT, Tag>> : true_type
{};
//
template <>
struct python_args<TIMEMORY_RECORD, component::data_tracker_integer>
{
    using type = type_list<intmax_t>;
};
//
template <>
struct python_args<TIMEMORY_RECORD, component::data_tracker_unsigned>
{
    using type = type_list<size_t>;
};
//
template <>
struct python_args<TIMEMORY_RECORD, component::data_tracker_floating>
{
    using type = type_list<double>;
};
//
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(data_tracker_integer, TIMEMORY_DATA_TRACKER_INTEGER,
                                 "data_tracker_integer", "integer_data_tracker")
TIMEMORY_PROPERTY_SPECIALIZATION(data_tracker_unsigned, TIMEMORY_DATA_TRACKER_UNSIGNED,
                                 "data_tracker_unsigned", "unsigned_data_tracker")
TIMEMORY_PROPERTY_SPECIALIZATION(data_tracker_floating, TIMEMORY_DATA_TRACKER_FLOATING,
                                 "data_tracker_floating", "floating_data_tracker")
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_METADATA_SPECIALIZATION(
    data_tracker_integer, "data_integer", "Stores signed integer data w.r.t. call-graph",
    "Useful for tracking +/- values in different call-graph contexts")
TIMEMORY_METADATA_SPECIALIZATION(
    data_tracker_unsigned, "data_unsigned",
    "Stores unsigned integer data w.r.t. call-graph",
    "Useful for tracking iterations, etc. in different call-graph contexts")
TIMEMORY_METADATA_SPECIALIZATION(
    data_tracker_floating, "data_floating",
    "Stores double-precision fp data w.r.t. call-graph",
    "Useful for tracking values in different call-graph contexts")
//
//======================================================================================//
