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
 * \file timemory/components/papi/types.hpp
 * \brief Declare the papi component types
 */

#pragma once

#include "timemory/backends/types/papi.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/papi/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

#if !defined(TIMEMORY_PAPI_ARRAY_SIZE)
#    define TIMEMORY_PAPI_ARRAY_SIZE 8
#endif

namespace tim
{
namespace component
{
struct papi_config;

template <typename Tp>
struct papi_config_factory;

template <typename Tp>
struct papi_common;
}  // namespace component
}  // namespace tim
//
TIMEMORY_DECLARE_COMPONENT(papi_vector)
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(papi_tuple, int... EventTypes)
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(papi_array, size_t MaxNumEvents)
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(papi_rate_tuple, typename RateT, int... EventTypes)
//
TIMEMORY_COMPONENT_ALIAS(papi_array8_t, papi_array<8>)
TIMEMORY_COMPONENT_ALIAS(papi_array16_t, papi_array<16>)
TIMEMORY_COMPONENT_ALIAS(papi_array32_t, papi_array<32>)
TIMEMORY_COMPONENT_ALIAS(papi_array_t, papi_array<TIMEMORY_PAPI_ARRAY_SIZE>)
//
TIMEMORY_SET_COMPONENT_API(component::papi_vector, tpls::papi, category::external,
                           category::hardware_counter, os::supports_linux)
//
TIMEMORY_SET_TEMPLATE_COMPONENT_API(TIMEMORY_ESC(size_t MaxNumEvents),
                                    TIMEMORY_ESC(component::papi_array<MaxNumEvents>),
                                    tpls::papi, category::external,
                                    category::hardware_counter, os::supports_linux)
//
TIMEMORY_SET_TEMPLATE_COMPONENT_API(TIMEMORY_ESC(int... Evts),
                                    TIMEMORY_ESC(component::papi_tuple<Evts...>),
                                    tpls::papi, category::external,
                                    category::hardware_counter, os::supports_linux)
//
TIMEMORY_SET_TEMPLATE_COMPONENT_API(
    TIMEMORY_ESC(typename RateT, int... Evts),
    TIMEMORY_ESC(component::papi_rate_tuple<RateT, Evts...>), tpls::papi,
    category::external, category::hardware_counter, category::timing, os::supports_linux)
//
//======================================================================================//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(component::papi_vector, std::vector<double>)
TIMEMORY_TEMPLATE_STATISTICS_TYPE(component::papi_array, std::vector<double>, size_t)
TIMEMORY_VARIADIC_TRAIT_TYPE(statistics, component::papi_tuple, TIMEMORY_ESC(int... Idx),
                             TIMEMORY_ESC(Idx...), std::array<double, sizeof...(Idx)>)
TIMEMORY_VARIADIC_TRAIT_TYPE(statistics, component::papi_rate_tuple,
                             TIMEMORY_ESC(typename RateT, int... Idx),
                             TIMEMORY_ESC(RateT, Idx...),
                             std::array<double, sizeof...(Idx)>)
//
//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_PAPI)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::papi, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::papi_vector, false_type)
TIMEMORY_DEFINE_TEMPLATE_TRAIT(is_available, component::papi_array, false_type, size_t)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::papi_tuple, false_type, int)
//
namespace tim
{
namespace trait
{
template <typename RateT, int... EventTypes>
struct is_available<component::papi_rate_tuple<RateT, EventTypes...>> : false_type
{};
}  // namespace trait
}  // namespace tim
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(array_serialization, component::papi_vector, true_type)
TIMEMORY_DEFINE_TEMPLATE_TRAIT(array_serialization, component::papi_array, true_type,
                               size_t)
TIMEMORY_DEFINE_VARIADIC_TRAIT(array_serialization, component::papi_tuple, true_type, int)
//
namespace tim
{
namespace trait
{
template <typename RateT, int... EventTypes>
struct array_serialization<component::papi_rate_tuple<RateT, EventTypes...>> : true_type
{};

template <typename Tp>
struct runtime_enabled<component::papi_common<Tp>>
: runtime_enabled<component::papi_config>
{
    static constexpr bool value = runtime_enabled<component::papi_config>::value;
    using runtime_enabled<component::papi_config>::get;
    using runtime_enabled<component::papi_config>::set;
};
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
//                              CUSTOM SERIALIZATION
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_serialization, component::papi_vector, true_type)
TIMEMORY_DEFINE_TEMPLATE_TRAIT(custom_serialization, component::papi_array, true_type,
                               size_t)
TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_serialization, component::papi_tuple, true_type,
                               int)
//
//--------------------------------------------------------------------------------------//
//
//                              SAMPLER
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(sampler, component::papi_vector, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(sampler, component::papi_array_t, true_type)
TIMEMORY_DEFINE_VARIADIC_TRAIT(sampler, component::papi_tuple, true_type, int)
//
//--------------------------------------------------------------------------------------//
//
//                              BASE HAS ACCUM
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace trait
{
template <typename RateT, int... EventTypes>
struct base_has_accum<component::papi_rate_tuple<RateT, EventTypes...>> : false_type
{};
}  // namespace trait
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
//                              PROPERTIES
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(papi_vector, TIMEMORY_PAPI_VECTOR, "papi_vector", "papi")
TIMEMORY_PROPERTY_SPECIALIZATION(papi_array_t, TIMEMORY_PAPI_ARRAY, "papi_array_t",
                                 "papi_array")
