//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file components/types.hpp
 * \headerfile components/types.hpp "timemory/components/types.hpp"
 *
 * This is a declaration of all the component structs.
 * Care should be taken to make sure that this includes a minimal
 * number of additional headers.
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/components/skeletons.hpp"
#include "timemory/mpl/quirks.hpp"
#include "timemory/utility/types.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

//======================================================================================//
//
namespace tim
{
//======================================================================================//
//  components that provide implementations (i.e. HOW to record a component)
//
namespace component
{
// define this short-hand from C++14 for C++11
template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

// holder that provides nothing
template <typename... Types>
struct placeholder;

struct nothing : base<nothing, skeleton::base>
{};

struct info;
}  // namespace component
}  // namespace tim

//======================================================================================//

#include "timemory/components/allinea/types.hpp"
#include "timemory/components/base/types.hpp"
#include "timemory/components/caliper/types.hpp"
#include "timemory/components/craypat/types.hpp"
#include "timemory/components/cuda/types.hpp"
#include "timemory/components/cupti/types.hpp"
#include "timemory/components/data_tracker/types.hpp"
#include "timemory/components/gotcha/types.hpp"
#include "timemory/components/gperftools/types.hpp"
#include "timemory/components/hip/types.hpp"
#include "timemory/components/io/types.hpp"
#include "timemory/components/likwid/types.hpp"
#include "timemory/components/network/types.hpp"
#include "timemory/components/ompt/types.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/components/perfetto/types.hpp"
#include "timemory/components/printer/types.hpp"
#include "timemory/components/roofline/types.hpp"
#include "timemory/components/rusage/types.hpp"
#include "timemory/components/tau_marker/types.hpp"
#include "timemory/components/timestamp/types.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/components/trip_count/types.hpp"
#include "timemory/components/user_bundle/types.hpp"
#include "timemory/components/vtune/types.hpp"

//======================================================================================//

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, quirk::explicit_start, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, quirk::explicit_stop, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, quirk::no_init, false_type)

//======================================================================================//
//
//          Define tuple_size now that all type-traits have been declared
//
//======================================================================================//

#include "timemory/mpl/available.hpp"

namespace std
{
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
TSTAG(struct)
tuple_size<tim::lightweight_tuple<Types...>>
{
private:
    using type = tim::stl_tuple_t<Types...>;

public:
    static constexpr size_t value = tuple_size<type>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
TSTAG(struct)
tuple_size<tim::component_bundle<Tag, Types...>>
{
private:
    // component_bundle does not apply concat
    using type = tim::convert_t<tim::mpl::available_t<tuple<Types...>>, tuple<>>;

public:
    static constexpr size_t value = tuple_size<type>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
TSTAG(struct)
tuple_size<tim::component_tuple<Types...>>
{
private:
    using type = tim::stl_tuple_t<Types...>;

public:
    static constexpr size_t value = tuple_size<type>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
TSTAG(struct)
tuple_size<tim::component_list<Types...>>
{
private:
    using type = tim::stl_tuple_t<Types...>;

public:
    static constexpr size_t value = tuple_size<type>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
TSTAG(struct)
tuple_size<tim::auto_bundle<Tag, Types...>>
{
private:
    // auto_bundle does not apply concat
    using type = tim::convert_t<tim::mpl::available_t<tuple<Types...>>, tuple<>>;

public:
    static constexpr size_t value = tuple_size<type>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
TSTAG(struct)
tuple_size<tim::auto_tuple<Types...>>
{
private:
    using type = tim::stl_tuple_t<Types...>;

public:
    static constexpr size_t value = tuple_size<type>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
TSTAG(struct)
tuple_size<tim::auto_list<Types...>>
{
private:
    using type = tim::stl_tuple_t<Types...>;

public:
    static constexpr size_t value = tuple_size<type>::value;
};
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_DEPRECATED)
//
template <typename Tuple, typename List>
TSTAG(struct)
tuple_size<tim::component_hybrid<Tuple, List>>
{
public:
    static constexpr auto value = tuple_size<Tuple>::value + tuple_size<List>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tuple, typename List>
TSTAG(struct)
tuple_size<tim::auto_hybrid<Tuple, List>>
{
public:
    using value_type            = size_t;
    static constexpr auto value = tuple_size<Tuple>::value + tuple_size<List>::value;
};
//
#endif
//
}  // namespace std

//======================================================================================//
