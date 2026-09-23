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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file timemory/variadic/component_tuple.hpp
 * \headerfile variadic/component_tuple.hpp "timemory/variadic/component_tuple.hpp"
 *
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/general/source_location.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/bundle.hpp"
#include "timemory/variadic/functional.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace tim
{
/// \class tim::component_tuple
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper where all components are allocated
/// on the stack and cannot be disabled at runtime. This bundler has the lowest
/// overhead. Accepts unlimited number of template parameters. This bundler
/// is used by \ref tim::auto_tuple whose constructor and destructor invoke the
/// start() and stop() member functions respectively.
///
/// \code{.cpp}
/// using bundle_t = tim::component_tuple<wall_clock, cpu_clock, peak_rss>;
///
/// void foo()
/// {
///     auto bar = bundle_t("foo");
///     bar.start();
///     // ...
///     bar.stop();
/// }
/// \endcode
///
/// The above code will record wall-clock, cpu-clock, and peak-rss. The intermediate
/// storage will happen on the stack and when the destructor is called, it will add itself
/// to the call-graph
///
template <typename... Types>
class component_tuple
: public bundle<TIMEMORY_API, component_tuple<>,
                tim::variadic::stack_wrapper_types<concat<Types...>>>
, public concepts::stack_wrapper
{
public:
    using captured_location_t = source_location::captured;

    using bundle_type    = bundle<TIMEMORY_API, component_tuple<>,
                               tim::variadic::stack_wrapper_types<concat<Types...>>>;
    using this_type      = component_tuple<Types...>;
    using component_type = component_tuple<Types...>;
    using auto_type      = auto_tuple<Types...>;

public:
    template <typename... Args>
    component_tuple(Args&&...);

    ~component_tuple()                      = default;
    component_tuple(const component_tuple&) = default;
    component_tuple(component_tuple&&)      = default;

    component_tuple& operator=(const component_tuple& rhs) = default;
    component_tuple& operator=(component_tuple&&) = default;
};

template <typename... Types>
template <typename... Args>
component_tuple<Types...>::component_tuple(Args&&... args)
: bundle_type{ std::forward<Args>(args)... }
{}

//======================================================================================//

template <typename... Types>
auto
get(const component_tuple<Types...>& _obj)
    -> decltype(std::declval<component_tuple<Types...>>().get())
{
    return _obj.get();
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get_labeled(const component_tuple<Types...>& _obj)
    -> decltype(std::declval<component_tuple<Types...>>().get_labeled())
{
    return _obj.get_labeled();
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//
//
//      std::get operator
//
namespace std
{
//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(::tim::component_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const ::tim::component_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(::tim::component_tuple<Types...>&& obj)
    -> decltype(get<N>(std::forward<::tim::component_tuple<Types...>>(obj).data()))
{
    using obj_type = ::tim::component_tuple<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std

//--------------------------------------------------------------------------------------//
