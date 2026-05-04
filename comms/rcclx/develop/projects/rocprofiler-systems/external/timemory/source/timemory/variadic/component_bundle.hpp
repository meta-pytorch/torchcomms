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

/** \file timemory/variadic/component_bundle.hpp
 * \headerfile variadic/component_bundle.hpp "timemory/variadic/component_bundle.hpp"
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

//======================================================================================//
/// \class tim::component_bundle
/// \tparam Tag unique identifying type for the bundle which when \ref
/// tim::trait::is_available<Tag> is false at compile-time or \ref
/// tim::trait::runtime_enabled<Tag>() is false at runtime, then none of the components
/// will be collected
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper which combines the features of \ref
/// tim::component_tuple<T...> and \ref tim::component_list<U..>. The "T" types
/// (compile-time fixed, allocated on stack) should be specified as usual, the "U" types
/// (runtime-time optional, allocated on the heap) should be specified as a pointer.
/// Initialization of the optional types is similar to \ref tim::auto_list<U...> but no
/// environment variable is built-in since, ideally, this environment variable should be
/// customized based on the Tag template parameter.
///
/// See also: \ref tim::auto_bundle.
/// The primary difference b/t the "component_*" and "auto_*" is that the latter
/// used the constructor/destructor to call start and stop and is thus easier to
/// just copy-and-paste into different places. However, the former is better suited for
/// special configuration, data-access, etc.
///
namespace tim
{
//
template <typename Tag, typename... Types>
class component_bundle<Tag, Types...>
: public bundle<Tag, component_bundle<Tag>,
                tim::variadic::mixed_wrapper_types<concat<Types...>>>
, public concepts::mixed_wrapper
{
public:
    using captured_location_t = source_location::captured;

    using bundle_type    = bundle<Tag, component_bundle<Tag>,
                               tim::variadic::mixed_wrapper_types<concat<Types...>>>;
    using this_type      = component_bundle<Tag, Types...>;
    using component_type = component_bundle<Tag, Types...>;
    using auto_type      = auto_bundle<Tag, Types...>;

public:
    template <typename... Args>
    component_bundle(Args&&...);

    ~component_bundle()                       = default;
    component_bundle(const component_bundle&) = default;
    component_bundle(component_bundle&&)      = default;

    component_bundle& operator=(const component_bundle& rhs) = default;
    component_bundle& operator=(component_bundle&&) = default;
};

template <typename Tag, typename... Types>
template <typename... Args>
component_bundle<Tag, Types...>::component_bundle(Args&&... args)
: bundle_type{ std::forward<Args>(args)... }
{}

//
//======================================================================================//
//
template <typename... Types>
auto
get(const component_bundle<Types...>& _obj)
    -> decltype(std::declval<component_bundle<Types...>>().get())
{
    return _obj.get();
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get_labeled(const component_bundle<Types...>& _obj)
    -> decltype(std::declval<component_bundle<Types...>>().get_labeled())
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

template <std::size_t N, typename Tag, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(::tim::component_bundle<Tag, Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename Tag, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const ::tim::component_bundle<Tag, Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename Tag, typename... Types>
auto
get(::tim::component_bundle<Tag, Types...>&& obj)
    -> decltype(get<N>(std::forward<::tim::component_bundle<Tag, Types...>>(obj).data()))
{
    using obj_type = ::tim::component_bundle<Tag, Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std

//--------------------------------------------------------------------------------------//
