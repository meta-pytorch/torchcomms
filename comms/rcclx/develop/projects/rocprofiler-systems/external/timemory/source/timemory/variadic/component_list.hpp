
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
/// \class tim::component_list
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper where all components are optional
/// at runtime. Accept unlimited number of parameters. The default behavior is
/// to query the TIMEMORY_COMPONENT_LIST_INIT environment variable once (the first
/// time the bundle is used) and use that list of components (if any) to
/// initialize the components which are part of it's template parameters.
/// This behavior can be modified by assigning a new lambda/functor to the
/// reference which is returned from \ref
/// tim::component_list<Types...>::get_initializer(). Assignment is not thread-safe since
/// this is relatively unnecessary... if a different set of components are required on a
/// particular thread, just create a different type with those particular components or
/// pass the initialization functor to the constructor.
///
/// \code{.cpp}
/// using bundle_t = tim::component_list<wall_clock, cpu_clock, peak_rss>;
///
/// void foo()
/// {
///     setenv("TIMEMORY_COMPONENT_LIST_INIT", "wall_clock", 0);
///
///     auto bar = bundle_t("bar");
///
///     bundle_t::get_initializer() = [](bundle_t& b)
///     {
///         b.initialize<cpu_clock, peak_rss>();
///     };
///
///     auto qix = bundle_t("qix");
///
///     auto local_init = [](bundle_t& b)
///     {
///         b.initialize<thread_cpu_clock, peak_rss>();
///     };
///
///     auto spam = bundle_t("spam", ..., local_init);
///
/// }
/// \endcode
///
/// The above code will record wall-clock timer on first use of "bar", and
/// will record cpu-clock, peak-rss at "qix", and peak-rss at "spam". If foo()
/// is called a second time, "bar" will record cpu-clock and peak-rss. "spam" will
/// always use the local initialized. If none of these initializers are set, wall-clock
/// will be recorded for all of them. The intermediate storage will happen on the heap and
/// when the destructor is called, it will add itself to the call-graph
template <typename... Types>
class component_list
: public bundle<TIMEMORY_API, component_list<>,
                tim::variadic::heap_wrapper_types<concat<Types...>>>
, public concepts::heap_wrapper
{
public:
    using captured_location_t = source_location::captured;

    using bundle_type    = bundle<TIMEMORY_API, component_list<>,
                               tim::variadic::heap_wrapper_types<concat<Types...>>>;
    using this_type      = component_list<Types...>;
    using component_type = component_list<Types...>;
    using auto_type      = auto_list<Types...>;

public:
    template <typename... Args>
    component_list(Args&&...);

    ~component_list()                     = default;
    component_list(const component_list&) = default;
    component_list(component_list&&)      = default;

    component_list& operator=(const component_list& rhs) = default;
    component_list& operator=(component_list&&) = default;
};

template <typename... Types>
template <typename... Args>
component_list<Types...>::component_list(Args&&... args)
: bundle_type{ std::forward<Args>(args)... }
{}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get(const component_list<Types...>& _obj)
    -> decltype(std::declval<component_list<Types...>>().get())
{
    return _obj.get();
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get_labeled(const component_list<Types...>& _obj)
    -> decltype(std::declval<component_list<Types...>>().get_labeled())
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
get(tim::component_list<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const tim::component_list<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(tim::component_list<Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::component_list<Types...>>(obj).data()))
{
    using obj_type = tim::component_list<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std
