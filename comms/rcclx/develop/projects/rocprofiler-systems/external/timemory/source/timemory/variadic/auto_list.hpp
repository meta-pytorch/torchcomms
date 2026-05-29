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
//

#pragma once

#include "timemory/variadic/auto_base_bundle.hpp"
#include "timemory/variadic/types.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tim
{
//--------------------------------------------------------------------------------------//
/// \class tim::auto_list
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper where all components are optional
/// at runtime. Accept unlimited number of parameters. The constructor starts the
/// components, the destructor stops the components. The default behavior is
/// to query the TIMEMORY_AUTO_LIST_INIT environment variable once (the first
/// time the bundle is used) and use that list of components (if any) to
/// initialize the components which are part of it's template parameters.
/// This behavior can be modified by assigning a new lambda/functor to the
/// reference which is returned from \ref tim::auto_list<Types...>::get_initializer().
/// Assignment is not thread-safe since this is relatively unnecessary... if a different
/// set of components are required on a particular thread, just create a different
/// type with those particular components or pass the initialization functor to the
/// constructor.
///
/// \code{.cpp}
/// using bundle_t = tim::auto_list<wall_clock, cpu_clock, peak_rss>;
///
/// void foo()
/// {
///     setenv("TIMEMORY_AUTO_LIST_COMPONENTS", "wall_clock", 0);
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
class auto_list
: public auto_base_bundle<TIMEMORY_API, component_list<>, auto_list<Types...>>
, public concepts::heap_wrapper
{
    using poly_base =
        auto_base_bundle<TIMEMORY_API, component_list<>, auto_list<Types...>>;

public:
    using this_type      = auto_list<Types...>;
    using base_type      = component_list<Types...>;
    using auto_type      = this_type;
    using component_type = typename base_type::component_type;
    using type           = convert_t<typename component_type::type, auto_list<>>;

    template <typename... Args>
    explicit auto_list(Args&&... args);

    // copy and move
    ~auto_list()                = default;
    auto_list(const auto_list&) = default;
    auto_list(auto_list&&)      = default;
    auto_list& operator=(const auto_list&) = default;
    auto_list& operator=(auto_list&&) = default;

    static constexpr std::size_t size() { return poly_base::size(); }

public:
    this_type&           print(std::ostream& os, bool _endl = false) const;
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        obj.print(os, false);
        return os;
    }
};
//
template <typename... Types>
template <typename... Args>
auto_list<Types...>::auto_list(Args&&... args)
: poly_base{ std::forward<Args>(args)... }
{}
//
template <typename... Types>
auto_list<Types...>&
auto_list<Types...>::print(std::ostream& os, bool _endl) const
{
    os << poly_base::m_temporary;
    if(_endl)
        os << '\n';
    return const_cast<this_type&>(*this);
}
//
//======================================================================================//
//
template <typename... Types>
auto
get(const auto_list<Types...>& _obj)
{
    return get(_obj.get_component());
}
//
template <typename... Types>
auto
get_labeled(const auto_list<Types...>& _obj)
{
    return get_labeled(_obj.get_component());
}
//
}  // namespace tim

//--------------------------------------------------------------------------------------//
// variadic versions
//
#if !defined(TIMEMORY_VARIADIC_BLANK_AUTO_LIST)
#    define TIMEMORY_VARIADIC_BLANK_AUTO_LIST(tag, ...)                                  \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_list<__VA_ARGS__>;       \
        TIMEMORY_BLANK_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

#if !defined(TIMEMORY_VARIADIC_BASIC_AUTO_LIST)
#    define TIMEMORY_VARIADIC_BASIC_AUTO_LIST(tag, ...)                                  \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_list<__VA_ARGS__>;       \
        TIMEMORY_BASIC_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

#if !defined(TIMEMORY_VARIADIC_AUTO_LIST)
#    define TIMEMORY_VARIADIC_AUTO_LIST(tag, ...)                                        \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_list<__VA_ARGS__>;       \
        TIMEMORY_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

//======================================================================================//
//
//      std::get operator
//
namespace std
{
//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(tim::auto_list<Types...>& obj) -> decltype(get<N>(obj.data()))
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(const tim::auto_list<Types...>& obj) -> decltype(get<N>(obj.data()))
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(tim::auto_list<Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::auto_list<Types...>>(obj).data()))
{
    using obj_type = tim::auto_list<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//

}  // namespace std

//======================================================================================//
