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
/// \class tim::auto_tuple
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper where all components are allocated
/// on the stack and cannot be disabled at runtime. This bundler has the lowest
/// overhead. Accepts unlimited number of template parameters. The constructor starts the
/// components, the destructor stops the components.
///
/// \code{.cpp}
/// using bundle_t = tim::auto_tuple<wall_clock, cpu_clock, peak_rss>;
///
/// void foo()
/// {
///     auto bar = bundle_t("foo");
///     // ...
/// }
/// \endcode
///
/// The above code will record wall-clock, cpu-clock, and peak-rss. The intermediate
/// storage will happen on the stack and when the destructor is called, it will add itself
/// to the call-graph
///
template <typename... Types>
class auto_tuple
: public auto_base_bundle<TIMEMORY_API, component_tuple<>, auto_tuple<Types...>>
, public concepts::stack_wrapper
{
    using poly_base =
        auto_base_bundle<TIMEMORY_API, component_tuple<>, auto_tuple<Types...>>;

public:
    using this_type      = auto_tuple<Types...>;
    using base_type      = component_tuple<Types...>;
    using auto_type      = this_type;
    using component_type = typename base_type::component_type;
    using type           = convert_t<mpl::available_t<concat<Types...>>, auto_tuple<>>;

    template <typename... Args>
    explicit auto_tuple(Args&&... args);

    // copy and move
    ~auto_tuple()                 = default;
    auto_tuple(const auto_tuple&) = default;
    auto_tuple(auto_tuple&&)      = default;
    auto_tuple& operator=(const auto_tuple&) = default;
    auto_tuple& operator=(auto_tuple&&) = default;

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
auto_tuple<Types...>::auto_tuple(Args&&... args)
: poly_base{ std::forward<Args>(args)... }
{}
//
template <typename... Types>
auto_tuple<Types...>&
auto_tuple<Types...>::print(std::ostream& os, bool _endl) const
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
get(const auto_tuple<Types...>& _obj)
{
    return get(_obj.get_component());
}
//
template <typename... Types>
auto
get_labeled(const auto_tuple<Types...>& _obj)
{
    return get_labeled(_obj.get_component());
}
//
}  // namespace tim

//======================================================================================//
//
// variadic versions
//
#if !defined(TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE)
#    define TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE(tag, ...)                                 \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_tuple<__VA_ARGS__>;      \
        TIMEMORY_BLANK_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

#if !defined(TIMEMORY_VARIADIC_BASIC_AUTO_TUPLE)
#    define TIMEMORY_VARIADIC_BASIC_AUTO_TUPLE(tag, ...)                                 \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_tuple<__VA_ARGS__>;      \
        TIMEMORY_BASIC_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

#if !defined(TIMEMORY_VARIADIC_AUTO_TUPLE)
#    define TIMEMORY_VARIADIC_AUTO_TUPLE(tag, ...)                                       \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_tuple<__VA_ARGS__>;      \
        TIMEMORY_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

//======================================================================================//
//
//      std::get operator
//
//======================================================================================//
//
namespace std
{
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(tim::auto_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const tim::auto_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename... Types>
auto
get(tim::auto_tuple<Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::auto_tuple<Types...>>(obj).data()))
{
    using obj_type = tim::auto_tuple<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace std
