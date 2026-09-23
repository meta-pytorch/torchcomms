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

/** \file mpl/stl.hpp
 * \headerfile mpl/stl.hpp "timemory/mpl/stl.hpp"
 * Provides operators on common STL structures such as <<, +=, -=, *=, /=, +, -, *, /
 *
 */

#pragma once

#include "timemory/macros/language.hpp"
#include "timemory/math/stl.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/type_list.hpp"

#include <array>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#if defined(CXX17)
#    include <variant>
#endif

namespace tim
{
inline namespace stl
{
/// \namespace tim::stl::ostream
/// \brief the namespace provides overloads to output complex data types w/ streams
namespace ostream
{
template <template <typename...> class Tuple, typename... Types, size_t... Idx>
void
tuple_printer(const Tuple<Types...>& obj, std::ostream& os, index_sequence<Idx...>);

template <typename... Types, size_t... Idx>
void
type_list_printer(type_list<Types...> obj, std::ostream& os, index_sequence<Idx...>);

//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename T, typename U>
std::ostream&
operator<<(std::ostream&, const std::pair<T, U>&);

template <typename Tp, typename... Extra>
std::ostream&
operator<<(std::ostream&, const std::vector<Tp, Extra...>&);

template <typename Tp, size_t N>
std::ostream&
operator<<(std::ostream&, const std::array<Tp, N>&);

template <typename... Types>
std::ostream&
operator<<(std::ostream&, const std::tuple<Types...>&);

template <typename... Types>
std::ostream&
operator<<(std::ostream&, type_list<Types...>);

//--------------------------------------------------------------------------------------//

template <typename T, typename U>
std::ostream&
operator<<(std::ostream& os, const std::pair<T, U>& p)
{
    os << "(" << p.first << "," << p.second << ")";
    return os;
}

template <typename Tp, typename... ExtraT>
std::ostream&
operator<<(std::ostream& os, const std::vector<Tp, ExtraT...>& p)
{
    os << "(";
    for(size_t i = 0; i < p.size(); ++i)
        os << p.at(i) << ((i + 1 < p.size()) ? "," : "");
    os << ")";
    return os;
}

template <typename Tp, size_t N>
std::ostream&
operator<<(std::ostream& os, const std::array<Tp, N>& p)
{
    os << "(";
    for(size_t i = 0; i < p.size(); ++i)
        os << p.at(i) << ((i + 1 < p.size()) ? "," : "");
    os << ")";
    return os;
}

template <typename... Types>
std::ostream&
operator<<(std::ostream& os, const std::tuple<Types...>& p)
{
    constexpr size_t N = sizeof...(Types);
    tuple_printer(p, os, make_index_sequence<N>{});
    return os;
}

template <typename... Types>
std::ostream&
operator<<(std::ostream& os, type_list<Types...> p)
{
    constexpr size_t N = sizeof...(Types);
    type_list_printer(p, os, make_index_sequence<N>{});
    return os;
}

template <template <typename...> class Tuple, typename... Types, size_t... Idx>
void
tuple_printer(const Tuple<Types...>& obj, std::ostream& os, index_sequence<Idx...>)
{
    using namespace ::tim::stl::ostream;
    constexpr size_t N = sizeof...(Types);

    if(N > 0)
        os << "(";
    char delim[N];
    TIMEMORY_FOLD_EXPRESSION(delim[Idx] = ',');
    delim[N - 1] = '\0';
    TIMEMORY_FOLD_EXPRESSION(os << std::get<Idx>(obj) << delim[Idx]);
    if(N > 0)
        os << ")";
}

template <typename... Types, size_t... Idx>
void
type_list_printer(type_list<Types...>, std::ostream& os, index_sequence<Idx...>)
{
    using namespace ::tim::stl::ostream;
    constexpr size_t N = sizeof...(Types);

    if(N > 0)
        os << "(";
    char delim[N];
    TIMEMORY_FOLD_EXPRESSION(delim[Idx] = ',');
    delim[N - 1] = '\0';
    TIMEMORY_FOLD_EXPRESSION(os << try_demangle<Types>() << delim[Idx]);
    if(N > 0)
        os << ")";
}

//--------------------------------------------------------------------------------------//

#if defined(CXX17)
template <typename Tp, typename... Types>
std::ostream&
operator<<(std::ostream&, const std::variant<Tp, Types...>&);

template <typename Tp, typename... Types>
std::ostream&
operator<<(std::ostream& _os, const std::variant<Tp, Types...>& _val)
{
    auto _func = [&_os](auto&& _v) { _os << _v; };
    std::visit(_func, _val);
    return _os;
}
#endif

}  // namespace ostream
}  // namespace stl
}  // namespace tim
