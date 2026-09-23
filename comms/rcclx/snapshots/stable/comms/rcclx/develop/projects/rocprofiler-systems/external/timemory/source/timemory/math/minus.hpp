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

#include "timemory/math/fwd.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

namespace tim
{
namespace math
{
template <typename Tp, typename Up>
TIMEMORY_INLINE auto
minus(Tp& _lhs, const Up& _rhs, type_list<>, ...)
    -> decltype(_lhs -= _rhs, std::declval<Tp&>())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    _lhs -= _rhs;
    return _lhs;
}

#if defined(CXX17)
template <typename... Tp, typename Up>
TIMEMORY_INLINE decltype(auto)
minus(std::variant<Tp...>& _lhs, Up _rhs, type_list<>, ...)
{
    utility::variant_apply(
        _lhs,
        [](auto& _out, auto&& _inp) {
            using type = concepts::unqualified_type_t<decltype(_out)>;
            minus(_out, static_cast<type>(_inp));
        },
        _rhs);
    return _lhs;
}
#endif

template <typename Tp, typename Up, typename Vp = typename Tp::value_type>
auto
minus(Tp& _lhs, const Up& _rhs, type_list<>, long)
    -> decltype(std::begin(_lhs), std::declval<Tp&>())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        minus(*litr, *ritr);
    }
    return _lhs;
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
minus(Tp& _lhs, const Up& _rhs, type_list<>, int)
    -> decltype(std::begin(_lhs), std::declval<Tp&>())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        minus(litr->second, ritr->second);
    }
    return _lhs;
}

template <typename Tp, typename Up>
TIMEMORY_INLINE decltype(auto)
minus(Tp& _lhs, const Up&, index_sequence<>, int)
{
    return _lhs;
}

template <typename Tp, typename Up, size_t... Idx>
auto
minus(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), std::declval<Tp&>())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    TIMEMORY_FOLD_EXPRESSION(minus(std::get<Idx>(_lhs), std::get<Idx>(_rhs)));
    return _lhs;
}

template <typename Tp, typename Up>
Tp&
minus(Tp& _lhs, const Up& _rhs)
{
    if constexpr(!concepts::is_null_type<Tp>::value && !concepts::is_null_type<Up>::value)
        minus(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
    return _lhs;
    (void) _rhs;
}
}  // namespace math
}  // namespace tim
