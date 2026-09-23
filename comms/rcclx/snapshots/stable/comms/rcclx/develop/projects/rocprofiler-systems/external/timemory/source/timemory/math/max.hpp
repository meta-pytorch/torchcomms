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
template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0>
TIMEMORY_INLINE Tp
max(Tp _lhs, Tp _rhs, type_list<>, ...)
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    return (_rhs < _lhs) ? _lhs : _rhs;
}

#if defined(CXX17)
template <typename... Tp, typename Up>
TIMEMORY_INLINE decltype(auto)
max(std::variant<Tp...> _lhs, Up _rhs, type_list<>, ...)
{
    utility::variant_apply(
        _lhs,
        [](auto& _out, auto&& _inp) {
            using type = concepts::unqualified_type_t<decltype(_out)>;
            _out       = max(_out, static_cast<type>(_inp));
        },
        _rhs);
    return _lhs;
}
#endif

template <typename Tp, typename Vp = typename Tp::value_type>
auto
max(const Tp& _lhs, const Tp& _rhs, type_list<>, ...) -> decltype(std::begin(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = ::tim::math::max(*litr, *ritr);
    }
    return _ret;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
max(const Tp& _lhs, const Tp& _rhs, type_list<>, int) -> decltype(std::begin(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr   = std::begin(_lhs) + i;
        auto ritr   = std::begin(_rhs) + i;
        auto itr    = std::begin(_ret) + i;
        itr->second = ::tim::math::max(litr->second, ritr->second);
    }
    return _ret;
}

template <typename Tp, size_t... Idx>
auto
max(const Tp& _lhs, const Tp& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    Tp _ret{};
    TIMEMORY_FOLD_EXPRESSION(
        std::get<Idx>(_ret) = ::tim::math::max(std::get<Idx>(_lhs), std::get<Idx>(_rhs)));
    return _ret;
}

template <typename Tp>
Tp
max(const Tp& _lhs, const Tp& _rhs)
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    return max(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
}
}  // namespace math
}  // namespace tim
