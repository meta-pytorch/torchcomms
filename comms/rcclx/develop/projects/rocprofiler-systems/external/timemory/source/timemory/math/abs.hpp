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
template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0,
          enable_if_t<std::is_integral<Tp>::value && std::is_unsigned<Tp>::value> = 0>
TIMEMORY_INLINE auto
abs(Tp _val, type_list<>, ...) -> decltype(Tp{})
{
    return _val;
}

#if defined(CXX17)
template <typename... Tp>
TIMEMORY_INLINE decltype(auto)
abs(std::variant<Tp...> _lhs, type_list<>, ...)
{
    utility::variant_apply(_lhs, [](auto& _out) { _out = abs(_out); });
    return _lhs;
}
#endif

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0,
          enable_if_t<!(std::is_integral<Tp>::value && std::is_unsigned<Tp>::value)> = 0>
auto TIMEMORY_INLINE
abs(Tp _val, type_list<>, int) -> decltype(std::abs(_val), Tp{})
{
    return std::abs(_val);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
abs(Tp _val, type_list<>, ...) -> decltype(std::begin(_val), Tp{})
{
    for(auto& itr : _val)
        itr = ::tim::math::abs(itr);
    return _val;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
abs(Tp _val, type_list<>, int) -> decltype(std::begin(_val), Tp{})
{
    for(auto& itr : _val)
    {
        itr.second = ::tim::math::abs(itr.second);
    }
    return _val;
}

template <template <typename...> class Tuple, typename... Types, size_t... Idx>
auto
abs(Tuple<Types...> _val, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_val), Tuple<Types...>())
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(_val) = ::tim::math::abs(std::get<Idx>(_val)));
    return _val;
}

template <typename Tp>
Tp
abs(Tp _val)
{
    return ::tim::math::abs(_val, get_index_sequence<Tp>::value, 0);
}
}  // namespace math
}  // namespace tim
