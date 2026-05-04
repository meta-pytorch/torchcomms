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
percent_diff(Tp _lhs, Tp _rhs, type_list<>, ...)
{
    constexpr Tp _zero    = Tp(0.0);
    constexpr Tp _one     = Tp(1.0);
    constexpr Tp _hundred = Tp(100.0);
    Tp&&         _pdiff   = (_rhs > _zero) ? ((_one - (_lhs / _rhs)) * _hundred) : _zero;
    return (_pdiff < _zero) ? _zero : _pdiff;
}

#if defined(CXX17)
template <typename... Tp, typename Up>
TIMEMORY_INLINE auto
percent_diff(std::variant<Tp...>& _lhs, Up _rhs, type_list<>, ...)
{
    utility::variant_apply(
        _lhs,
        [](auto& _out, auto&& _inp) {
            using type = concepts::unqualified_type_t<decltype(_out)>;
            percent_diff(_out, static_cast<type>(_inp));
        },
        _rhs);
    return _lhs;
}
#endif

template <typename Tp,
          enable_if_t<!std::is_arithmetic<Tp>::value && std::is_class<Tp>::value> = 0>
TIMEMORY_INLINE auto
percent_diff(Tp _lhs, Tp _rhs, type_list<>, ...) -> decltype(_lhs.percent_diff(_rhs))
{
    return _lhs.percent_diff(_rhs);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, type_list<>, long)
    -> decltype(std::begin(_lhs), Tp{})
{
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    // initialize
    for(auto& itr : _ret)
        itr = Vp{};

    // compute
    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = percent_diff(*litr, *ritr);
    }
    return _ret;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, type_list<>, int)
    -> decltype(std::begin(_lhs), Tp{})
{
    assert(_lhs.size() == _rhs.size());
    Tp _ret{};
    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
        {
            _ret[litr->first] = Mp{};
        }
        else
        {
            _ret[litr->first] = percent_diff(litr->second, ritr->second);
        }
    }
    return _ret;
}

template <typename Tp, size_t... Idx>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, index_sequence<Idx...>, ...)
    -> decltype(std::get<0>(_lhs), Tp{})
{
    Tp _ret{};
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(_ret) =
                                 percent_diff(std::get<Idx>(_lhs), std::get<Idx>(_rhs)));
    return _ret;
}

template <typename Tp>
Tp
percent_diff(const Tp& _lhs, const Tp& _rhs)
{
    return percent_diff(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
}
}  // namespace math
}  // namespace tim
