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

/** \file timemory/math/stl.hpp
 * \headerfile timemory/math/stl.hpp "timemory/math/stl.hpp"
 * Provides operators on common STL structures such as <<, +=, -=, *=, /=, +, -, *, /
 *
 */

#pragma once

#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/math/fwd.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <chrono>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(CXX17)
#    include <variant>
#endif

namespace tim
{
/// \namespace tim::stl
/// \brief the namespace is provided to hide stl overload from global namespace but
/// provide a method of using the namespace without a "using namespace tim;"
inline namespace stl
{
//--------------------------------------------------------------------------------------//
//
//      operator += (same type)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N, typename OtherT>
std::array<Tp, N>&
operator+=(std::array<Tp, N>& lhs, OtherT&& rhs)
{
    math::plus(lhs, std::forward<OtherT>(rhs));
    return lhs;
}

template <typename Lhs, typename Rhs, typename OtherT>
std::pair<Lhs, Rhs>&
operator+=(std::pair<Lhs, Rhs>& lhs, OtherT&& rhs)
{
    math::plus(lhs, std::forward<OtherT>(rhs));
    return lhs;
}

template <typename Tp, typename... ExtraT, typename OtherT>
std::vector<Tp, ExtraT...>&
operator+=(std::vector<Tp, ExtraT...>& lhs, OtherT&& rhs)
{
    math::plus(lhs, std::forward<OtherT>(rhs));
    return lhs;
}

template <typename... Types, typename OtherT>
std::tuple<Types...>&
operator+=(std::tuple<Types...>& lhs, OtherT&& rhs)
{
    math::plus(lhs, std::forward<OtherT>(rhs));
    return lhs;
}

#if defined(CXX17)
template <typename... Types, typename OtherT>
std::variant<Types...>&
operator+=(std::variant<Types...>& lhs, OtherT&& rhs)
{
    math::plus(lhs, std::forward<OtherT>(rhs));
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator -= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator-=(std::array<Tp, N>& lhs, const std::array<Tp, N>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator-=(std::pair<Lhs, Rhs>& lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

template <typename Tp, typename... ExtraT>
std::vector<Tp, ExtraT...>&
operator-=(std::vector<Tp, ExtraT...>& lhs, const std::vector<Tp, ExtraT...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

template <typename... Types>
std::tuple<Types...>&
operator-=(std::tuple<Types...>& lhs, const std::tuple<Types...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

#if defined(CXX17)
template <typename... Types>
std::variant<Types...>&
operator-=(std::variant<Types...>& lhs, const std::variant<Types...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator *= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator*=(std::array<Tp, N>& lhs, const std::array<Tp, N>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>& lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

template <typename Tp, typename... ExtraT>
std::vector<Tp, ExtraT...>&
operator*=(std::vector<Tp, ExtraT...>& lhs, const std::vector<Tp, ExtraT...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

template <typename... Types>
std::tuple<Types...>&
operator*=(std::tuple<Types...>& lhs, const std::tuple<Types...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

#if defined(CXX17)
template <typename... Types>
std::variant<Types...>&
operator*=(std::variant<Types...>& lhs, const std::variant<Types...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator /= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator/=(std::array<Tp, N>& lhs, const std::array<Tp, N>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>& lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

template <typename Tp, typename... ExtraT>
std::vector<Tp, ExtraT...>&
operator/=(std::vector<Tp, ExtraT...>& lhs, const std::vector<Tp, ExtraT...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

template <typename... Types>
std::tuple<Types...>&
operator/=(std::tuple<Types...>& lhs, const std::tuple<Types...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

#if defined(CXX17)
template <typename... Types>
std::variant<Types...>&
operator/=(std::variant<Types...>& lhs, const std::variant<Types...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator *= (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::array<Lhs, N>&
operator*=(std::array<Lhs, N>& lhs, Rhs rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int>>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>& lhs, ArithT rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

template <typename Lhs, typename Rhs, typename... ExtraT,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::vector<Lhs, ExtraT...>&
operator*=(std::vector<Lhs, ExtraT...>& lhs, Rhs rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::tuple<Lhs...>&
operator*=(std::tuple<Lhs...>& lhs, Rhs rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

#if defined(CXX17)
template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::variant<Lhs...>&
operator*=(std::variant<Lhs...>& lhs, Rhs rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator /= (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::array<Lhs, N>&
operator/=(std::array<Lhs, N>& lhs, Rhs rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int>>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>& lhs, ArithT rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

template <typename Lhs, typename Rhs, typename... ExtraT,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::vector<Lhs, ExtraT...>&
operator/=(std::vector<Lhs, ExtraT...>& lhs, Rhs rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::tuple<Lhs...>&
operator/=(std::tuple<Lhs...>& lhs, Rhs rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

#if defined(CXX17)
template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::variant<Lhs...>&
operator/=(std::variant<Lhs...>& lhs, Rhs rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator * (fundamental)
//      operator / (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
Lhs operator*(Lhs lhs, Rhs rhs)
{
    return (lhs *= rhs);
}

template <typename Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
Lhs
operator/(Lhs lhs, Rhs rhs)
{
    return (lhs /= rhs);
}

template <typename Rhs, enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
inline std::chrono::system_clock::time_point&
operator/=(std::chrono::system_clock::time_point& lhs, Rhs)
{
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator +
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator+(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator+(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator+(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>
operator+(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

#if defined(CXX17)
template <typename... Types>
std::variant<Types...>
operator+(std::variant<Types...> lhs, const std::variant<Types...>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator -
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator-(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator-(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator-(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>
operator-(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

#if defined(CXX17)
template <typename... Types>
std::variant<Types...>
operator-(std::variant<Types...> lhs, const std::variant<Types...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator *
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N> operator*(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs> operator*(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...> operator*(std::vector<Tp, Extra...>        lhs,
                                    const std::vector<Tp, Extra...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...> operator*(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

#if defined(CXX17)
template <typename... Types>
std::variant<Types...> operator*(std::variant<Types...>        lhs,
                                 const std::variant<Types...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}
#endif

//--------------------------------------------------------------------------------------//
//
//      operator /
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator/(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator/(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator/(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>
operator/(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

#if defined(CXX17)
template <typename... Types>
std::variant<Types...>
operator/(std::variant<Types...> lhs, const std::variant<Types...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}
#endif
}  // namespace stl
}  // namespace tim

namespace std
{
#if defined(TIMEMORY_WINDOWS)

template <typename Lhs, typename Rhs>
const pair<Lhs, Rhs>
operator-(pair<Lhs, Rhs> lhs, const pair<Lhs, Rhs>& rhs)
{
    ::tim::math::minus(lhs, rhs);
    return lhs;
}

template <typename... Types>
const tuple<Types...>
operator-(tuple<Types...> lhs, const tuple<Types...>& rhs)
{
    ::tim::math::minus(lhs, rhs);
    return lhs;
}

#endif

template <typename Tp>
tuple<>&
operator+=(tuple<>& _lhs, const Tp&)
{
    return _lhs;
}

#if defined(CXX17)
template <typename... Types>
const variant<Types...>
operator-(variant<Types...> lhs, const variant<Types...>& rhs)
{
    ::tim::math::minus(lhs, rhs);
    return lhs;
}
#endif
}  // namespace std
