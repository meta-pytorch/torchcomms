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

/**
 * \file timemory/operations/types/math.hpp
 * \brief Definition for various functions for math in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::plus
/// \brief Define addition operations
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct plus
{
    using type = Tp;

    template <typename Up>
    using base_t = typename Up::base_type;

    TIMEMORY_DEFAULT_OBJECT(plus)

    template <typename... Args>
    explicit plus(type& obj, Args&&... args)
    {
        (*this)(obj, std::forward<Args>(args)...);
    }

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    auto& operator()(type& obj, const type& rhs) const
    {
        using namespace tim::stl;
        obj += rhs;
        // ensures update to laps
        sfinae(obj, 0, 0, rhs);
        return obj;
    }

    template <typename Vt, typename Up = Tp, enable_if_t<!has_data<Up>::value, char> = 0>
    auto& operator()(type& obj, const Vt&) const
    {
        return obj;
    }

private:
    template <typename Up>
    static auto sfinae(Up& obj, int, int, const Up& rhs)
        -> decltype(static_cast<base_t<Up>&>(obj).plus(crtp::base{},
                                                       static_cast<base_t<Up>&>(rhs)))
    {
        return static_cast<base_t<Up>&>(obj).plus(crtp::base{},
                                                  static_cast<base_t<Up>&>(rhs));
    }

    template <typename Up>
    static auto sfinae(Up& obj, int, long, const Up& rhs) -> decltype(obj.plus(rhs))
    {
        return obj.plus(rhs);
    }

    template <typename Up>
    static auto& sfinae(Up& _v, long, long, const Up&)
    {
        return _v;
    }
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::minus
/// \brief Define subtraction operations
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct minus
{
    using type = Tp;

    template <typename Up>
    using base_t = typename Up::base_type;

    TIMEMORY_DEFAULT_OBJECT(minus)

    template <typename... Args>
    explicit minus(type& obj, Args&&... args)
    {
        (*this)(obj, std::forward<Args>(args)...);
    }

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    inline auto& operator()(type& obj, const type& rhs) const
    {
        using namespace tim::stl;
        obj -= rhs;
        // ensures update to laps
        sfinae(obj, 0, 0, rhs);
        return obj;
    }

    template <typename Vt, typename Up = Tp, enable_if_t<!has_data<Up>::value, char> = 0>
    inline auto& operator()(type& _v, const Vt&) const
    {
        return _v;
    }

private:
    template <typename Up>
    static auto sfinae(Up& obj, int, int, const Up& rhs)
        -> decltype(static_cast<base_t<Up>&>(obj).minus(crtp::base{},
                                                        static_cast<base_t<Up>&>(rhs)))
    {
        return static_cast<base_t<Up>&>(obj).minus(crtp::base{},
                                                   static_cast<base_t<Up>&>(rhs));
    }

    template <typename Up>
    static auto sfinae(Up& obj, int, long, const Up& rhs) -> decltype(obj.minus(rhs))
    {
        return obj.minus(rhs);
    }

    template <typename Up>
    static auto& sfinae(Up& _v, long, long, const Up&)
    {
        return _v;
    }
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::multiply
/// \brief This operation class is used for multiplication of a component
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct multiply
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(multiply)

    template <typename... Args>
    explicit multiply(type& obj, Args&&... args)
    {
        (*this)(obj, std::forward<Args>(args)...);
    }

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    inline auto& operator()(type& obj, int64_t rhs) const
    {
        using namespace tim::stl;
        obj *= rhs;
        return obj;
    }

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    inline auto& operator()(type& obj, const type& rhs) const
    {
        using namespace tim::stl;
        obj *= rhs;
        return obj;
    }

    template <typename Vt, typename Up = Tp, enable_if_t<!has_data<Up>::value, char> = 0>
    inline auto& operator()(type& _v, const Vt&) const
    {
        return _v;
    }
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::divide
/// \brief This operation class is used for division of a component
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct divide
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(divide)

    template <typename... Args>
    explicit divide(type& obj, Args&&... args)
    {
        (*this)(obj, std::forward<Args>(args)...);
    }

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    inline auto& operator()(type& obj, int64_t rhs) const
    {
        using namespace tim::stl;
        obj /= rhs;
        return obj;
    }

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    inline auto& operator()(type& obj, const type& rhs) const
    {
        using namespace tim::stl;
        obj /= rhs;
        return obj;
    }

    template <typename Vt, typename Up = Tp, enable_if_t<!has_data<Up>::value, char> = 0>
    inline auto& operator()(type& _v, const Vt&) const
    {
        return _v;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
