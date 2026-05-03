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
 * \file timemory/operations/types/copy.hpp
 * \brief Definition for various functions for copy in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/set.hpp"

#include <memory>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::copy
/// \brief This operation class is used for copying the object generically
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct copy
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(copy)

    copy(Tp& obj, const Tp& rhs) { (*this)(obj, rhs); }
    copy(Tp*& obj, const Tp* rhs) { (*this)(obj, rhs); }
    copy(Tp*& obj, const std::shared_ptr<Tp>& rhs) { (*this)(obj, rhs.get()); }
    copy(Tp*& obj, const std::unique_ptr<Tp>& rhs) { (*this)(obj, rhs.get()); }

    template <typename Up = Tp>
    TIMEMORY_INLINE auto operator()(Tp& obj, Up&& v) const
    {
        return sfinae(obj, std::forward<Up>(v));
    }

    template <typename Up = Tp>
    TIMEMORY_INLINE auto operator()(Tp*& obj, Up&& v) const
    {
        return sfinae(obj, std::forward<Up>(v));
    }

private:
    template <typename Up, typename Dp = decay_t<Up>>
    TIMEMORY_INLINE Tp& sfinae(Tp& obj, Up&& rhs,
                               enable_if_t<trait::is_available<Dp>::value &&
                                               !std::is_pointer<decay_t<Dp>>::value,
                                           int> = 0) const;

    template <typename Up, typename Dp = decay_t<Up>>
    TIMEMORY_INLINE Tp* sfinae(
        Tp*& obj, Up&& rhs,
        enable_if_t<trait::is_available<Dp>::value && std::is_pointer<decay_t<Dp>>::value,
                    long> = 0) const;

    template <typename Up, typename Dp = decay_t<Up>>
    TIMEMORY_INLINE auto sfinae(
        Tp&, Up&&, enable_if_t<!trait::is_available<Dp>::value, int> = 0) const
    {}

    template <typename Up, typename Dp = decay_t<Up>>
    TIMEMORY_INLINE auto sfinae(
        Tp*&, Up&&, enable_if_t<!trait::is_available<Dp>::value, int> = 0) const
    {}
};
//
template <typename Tp>
template <typename Up, typename Dp>
Tp&
copy<Tp>::sfinae(
    Tp& obj, Up&& rhs,
    enable_if_t<trait::is_available<Dp>::value && !std::is_pointer<decay_t<Dp>>::value,
                int>) const
{
    obj = Up{ std::forward<Up>(rhs) };
    operation::set_iterator<Tp>{}(obj, nullptr);
    return obj;
}

template <typename Tp>
template <typename Up, typename Dp>
Tp*
copy<Tp>::sfinae(
    Tp*& obj, Up&& rhs,
    enable_if_t<trait::is_available<Dp>::value && std::is_pointer<decay_t<Dp>>::value,
                long>) const
{
    if(rhs)
    {
        if(!obj)
        {
            obj = new type{ *std::forward<Up>(rhs) };
        }
        else
        {
            *obj = type{ *std::forward<Up>(rhs) };
        }
        operation::set_iterator<Tp>{}(*obj, nullptr);
    }
    return obj;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
