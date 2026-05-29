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

#include "timemory/macros/attributes.hpp"  // TIMEMORY_INLINE
#include "timemory/operations/types.hpp"

#include <utility>  // std::forward

namespace tim
{
namespace operation
{
template <typename Tp>
struct set_data
{
    TIMEMORY_DEFAULT_OBJECT(set_data)

    template <typename DataT>
    TIMEMORY_INLINE auto operator()(Tp& obj, DataT&& _data) const
    {
        return sfinae(obj, 0, std::forward<DataT>(_data));
    }

    template <typename DataT>
    TIMEMORY_INLINE auto operator()(DataT&& _data) const
    {
        return static_sfinae<Tp>(null_type{}, 0, std::forward<DataT>(_data));
    }

private:
    //  If the component has a set_data(...) member function
    template <typename T, typename DataT>
    TIMEMORY_INLINE static auto sfinae(T& obj, int, DataT&& _data)
        -> decltype(obj.set_data(std::forward<DataT>(_data)))
    {
        return obj.set_data(std::forward<DataT>(_data));
    }

    //  If the component does not have a set_data(...) member function
    template <typename T, typename DataT>
    TIMEMORY_INLINE static void sfinae(T&, long, DataT&&)
    {}

    //  If the component has a set_data(...) static member function
    template <typename Up = Tp, typename DataT>
    TIMEMORY_INLINE static auto static_sfinae(null_type&&, int, DataT&& _data)
        -> decltype(Up::set_data(std::forward<DataT>(_data)))
    {
        return Up::set_data(std::forward<DataT>(_data));
    }

    //  If the component does not have a set_data(...) static member function
    template <typename Up = Tp, typename DataT>
    TIMEMORY_INLINE static void static_sfinae(null_type&&, long, DataT&&)
    {}
};

template <>
struct set_data<void>
{
    TIMEMORY_DEFAULT_OBJECT(set_data)

    template <typename DataT>
    TIMEMORY_INLINE auto operator()(DataT&&) const
    {}
};
}  // namespace operation
}  // namespace tim
