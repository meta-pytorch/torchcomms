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
 * \file timemory/operations/types/store.hpp
 * \brief Definition for various functions for store in operations
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
template <typename Tp>
struct store
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(store)

    template <typename... Args>
    TIMEMORY_HOT explicit store(type& obj, Args&&... args)
    {
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

    template <typename... Args>
    TIMEMORY_HOT auto operator()(type& obj, Args&&... args)
    {
        return sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename Up, typename... Args>
    TIMEMORY_HOT auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.store(std::forward<Args>(args)...))
    {
        return obj.store(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Up, typename... Args>
    TIMEMORY_HOT auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.store())
    {
        return obj.store();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Up, typename... Args>
    TIMEMORY_INLINE void sfinae(Up&, long, long, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
