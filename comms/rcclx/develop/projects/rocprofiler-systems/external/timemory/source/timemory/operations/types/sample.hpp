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
 * \file timemory/operations/types/sample.hpp
 * \brief Definition for various functions for sample in operations
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
/// \struct tim::operation::sample
/// \brief This operation class is used for sampling
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct sample
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(sample)

    explicit sample(type& obj);

    template <typename Arg, typename... Args>
    sample(type& obj, Arg&& arg, Args&&... args);

    template <typename... Args>
    TIMEMORY_INLINE auto operator()(type& obj, Args&&... args) const
    {
        return sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //  satisfies mpl condition and accepts arguments
    template <typename Up, typename... Args>
    static TIMEMORY_INLINE auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.sample(std::forward<Args>(args)...))
    {
        return obj.sample(std::forward<Args>(args)...);
    }

    //  satisfies mpl condition but does not accept arguments
    template <typename Up, typename... Args>
    static TIMEMORY_INLINE auto sfinae(Up& obj, int, long, Args&&...)
        -> decltype(obj.sample())
    {
        return obj.sample();
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    static TIMEMORY_INLINE null_type sfinae(Up&, long, long, Args&&...)
    {
        return null_type{};
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
sample<Tp>::sample(Tp& obj)
{
    sfinae(obj, 0, 0, null_type{});
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Arg, typename... Args>
sample<Tp>::sample(Tp& obj, Arg&& arg, Args&&... args)
{
    sfinae(obj, 0, 0, std::forward<Arg>(arg), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
