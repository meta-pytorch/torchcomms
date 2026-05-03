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
 * \file timemory/operations/types/measure.hpp
 * \brief Definition for various functions for measure in operations
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
/// \struct tim::operation::measure
/// \brief This operation class is used for a single measurement for a component
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct measure
{
    using type = Tp;

    TIMEMORY_DELETED_OBJECT(measure)

    template <typename... Args>
    explicit measure(type& obj, Args&&... args);

private:
    //  satisfies mpl condition and accepts arguments
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.measure(std::forward<Args>(args)...), void())
    {
        init_storage<Tp>::init();
        obj.measure(std::forward<Args>(args)...);
    }

    //  satisfies mpl condition but does not accept arguments
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.measure(), void())
    {
        init_storage<Tp>::init();
        obj.measure();
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {
        SFINAE_WARNING(type);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
measure<Tp>::measure(type& obj, Args&&... args)
{
    sfinae(obj, 0, 0, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
