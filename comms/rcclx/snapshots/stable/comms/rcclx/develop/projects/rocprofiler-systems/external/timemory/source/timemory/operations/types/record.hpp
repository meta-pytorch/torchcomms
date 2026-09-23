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
 * \file timemory/operations/types/record.hpp
 * \brief Definition for various functions for record in operations
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
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct record
{
    using type = Tp;

    TIMEMORY_DELETED_OBJECT(record)

    record(type& obj, const type& rhs);

    template <typename T                                     = type, typename... Args,
              enable_if_t<check_record_type<T>::value, char> = 0>
    explicit record(T& obj, Args&&... args);

    template <typename T                                      = type, typename... Args,
              enable_if_t<!check_record_type<T>::value, char> = 0>
    explicit record(T&, Args&&...);

private:
    //  satisfies mpl condition and accepts arguments
    template <typename Up, typename T, typename... Args,
              typename Vp = typename Up::value_type,
              enable_if_t<check_record_type<Up, Vp>::value, int> = 0>
    auto sfinae(T& obj, int, int, Args&&... args)
        -> decltype((std::declval<T&>().value = obj.record(std::forward<Args>(args)...)),
                    void())
    {
        obj.value = obj.record(std::forward<Args>(args)...);
    }

    //  satisfies mpl condition but does not accept arguments
    template <typename Up, typename T, typename... Args,
              typename Vp = typename Up::value_type,
              enable_if_t<check_record_type<Up, Vp>::value, int> = 0>
    auto sfinae(T& obj, int, long, Args&&...)
        -> decltype((std::declval<T&>().value = obj.record()), void())
    {
        obj.value = obj.record();
    }

    //  satisfies mpl condition but does not accept arguments
    template <typename Up, typename T, typename... Args,
              typename Vp = typename Up::value_type,
              enable_if_t<check_record_type<Up, Vp>::value, int> = 0>
    auto sfinae(T&, long, long, Args&&...) -> decltype(void(), void())
    {}

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename T, typename... Args>
    void sfinae(T&, long, long, Args&&...)
    {
        SFINAE_WARNING(type);
    }

    //  satisfies mpl condition but does not accept arguments
    template <typename T>
    auto sfinae(T& obj, const T& rhs, int, long) -> decltype(obj += rhs)
    {
        return obj += rhs;
    }

    //  no member function or does not satisfy mpl condition
    template <typename T>
    void sfinae(T&, const T&, long, long)
    {
        SFINAE_WARNING(type);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
record<Tp>::record(type& obj, const type& rhs)
{
    sfinae(obj, rhs, 0, 0);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename T, typename... Args, enable_if_t<check_record_type<T>::value, char>>
record<Tp>::record(T& obj, Args&&... args)
{
    sfinae<type>(obj, 0, 0, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename T, typename... Args, enable_if_t<!check_record_type<T>::value, char>>
record<Tp>::record(T&, Args&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
