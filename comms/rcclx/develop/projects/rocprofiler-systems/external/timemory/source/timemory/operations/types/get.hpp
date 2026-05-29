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
 * \file timemory/operations/types/get.hpp
 * \brief Definition for various functions for get in operations
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
/// \struct tim::operation::get
///
/// \brief The purpose of this operation class is to provide a non-template hook to get
/// the object itself
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct get
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(get)

    get(const type& obj, void*& ptr, size_t nhash) { get_sfinae(obj, 0, 0, ptr, nhash); }

    auto operator()(const type& obj, void*& ptr, size_t nhash) const
    {
        return get_sfinae(obj, 0, 0, ptr, nhash);
    }

private:
    template <typename U = type>
    static auto get_sfinae(const U& obj, int, int, void*& ptr, size_t nhash)
        -> decltype(obj.get(ptr, nhash))
    {
        using return_type = decltype(obj.get(ptr, nhash));

        if(!ptr)
            return obj.get(ptr, nhash);

        if constexpr(!std::is_void<return_type>::value)
            return return_type{};
    }

    template <typename U = type, typename base_type = typename U::base_type>
    static auto get_sfinae(const U& obj, int, long, void*& ptr, size_t nhash)
        -> decltype(static_cast<const base_type&>(obj).get(ptr, nhash))
    {
        using return_type = decltype(static_cast<const base_type&>(obj).get(ptr, nhash));

        if(!ptr)
            static_cast<const base_type&>(obj).get(ptr, nhash);

        if constexpr(!std::is_void<return_type>::value)
            return return_type{};
    }

    template <typename U = type>
    static type* get_sfinae(const U&, long, long, void*&, size_t)
    {
        return nullptr;
    }
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::get_data
/// \brief The purpose of this operation class is to combine the output types from the
/// "get()" member function for multiple components -- this is specifically used in the
/// Python interface to provide direct access to the results
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct get_data
{
    using type      = Tp;
    using data_type = decltype(std::declval<type>().get());

    TIMEMORY_DEFAULT_OBJECT(get_data)

    //----------------------------------------------------------------------------------//
    // SFINAE
    //
    get_data(const type& obj, data_type& dst) { sfinae(obj, 0, 0, dst); }

    template <typename Dp, typename... Args>
    get_data(const type& obj, Dp& dst, Args&&... args)
    {
        static_assert(std::is_same<Dp, data_type>::value, "Error! Dp != type::get()");
        sfinae(obj, 0, 0, dst, std::forward<Args>(args)...);
    }

    data_type operator()(const type& _obj)
    {
        data_type _data{};
        sfinae(_obj, 0, 0, _data);
        return _data;
    }

private:
    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<has_data<Up>::value, char> = 0>
    static auto sfinae(const Up& obj, int, int, Dp& dst, Args&&... args)
        -> decltype(obj.get(std::forward<Args>(args)...))
    {
        return (dst = obj.get(std::forward<Args>(args)...));
    }

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<has_data<Up>::value, char> = 0>
    static auto sfinae(const Up& obj, int, long, Dp& dst, Args&&...)
        -> decltype(obj.get())
    {
        return (dst = obj.get());
    }

    //----------------------------------------------------------------------------------//
    // component is available but no "get" function
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<has_data<Up>::value, char> = 0>
    static auto sfinae(const Up&, long, long, Dp&, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    // nothing if component is not available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<!has_data<Up>::value, char> = 0>
    static auto sfinae(const Up&, long, long, Dp&, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::get_data
/// \brief The purpose of this operation class is to combine the output types from the
/// "get()" member function for multiple components -- this is specifically used in the
/// Python interface to provide direct access to the results
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct get_labeled_data
{
    using type      = Tp;
    using data_type = std::tuple<std::string, decltype(std::declval<type>().get())>;

    TIMEMORY_DEFAULT_OBJECT(get_labeled_data)

    //----------------------------------------------------------------------------------//
    // SFINAE
    //
    get_labeled_data(const type& obj, data_type& dst) { sfinae(obj, 0, 0, dst); }

    template <typename Dp, typename... Args>
    get_labeled_data(const type& obj, Dp& dst, Args&&... args)
    {
        static_assert(std::is_same<Dp, data_type>::value,
                      "Error! Dp != tuple<string, type::get()>");
        sfinae(obj, 0, 0, dst, std::forward<Args>(args)...);
    }

    data_type operator()(const type& _obj)
    {
        data_type _data{};
        sfinae(_obj, 0, 0, _data);
        return _data;
    }

private:
    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<has_data<Up>::value, char> = 0>
    static auto sfinae(const Up& obj, int, int, Dp& dst, Args&&... args)
        -> decltype(data_type(type::get_label(), obj.get(std::forward<Args>(args)...)))
    {
        return (dst = data_type(type::get_label(), obj.get(std::forward<Args>(args)...)));
    }

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<has_data<Up>::value, char> = 0>
    static auto sfinae(const Up& obj, int, long, Dp& dst, Args&&...)
        -> decltype(data_type(type::get_label(), obj.get()))
    {
        return (dst = data_type(type::get_label(), obj.get()));
    }

    //----------------------------------------------------------------------------------//
    // component is available but no "get" function
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<has_data<Up>::value, char> = 0>
    static auto sfinae(const Up&, long, long, Dp&, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    // nothing if component is not available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<!has_data<Up>::value, char> = 0>
    static auto sfinae(const Up&, long, long, Dp&, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
