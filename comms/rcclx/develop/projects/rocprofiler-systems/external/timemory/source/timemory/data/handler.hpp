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

/** \file timemory/data/handler.hpp
 * \headerfile timemory/data/handler.hpp "timemory/data/handler.hpp"
 * Defines the data_tracker handler component interface
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/data/types.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"

#include <sstream>
#include <string>

namespace tim
{
namespace data
{
/// \struct tim::data::handler<V, Tag>
/// \brief This class is used to provide the implementation for tracking data
/// via the data_tracker component. It is written such that it can be
/// defined for a specific data type (e.g. int) with a tag (e.g. ConvergenceIterations)
/// and shared among multiple template types of data_tracker<T>
///
template <typename V, typename Tag>
struct handler
{
    using value_type = V;

public:
    /// this function is returns the current value
    template <typename T>
    static decltype(auto) get(const T& obj)
    {
        return handler::get(obj, 0);
    }

    /// this function is returns the current value in a form suitable for display.
    /// It may be necessary to specialize this function downstream.
    template <typename T>
    static decltype(auto) get_display(const T& obj)
    {
        return handler::get(obj);
    }

public:
    //----------------------------------------------------------------------------------//
    // const ref semantics
    //----------------------------------------------------------------------------------//
    /// this function is used to store a value
    template <typename T>
    static void store(T& obj, const value_type& v)
    {
        obj.set_value(v);
    }

    /// \tparam FuncT Should be binary operation which takes two inputs and returns one
    /// this function is used to store the current value after performing some operation.
    template <typename T, typename FuncT>
    static auto store(T& obj, FuncT&& f, const value_type& v)
        -> decltype(obj.set_value(std::forward<FuncT>(f)(obj.get_value(), v)), void())
    {
        obj.set_value(std::forward<FuncT>(f)(obj.get_value(), v));
    }

    /// this function sets the value of the temporary in `mark_begin`.
    static void begin(value_type& obj, const value_type& v) { obj = v; }

    /// this function sets the value of the temporary in `mark_begin`.
    template <typename FuncT>
    static void begin(value_type& obj, FuncT&& f, const value_type& v)
    {
        obj = std::forward<FuncT>(f)(obj, v);
    }

    /// this function computes the difference between the provided value and the
    /// temporary from `mark_begin` and then updates the current value
    template <typename T>
    static void end(T& obj, const value_type& v)
    {
        auto _v = v;
        _v      = math::minus(_v, obj.get_temporary());
        store(obj, std::move(_v));
    }

    /// this function computes the difference between the provided value and the
    /// temporary from `mark_begin` and then updates the current value
    template <typename T, typename FuncT>
    static void end(T& obj, FuncT&& f, const value_type& v)
    {
        auto _v = v;
        _v      = math::minus(_v, obj.get_temporary());
        store(obj, std::forward<FuncT>(f), std::move(_v));
    }

public:
    //----------------------------------------------------------------------------------//
    // move semantics
    //----------------------------------------------------------------------------------//
    /// overload with move semantics
    template <typename T>
    static void store(T& obj, value_type&& v)
    {
        obj.set_value(std::move(v));
    }

    /// overload with move semantics
    template <typename T, typename FuncT>
    static auto store(T& obj, FuncT&& f, value_type&& v)
        -> decltype(obj.set_value(std::forward<FuncT>(f)(obj.get_value(), std::move(v))),
                    void())
    {
        obj.set_value(std::forward<FuncT>(f)(obj.get_value(), std::move(v)));
    }

    /// overload with move semantics
    static void begin(value_type& obj, value_type&& v) { obj = v; }

    /// overload with move semantics
    template <typename T, typename FuncT>
    static auto begin(value_type& obj, FuncT&& f, value_type&& v)
    {
        obj = std::forward<FuncT>(f)(obj, v);
    }

    /// overload with move semantics
    template <typename T>
    static void end(T& obj, value_type&& v)
    {
        auto&& _v = std::move(v);
        math::minus(_v, obj.get_temporary());
        obj.set_value(std::move(_v));
    }

    /// overload with move semantics
    template <typename T, typename FuncT>
    static void end(T& obj, FuncT&& f, value_type&& v)
    {
        auto&& _v = std::move(v);
        _v        = math::minus(_v, obj.get_temporary());
        store(obj, std::forward<FuncT>(f), std::move(_v));
    }

private:
    // prefer the get_value version since value is updated by default
    template <typename T>
    static auto get(const T& obj, int) -> decltype(obj.get_value())
    {
        return obj.get_value();
    }

    template <typename T>
    static decltype(auto) get(const T& obj, long)
    {
        return obj.load();
    }
};
}  // namespace data
}  // namespace tim
