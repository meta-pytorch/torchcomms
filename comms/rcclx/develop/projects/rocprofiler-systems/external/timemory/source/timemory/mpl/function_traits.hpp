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

#if defined(__GNUC__) && (__GNUC__ >= 7) && (__cplusplus < 201703L)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wnoexcept-type"
#    pragma GCC diagnostic ignored "-Wignored-attributes"
#elif defined(__GNUC__) && (__GNUC__ >= 6)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "timemory/utility/type_list.hpp"

#include <functional>
#include <type_traits>

namespace tim
{
namespace mpl
{
//--------------------------------------------------------------------------------------//
//
//  Function traits
//
//--------------------------------------------------------------------------------------//

template <typename T>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<std::function<R(Args...)>>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R(Args...)>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

// member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = type_list<C&, Args...>;
};

// const member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = true;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = type_list<C&, Args...>;
};

// member object pointer
template <typename C, typename R>
struct function_traits<R(C::*)>
{
    static constexpr bool is_memfun = true;
    static constexpr bool is_const  = false;
    static const size_t   nargs     = 0;
    using result_type               = R;
    using args_type                 = type_list<>;
    using call_type                 = type_list<C&>;
};

#if __cplusplus >= 201703L

template <typename R, typename... Args>
struct function_traits<std::function<R(Args...) noexcept>>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...) noexcept>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R(Args...) noexcept>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

// member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) noexcept>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = type_list<C&, Args...>;
};

// const member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const noexcept>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = true;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = type_list<C&, Args...>;
};

#endif
//
}  // namespace mpl
}  // namespace tim

#if defined(__GNUC__) && (__GNUC__ >= 7) && (__cplusplus < 201703L)
#    pragma GCC diagnostic pop
#elif defined(__GNUC__) && (__GNUC__ >= 6)
#    pragma GCC diagnostic pop
#endif
