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
 * \file timemory/operations/types/cache.hpp
 * \brief Definition for various functions for cache in operations
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/type_traits.hpp"
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
struct cache
{
    using type       = Tp;
    using cache_type = typename trait::cache<Tp>::type;

    TIMEMORY_DEFAULT_OBJECT(cache)

    template <typename RetT                                             = cache_type,
              enable_if_t<!concepts::is_null_type<RetT>::value &&
                          std::is_trivially_constructible<RetT>::value> = 0>
    RetT operator()()
    {
        return RetT{};
    }

    template <typename RetT, enable_if_t<std::is_same<RetT, cache_type>::value> = 0>
    auto operator()(RetT&& val)
    {
        return std::forward<RetT>(val);
    }

    template <typename RetT                                    = cache_type,
              enable_if_t<concepts::is_null_type<RetT>::value> = 0>
    void operator()()
    {}

    template <typename RetT, enable_if_t<!std::is_same<RetT, cache_type>::value ||
                                         concepts::is_null_type<RetT>::value> = 0>
    void operator()(RetT&& val)
    {
        return std::forward<RetT>(val);
    }

    template <typename Func, typename RetT,
              enable_if_t<std::is_same<RetT, cache_type>::value &&
                          !concepts::is_null_type<RetT>::value> = 0>
    auto operator()(type& _obj, Func&& _func, RetT&& _arg)
    {
        return ((_obj).*(_func))(std::forward<RetT>(_arg));
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp>
struct construct_cache
{
    using data_type = std::tuple<std::remove_pointer_t<Tp>...>;
    using type =
        mpl::unique_t<mpl::get_trait_type_t<trait::cache, data_type>, std::tuple<>>;

    auto operator()() const { return type{}; }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename... Tp>
struct construct_cache<TupleT<Tp...>>
: conditional_t<concepts::is_wrapper<TupleT<Tp...>>::value, construct_cache<Tp...>,
                construct_cache<TupleT<Tp...>>>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp>
struct construct_cache<std::tuple<Tp...>> : construct_cache<Tp...>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp>
struct construct_cache<type_list<Tp...>> : construct_cache<Tp...>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp>
using construct_cache_t = typename construct_cache<Tp...>::type;
//
}  // namespace operation
}  // namespace tim
