// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "timemory/backends/threading.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/mpl/types.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <type_traits>

namespace tim
{
namespace threading
{
using construct_on_init = std::true_type;

template <typename ApiT, typename Tp, typename... ExtraT>
struct TIMEMORY_VISIBLE tls_data;

template <typename ApiT, typename Tp, typename... ExtraT>
struct tls_data
{
    static constexpr size_t value = trait::max_threads<ApiT, Tp, ExtraT...>::value;
    using value_type              = std::unique_ptr<Tp>;
    using array_type              = std::array<value_type, value>;

    static array_type& instances();
    static value_type& instance();

    template <typename... Args>
    static value_type& instance(construct_on_init, Args&&...);

    template <typename... Args>
    static array_type& instances(construct_on_init, Args&&...);

    template <typename... Args>
    static void construct(Args&&...);

    static constexpr size_t size() { return value; }

    static auto begin() { return instances().begin(); }
    static auto cbegin() { return instances().cbegin(); }
    static auto rbegin() { return instances().rbegin(); }
    static auto end() { return instances().end(); }
    static auto cend() { return instances().cend(); }
    static auto rend() { return instances().rend(); }
};

template <typename ApiT, typename Tp, typename... ExtraT>
template <typename... Args>
void
tls_data<ApiT, Tp, ExtraT...>::construct(Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto&             _instances = instances();
    static thread_local bool _v         = [&_args...]() {
        _instances.at(threading::get_id()) =
            std::make_unique<Tp>(std::forward<Args>(_args)...);
        return true;
    }();
    (void) _v;
}

template <typename ApiT, typename Tp, typename... ExtraT>
typename tls_data<ApiT, Tp, ExtraT...>::value_type&
tls_data<ApiT, Tp, ExtraT...>::instance()
{
    return instances().at(threading::get_id());
}

template <typename ApiT, typename Tp, typename... ExtraT>
typename tls_data<ApiT, Tp, ExtraT...>::array_type&
tls_data<ApiT, Tp, ExtraT...>::instances()
{
    static auto _v = array_type{};
    return _v;
}

template <typename ApiT, typename Tp, typename... ExtraT>
template <typename... Args>
typename tls_data<ApiT, Tp, ExtraT...>::value_type&
tls_data<ApiT, Tp, ExtraT...>::instance(construct_on_init, Args&&... _args)
{
    construct(std::forward<Args>(_args)...);
    return instances().at(threading::get_id());
}

template <typename ApiT, typename Tp, typename... ExtraT>
template <typename... Args>
typename tls_data<ApiT, Tp, ExtraT...>::array_type&
tls_data<ApiT, Tp, ExtraT...>::instances(construct_on_init, Args&&... _args)
{
    static auto _v = [&]() {
        auto _internal = array_type{};
        for(size_t i = 0; i < size(); ++i)
            _internal.at(i) = std::make_unique<Tp>(std::forward<Args>(_args)...);
        return _internal;
    }();
    return _v;
}
}  // namespace threading
}  // namespace tim
