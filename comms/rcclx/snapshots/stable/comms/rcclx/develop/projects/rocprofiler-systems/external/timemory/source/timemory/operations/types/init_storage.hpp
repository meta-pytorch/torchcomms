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
 * \file timemory/operations/types/init_storage.hpp
 * \brief Definition for various functions for init_storage in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/storage/declaration.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up>
init_storage<Tp>::init_storage(enable_if_t<trait::uses_value_storage<Up>::value, int>)
{
#if defined(TIMEMORY_DISABLE_COMPONENT_STORAGE_INIT)
#else
    using value_type = typename type::value_type;
    base::storage::template base_instance<type, value_type>();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename U, typename V>
typename init_storage<Tp>::get_type init_storage<Tp>::get(
    enable_if_t<trait::uses_value_storage<U, V>::value, int>)
{
#if defined(TIMEMORY_DISABLE_COMPONENT_STORAGE_INIT)
    return get_type{ nullptr, false, false, false };
#else
    static thread_local auto _instance = []() {
        if(!trait::runtime_enabled<Tp>::get())
            return get_type{ nullptr, false, false, false };

        using value_type = typename type::value_type;
        auto this_inst   = tim::base::storage::template base_instance<Tp, value_type>();
        this_inst->initialize();
        bool this_glob = true;
        bool this_work = true;
        bool this_data = this_inst->data_init();
        return get_type{ this_inst, this_glob, this_work, this_data };
    }();
    return _instance;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename U, typename V>
typename init_storage<Tp>::get_type init_storage<Tp>::get(
    enable_if_t<
        !trait::uses_value_storage<U, V>::value &&
            !std::is_same<typename U::storage_type, component::empty_storage>::value,
        int>)
{
#if defined(TIMEMORY_DISABLE_COMPONENT_STORAGE_INIT)
    return get_type{ nullptr, false, false, false };
#else
    static thread_local auto _instance = []() {
        if(!trait::runtime_enabled<Tp>::get())
            return get_type{ nullptr, false, false, false };

        using value_type = typename type::value_type;
        auto this_inst   = tim::base::storage::template base_instance<Tp, value_type>();
        this_inst->initialize();
        return get_type{ this_inst, false, false, false };
    }();
    return _instance;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename U, typename V>
typename init_storage<Tp>::get_type init_storage<Tp>::get(
    enable_if_t<
        !trait::uses_value_storage<U, V>::value &&
            std::is_same<typename U::storage_type, component::empty_storage>::value,
        int>)
{
    return get_type{ nullptr, false, false, false };
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
init_storage<Tp>::init()
{
    if(!trait::runtime_enabled<Tp>::get())
        return;

#if defined(TIMEMORY_DISABLE_COMPONENT_STORAGE_INIT)
#else
    static thread_local auto _init = this_type::get();
    consume_parameters(_init);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
