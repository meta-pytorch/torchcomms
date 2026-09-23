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
 * \file timemory/operations/types/storage_initializer.hpp
 * \brief Definition for various functions for storage_initializer in operations
 */

#pragma once

#include "timemory/components/properties.hpp"
#include "timemory/manager/manager.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/type_list.hpp"

#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::storage_initializer
/// \brief This operation class is used for generic storage initalization
///
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
auto
invoke_preinit(int) -> decltype(std::declval<T>().preinit(), void())
{
    T::preinit();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
auto
invoke_preinit(long)
{}
//
//--------------------------------------------------------------------------------------//
//
struct preinitializer
{
    template <typename... Tp>
    bool operator()(type_list<Tp...>) const
    {
        TIMEMORY_FOLD_EXPRESSION(invoke_preinit<Tp>(0));
        return true;
    }
    template <size_t... Idx>
    bool operator()(std::index_sequence<Idx...>) const
    {
        TIMEMORY_FOLD_EXPRESSION(invoke_preinit<component::enumerator_t<Idx>>(0));
        return true;
    }

    template <typename... Tp>
    bool operator()() const
    {
        return (*this)(type_list<Tp...>{});
    }
    template <size_t... Idx>
    bool operator()() const
    {
        return (*this)(std::index_sequence<Idx...>{});
    }
};
//
//--------------------------------------------------------------------------------------//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
enable_if_t<trait::uses_storage<T>::value, storage_initializer> storage_initializer::get(
    std::true_type)
{
    auto library_ctor = tim::get_env<bool>("TIMEMORY_LIBRARY_CTOR", true);
    if(!library_ctor)
        return storage_initializer{};

    if(!trait::runtime_enabled<T>::get())
        return storage_initializer{};

    preinitializer{}.template operator()<T>();

    using storage_type = storage<T, typename T::value_type>;

    auto _main_init = []() {
        auto _manager = manager::master_instance();
        if(_manager)
            _manager->add_initializer([]() {
                (void) storage_type::instance();
                return false;
            });
        (void) storage_type::master_instance();
        return storage_initializer{};
    };
    static auto              _master = _main_init();
    static thread_local auto _worker = (storage_type::instance(), storage_initializer{});
    consume_parameters(_master);

    return _worker;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
enable_if_t<!trait::uses_storage<T>::value, storage_initializer> storage_initializer::get(
    std::false_type)
{
    return storage_initializer{};
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, enable_if_t<Idx != TIMEMORY_COMPONENTS_END>>
storage_initializer
storage_initializer::get()
{
    return storage_initializer::get<component::enumerator_t<Idx>>();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
