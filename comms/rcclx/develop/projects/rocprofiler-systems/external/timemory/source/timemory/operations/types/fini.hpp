// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a finalize
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, finalize, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above finalizeright notice and this permission notice shall be included in all
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
 * \file timemory/operations/types/fini.hpp
 * \brief Definition for global and thread-local finalization functions for a component
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
/// \struct tim::operation::fini
/// \brief This operation class is used for invoking the static initializer and
/// thread-local initializer of a component.
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct fini
{
    using type = Tp;

    fini()  = default;
    ~fini() = default;

    TIMEMORY_DELETE_COPY_MOVE_OBJECT(fini)

    template <typename Up                                       = Tp, int StateT,
              enable_if_t<trait::is_available<Up>::value, char> = 0>
    explicit TIMEMORY_COLD fini(mode_constant<StateT>)
    {
        auto _storage = storage<Tp>::noninit_instance();
        if(_storage)
            sfinae<type, StateT>(_storage, 0, 0);
    }

    template <typename Up = Tp, int StateT, typename StorageT,
              enable_if_t<trait::is_available<Up>::value, char> = 0>
    explicit TIMEMORY_COLD fini(StorageT _storage, mode_constant<StateT>)
    {
        sfinae<type, StateT>(_storage, 0, 0);
    }

    template <typename Up = Tp, int StateT, typename StorageT,
              enable_if_t<!trait::is_available<Up>::value, char> = 0>
    explicit TIMEMORY_COLD fini(StorageT, mode_constant<StateT>)
    {}

    template <typename Up                                       = Tp, int StateT,
              enable_if_t<trait::is_available<Up>::value, char> = 0>
    TIMEMORY_COLD auto operator()(mode_constant<StateT>)
    {
        auto _storage = storage<Tp>::noninit_instance();
        if(_storage)
            return sfinae<type, StateT>(_storage, 0, 0);
        return false;
    }

    template <typename Up                                        = Tp, int StateT,
              enable_if_t<!trait::is_available<Up>::value, char> = 0>
    TIMEMORY_COLD auto operator()(mode_constant<StateT>)
    {
        return false;
    }

    /// returns whether the finalization function has been executed (will return false
    /// if the component does not support any overload of `global_finalize(...)` or
    /// `thread_finalize(...)`
    template <int StateT>
    static bool get_executed()
    {
        return was_executed<StateT>();
    }

private:
    template <typename Up, int StateT, typename StorageT,
              enable_if_t<StateT == fini_mode::global, char> = 0>
    TIMEMORY_COLD auto sfinae(StorageT _storage, int, int)
        -> decltype(std::declval<Up>().global_finalize(_storage), bool())
    {
        if(was_executed<StateT>())
            return false;
        type::global_finalize(_storage);
        was_executed<StateT>() = true;
        return true;
    }

    template <typename Up, int StateT, typename StorageT,
              enable_if_t<StateT == fini_mode::global, int> = 0>
    TIMEMORY_COLD auto sfinae(StorageT, int, long)
        -> decltype(std::declval<Up>().global_finalize(), bool())
    {
        if(was_executed<StateT>())
            return false;
        type::global_finalize();
        was_executed<StateT>() = true;
        return true;
    }

    template <typename Up, int StateT, typename StorageT,
              enable_if_t<StateT == fini_mode::thread, char> = 0>
    TIMEMORY_COLD auto sfinae(StorageT _storage, int, int)
        -> decltype(std::declval<Up>().thread_finalize(_storage), bool())
    {
        if(was_executed<StateT>())
            return false;
        type::thread_finalize(_storage);
        was_executed<StateT>() = true;
        return true;
    }

    template <typename Up, int StateT, typename StorageT,
              enable_if_t<StateT == fini_mode::thread, int> = 0>
    TIMEMORY_COLD auto sfinae(StorageT, int, long)
        -> decltype(std::declval<Up>().thread_finalize(), bool())
    {
        if(was_executed<StateT>())
            return false;
        type::thread_finalize();
        was_executed<StateT>() = true;
        return true;
    }

    template <typename Up, int StateT, typename StorageT>
    bool sfinae(StorageT, long, long)
    {
        return false;
    }

private:
    template <int StateT, enable_if_t<StateT == fini_mode::global, char> = 0>
    static bool& was_executed()
    {
        static bool _instance = false;
        return _instance;
    }

    template <int StateT, enable_if_t<StateT == fini_mode::thread, char> = 0>
    static bool& was_executed()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
