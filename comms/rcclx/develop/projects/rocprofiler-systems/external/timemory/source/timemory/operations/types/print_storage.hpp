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
 * \file timemory/operations/types/print_storage.hpp
 * \brief Definition for various functions for print_storage in operations
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
/// \struct tim::operation::print_storage
/// \brief Print the storage for a component
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print_storage
{
    using type = Tp;

    TIMEMORY_COLD print_storage() { (*this)(); }

    TIMEMORY_COLD void operator()() const { this->sfinae<type>(0); }

private:
    //----------------------------------------------------------------------------------//
    // only if component is available
    //
    template <typename Up, typename Vp = typename Up::value_type,
              enable_if_t<is_enabled<Up>::value, char> = 0>
    TIMEMORY_COLD void sfinae(int) const
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        auto _storage = storage<Tp, Vp>::noninit_instance();
        if(_storage)
        {
            _storage->stack_clear();
            _storage->print();
        }
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename Up>
    void sfinae(long) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print_storage<Tp*> : print_storage<Tp>
{
    print_storage()
    : print_storage<Tp>()
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
