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
 * \file timemory/operations/types/compose.hpp
 * \brief Definition for various functions for compose in operations
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
/// \struct tim::operation::compose
/// \brief The purpose of this operation class is operating on two components to compose
/// a result, e.g. use system-clock and user-clock to get a cpu-clock
///
//
//--------------------------------------------------------------------------------------//
//
template <typename RetType, typename LhsType, typename RhsType>
struct compose
{
    using ret_value_type = typename RetType::value_type;
    using lhs_value_type = typename LhsType::value_type;
    using rhs_value_type = typename RhsType::value_type;

    using ret_base_type = typename RetType::base_type;
    using lhs_base_type = typename LhsType::base_type;
    using rhs_base_type = typename RhsType::base_type;

    TIMEMORY_DELETED_OBJECT(compose)

    static_assert(std::is_same<ret_value_type, lhs_value_type>::value,
                  "Value types of RetType and LhsType are different!");

    static_assert(std::is_same<lhs_value_type, rhs_value_type>::value,
                  "Value types of LhsType and RhsType are different!");

    static RetType generate(const lhs_base_type& lhs, const rhs_base_type& rhs)
    {
        RetType ret;
        ret.set_is_running(false);
        ret.set_is_on_stack(false);
        ret.set_is_transient(lhs.get_is_transient() && rhs.get_is_transient());
        ret.laps  = std::min(lhs.laps, rhs.laps);
        ret.value = (lhs.value + rhs.value);
        ret.accum = (lhs.accum + rhs.accum);
        return ret;
    }

    template <typename Func, typename... Args>
    static RetType generate(const lhs_base_type& lhs, const rhs_base_type& rhs,
                            const Func& func, Args&&... args)
    {
        RetType ret(std::forward<Args>(args)...);
        ret.set_is_running(false);
        ret.set_is_on_stack(false);
        ret.set_is_transient(lhs.get_is_transient() && rhs.get_is_transient());
        ret.laps  = std::min(lhs.laps, rhs.laps);
        ret.value = func(lhs.value, rhs.value);
        ret.accum = func(lhs.accum, rhs.accum);
        return ret;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
