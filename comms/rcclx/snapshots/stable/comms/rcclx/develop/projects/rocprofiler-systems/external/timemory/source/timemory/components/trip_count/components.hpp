//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/**
 * \file timemory/components/trip_count/components.hpp
 * \brief Implementation of the trip_count component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/trip_count/backends.hpp"
#include "timemory/components/trip_count/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
/// \struct tim::component::trip_count
/// \brief Records the number of invocations. This is the most lightweight metric
/// available since it only increments an integer and never records any statistics.
/// If dynamic instrumentation is used and the overhead is significant, it is recommended
/// to set this as the only component (-d trip_count) and then use the regex exclude
/// option (-E) to remove any non-critical function calls which have very high
/// trip-counts.
//
struct trip_count : public base<trip_count>
{
    using value_type = int64_t;
    using this_type  = trip_count;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "trip_count"; }
    static std::string description() { return "Counts number of invocations"; }
    static value_type  record() { return 1; }

    TIMEMORY_NODISCARD value_type get() const { return value; }
    TIMEMORY_NODISCARD value_type get_display() const { return get(); }

    void start() { value += 1; }
    void stop() {}
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
