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

#pragma once

#include "timemory/components/base/declaration.hpp"
#include "timemory/components/timing/backends.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/units.hpp"

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          Timing types
//
//--------------------------------------------------------------------------------------//
//
struct wall_clock : public base<wall_clock, int64_t>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<wall_clock, value_type>;

    static std::string label() { return "wall_clock"; }
    static std::string description()
    {
        return "Real-clock timer (i.e. wall-clock timer)";
    }
    static value_type record() noexcept
    {
        return tim::get_clock_real_now<int64_t, ratio_t>();
    }

    TIMEMORY_NODISCARD double get() const noexcept
    {
        return load() / static_cast<double>(get_unit());
    }
    TIMEMORY_NODISCARD auto get_display() const noexcept { return get(); }

    void start() noexcept { value = record(); }
    void stop() noexcept { accum += (value = (record() - value)); }
};

//--------------------------------------------------------------------------------------//
// alias for "wall_clock"
using real_clock = wall_clock;
// alias for "wall_clock" since time is a construct of our consciousness
using virtual_clock = wall_clock;
//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
