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

#include "timemory/components/timing/backends.hpp"
#include "timemory/operations/types/base_printer.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/units.hpp"

#include <cstdint>
#include <iosfwd>
#include <iostream>

namespace tim
{
namespace component
{
struct ert_timer
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using this_type  = ert_timer;
    using fmtflags   = std::ios_base::fmtflags;

    static const short    precision = 3;
    static const short    width     = 8;
    static const fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    static std::string get_label() { return "wall_clock"; }
    static std::string get_description() { return "wall-clock timer for ERT"; }
    static int64_t     get_unit() { return units::sec; }
    static std::string get_display_unit() { return units::time_repr(units::sec); }

    static auto get_width() { return width; }
    static auto get_precision() { return precision; }
    static auto get_format_flags() { return format_flags; }

    static value_type record() noexcept
    {
        return tim::get_clock_real_now<int64_t, ratio_t>();
    }

    auto load() const { return value; }

    double get() const noexcept { return load() / static_cast<double>(get_unit()); }

    auto get_display() const noexcept { return get(); }

    void start() noexcept { value = record(); }
    void stop() noexcept
    {
        value = (record() - value);
        ++laps;
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("laps", laps));
        ar(cereal::make_nvp("value", value));
        ar(cereal::make_nvp("accum", value));
    }

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        obj.print(os);
        return os;
    }

    void print(std::ostream& os) const { operation::base_printer<this_type>(os, *this); }

private:
    int64_t    laps  = 0;
    value_type value = 0;
};
}  // namespace component
}  // namespace tim
