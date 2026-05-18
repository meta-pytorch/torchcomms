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

#pragma once

#include "timemory/components/base/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/settings/settings.hpp"

#include <ios>

namespace tim
{
namespace component
{
template <typename Tp>
struct base_format
{
    using fmtflags = std::ios_base::fmtflags;

    static short    get_precision();
    static short    get_width();
    static fmtflags get_format_flags();

    static short    set_precision(short);
    static short    set_width(short);
    static fmtflags set_format_flags(fmtflags);

private:
    enum
    {
        PRECISION_idx = 0,
        WIDTH_idx,
        FORMAT_idx,
        LAST_idx,
    };

    using value_type = std::tuple<short, short, fmtflags>;
    using array_type = std::array<bool, LAST_idx>;

    // returns current precision, width, and format
    static value_type& get_format_value();

    // returns whether values have been explicitly set
    static array_type& get_format_status();
};

template <typename Tp>
short
base_format<Tp>::get_precision()
{
    constexpr bool timing_category_v = trait::is_timing_category<Tp>::value;
    constexpr bool memory_category_v = trait::is_memory_category<Tp>::value;

    auto _v = std::get<PRECISION_idx>(get_format_value());
    if(std::get<PRECISION_idx>(get_format_status()))
        return _v;

    auto* _settings = settings::instance();
    if(_settings)
    {
        if(_settings->get_precision() >= 0)
            _v = _settings->get_precision();

        if(timing_category_v && _settings->get_timing_precision() >= 0)
        {
            _v = _settings->get_timing_precision();
        }
        else if(memory_category_v && _settings->get_memory_precision() >= 0)
        {
            _v = _settings->get_memory_precision();
        }
    }

    return _v;
}

template <typename Tp>
short
base_format<Tp>::get_width()
{
    constexpr bool timing_category_v = trait::is_timing_category<Tp>::value;
    constexpr bool memory_category_v = trait::is_memory_category<Tp>::value;

    auto _v = std::get<WIDTH_idx>(get_format_value());
    if(std::get<WIDTH_idx>(get_format_status()))
        return _v;

    auto* _settings = settings::instance();
    if(_settings)
    {
        if(_settings->get_width() >= 0)
            _v = _settings->get_width();

        if(timing_category_v && _settings->get_timing_width() >= 0)
        {
            _v = _settings->get_timing_width();
        }
        else if(memory_category_v && _settings->get_memory_width() >= 0)
        {
            _v = _settings->get_memory_width();
        }
    }

    return _v;
}

template <typename Tp>
typename base_format<Tp>::fmtflags
base_format<Tp>::get_format_flags()
{
    constexpr bool timing_category_v = trait::is_timing_category<Tp>::value;
    constexpr bool memory_category_v = trait::is_memory_category<Tp>::value;
    constexpr bool percent_units_v   = trait::uses_percent_units<Tp>::value;

    auto _v = std::get<FORMAT_idx>(get_format_value());
    if(std::get<FORMAT_idx>(get_format_status()))
        return _v;

    if(!percent_units_v &&
       (settings::scientific() || (timing_category_v && settings::timing_scientific()) ||
        (memory_category_v && settings::memory_scientific())))
    {
        _v &= (std::ios_base::fixed & std::ios_base::scientific);
        _v |= (std::ios_base::scientific);
    }

    return _v;
}

template <typename Tp>
short
base_format<Tp>::set_precision(short _new)
{
    auto _old                                    = get_precision();
    std::get<PRECISION_idx>(get_format_value())  = _new;
    std::get<PRECISION_idx>(get_format_status()) = true;
    return _old;
}

template <typename Tp>
short
base_format<Tp>::set_width(short _new)
{
    auto _old                                = get_width();
    std::get<WIDTH_idx>(get_format_value())  = _new;
    std::get<WIDTH_idx>(get_format_status()) = true;
    return _old;
}

template <typename Tp>
typename base_format<Tp>::fmtflags
base_format<Tp>::set_format_flags(fmtflags _new)
{
    auto _old                                 = get_format_flags();
    std::get<FORMAT_idx>(get_format_value())  = _new;
    std::get<FORMAT_idx>(get_format_status()) = true;
    return _old;
}

template <typename Tp>
typename base_format<Tp>::value_type&
base_format<Tp>::get_format_value()
{
    static auto _v =
        value_type{ trait::format_precision<Tp>{}(), trait::format_width<Tp>{}(),
                    trait::format_flags<Tp>{}() };
    return _v;
}

template <typename Tp>
typename base_format<Tp>::array_type&
base_format<Tp>::get_format_status()
{
    static auto _v = array_type{ false, false, false };
    return _v;
}
}  // namespace component
}  // namespace tim
