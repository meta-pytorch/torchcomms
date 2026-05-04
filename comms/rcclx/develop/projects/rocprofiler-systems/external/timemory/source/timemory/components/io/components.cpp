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
 * \file timemory/components/io/components.hpp
 * \brief Implementation of the io component(s)
 */

#ifndef TIMEMORY_COMPONENTS_IO_CPP_
#    define TIMEMORY_COMPONENTS_IO_CPP_
#endif

#include "timemory/components/io/macros.hpp"

#ifndef TIMEMORY_COMPONENTS_IO_HPP_
#    include "timemory/components/io/components.hpp"
#endif

#include "timemory/components/base.hpp"
#include "timemory/components/io/backends.hpp"
#include "timemory/components/io/types.hpp"
#include "timemory/components/timing/backends.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
TIMEMORY_IO_INLINE
std::string
read_char::description()
{
    return "Number of bytes which this task has caused to be read from storage. Sum "
           "of bytes which this process passed to read() and pread(). Not disk IO.";
}

TIMEMORY_IO_INLINE
read_char::unit_type
read_char::unit()
{
    return std::pair<double, double>{ units::megabyte,
                                      static_cast<double>(units::megabyte) / units::sec };
}

TIMEMORY_IO_INLINE
std::vector<std::string>
read_char::display_unit_array()
{
    return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                     std::get<1>(get_display_unit()) };
}

TIMEMORY_IO_INLINE
std::vector<std::string>
read_char::label_array()
{
    return std::vector<std::string>{ label(), "read_rate" };
}

TIMEMORY_IO_INLINE
read_char::display_unit_type
read_char::display_unit()
{
    return display_unit_type{ "MB", "MB/sec" };
}

TIMEMORY_IO_INLINE
read_char::unit_type
read_char::unit_array()
{
    return unit();
}

TIMEMORY_IO_INLINE
std::vector<std::string>
read_char::description_array()
{
    return std::vector<std::string>{ "Number of char read", "Rate of char read" };
}

TIMEMORY_IO_INLINE
int64_t
read_char::get_timestamp()
{
    return tim::get_clock_real_now<int64_t, std::nano>();
}

TIMEMORY_IO_INLINE
read_char::value_type
read_char::record()
{
    return value_type(get_char_read(), get_timestamp());
}

TIMEMORY_IO_INLINE
int64_t
read_char::get_timing_unit()
{
    auto _value = units::sec;
    if(settings::timing_units().length() > 0)
        _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
    return _value;
}

TIMEMORY_IO_INLINE
std::string
read_char::get_display() const
{
    std::stringstream ss;
    std::stringstream ssv;
    std::stringstream ssr;
    auto              _prec  = base_type::get_precision();
    auto              _width = base_type::get_width();
    auto              _flags = base_type::get_format_flags();
    auto              _disp  = get_display_unit();

    auto _val = get();

    ssv.setf(_flags);
    ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
    if(!std::get<0>(_disp).empty())
        ssv << " " << std::get<0>(_disp);

    ssr.setf(_flags);
    ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
    if(!std::get<1>(_disp).empty())
        ssr << " " << std::get<1>(_disp);

    ss << ssv.str() << ", " << ssr.str();
    ss << " rchar";
    return ss.str();
}

TIMEMORY_IO_INLINE
read_char::result_type
read_char::get() const
{
    auto val = base_type::load();

    double data  = std::get<0>(val);
    double delta = std::get<1>(val);

    delta /= static_cast<double>(std::nano::den);
    delta *= get_timing_unit();

    double rate = 0.0;
    if(delta != 0.0)
        rate = data / delta;

    if(laps > 0)
        rate *= laps;

    data /= std::get<0>(get_unit());
    rate /= std::get<0>(get_unit());

    if(!std::isfinite(rate))
        rate = 0.0;

    return result_type(data, rate);
}

TIMEMORY_IO_INLINE
void
read_char::start()
{
    value = record();
}

TIMEMORY_IO_INLINE
void
read_char::stop()
{
    using namespace tim::component::operators;
    auto diff         = (record() - value);
    std::get<0>(diff) = std::abs(std::get<0>(diff));
    accum += (value = diff);
}

TIMEMORY_IO_INLINE
read_char::unit_type
read_char::get_unit()
{
    auto  _instance = this_type::unit();
    auto& _mem      = std::get<0>(_instance);
    auto& _rate     = std::get<1>(_instance);

    if(settings::memory_units().length() > 0)
        _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

    if(settings::timing_units().length() > 0)
    {
        auto _timing_val = std::get<1>(units::get_timing_unit(settings::timing_units()));
        _rate            = _mem / (_timing_val);
    }

    const auto factor = static_cast<double>(std::nano::den);
    unit_type  _tmp   = _instance;
    std::get<1>(_tmp) *= factor;

    return _tmp;
}

TIMEMORY_IO_INLINE
read_char::display_unit_type
read_char::get_display_unit()
{
    display_unit_type _instance = this_type::display_unit();
    auto&             _mem      = std::get<0>(_instance);
    auto&             _rate     = std::get<1>(_instance);

    if(settings::memory_units().length() > 0)
        _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

    if(settings::timing_units().length() > 0)
    {
        auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
        _rate      = mpl::apply<std::string>::join('/', _mem, _tval);
    }
    else if(settings::memory_units().length() > 0)
    {
        _rate = mpl::apply<std::string>::join('/', _mem, "sec");
    }

    return _instance;
}

TIMEMORY_IO_INLINE
void
read_char::sample()
{
    std::get<0>(accum) = std::get<0>(value) =
        std::max<int64_t>(std::get<0>(value), get_char_read());
}

TIMEMORY_IO_INLINE
void
read_char::sample(const cache_type& _cache)
{
    std::get<0>(accum) = std::get<0>(value) =
        std::max<int64_t>(std::get<0>(value), _cache.get_char_read());
}

TIMEMORY_IO_INLINE
read_char::value_type
read_char::record(const cache_type& _cache)
{
    return value_type{ _cache.get_char_read(), get_timestamp() };
}

TIMEMORY_IO_INLINE
void
read_char::start(const cache_type& _cache)
{
    value = record(_cache);
}

TIMEMORY_IO_INLINE
void
read_char::stop(const cache_type& _cache)
{
    using namespace tim::component::operators;
    auto diff         = (record(_cache) - value);
    std::get<0>(diff) = std::abs(std::get<0>(diff));
    accum += (value = diff);
}

//--------------------------------------------------------------------------------------//

TIMEMORY_IO_INLINE
std::string
written_char::description()
{
    return "Number of bytes which this task has caused, or shall cause to be written "
           "to disk. Similar caveats to read_char.";
}

TIMEMORY_IO_INLINE
written_char::unit_type
written_char::unit()
{
    return result_type{ { units::megabyte,
                          static_cast<double>(units::megabyte) / units::sec } };
}

TIMEMORY_IO_INLINE
std::vector<std::string>
written_char::display_unit_array()
{
    return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                     std::get<1>(get_display_unit()) };
}

TIMEMORY_IO_INLINE
std::vector<std::string>
written_char::label_array()
{
    return std::vector<std::string>{ label(), "written_rate" };
}

TIMEMORY_IO_INLINE
written_char::display_unit_type
written_char::display_unit()
{
    return display_unit_type{ { "MB", "MB/sec" } };
}

TIMEMORY_IO_INLINE
written_char::unit_type
written_char::unit_array()
{
    return unit();
}

TIMEMORY_IO_INLINE
std::vector<std::string>
written_char::description_array()
{
    return std::vector<std::string>{ "Number of char written", "Rate of char written" };
}

TIMEMORY_IO_INLINE
int64_t
written_char::get_timestamp()
{
    return tim::get_clock_real_now<int64_t, std::nano>();
}

TIMEMORY_IO_INLINE
written_char::value_type
written_char::record()
{
    return value_type{ { get_char_written(), get_timestamp() } };
}

TIMEMORY_IO_INLINE
int64_t
written_char::get_timing_unit()
{
    auto _value = units::sec;
    if(settings::timing_units().length() > 0)
        _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
    return _value;
}

TIMEMORY_IO_INLINE
std::string
written_char::get_display() const
{
    std::stringstream ss;
    std::stringstream ssv;
    std::stringstream ssr;
    auto              _prec  = base_type::get_precision();
    auto              _width = base_type::get_width();
    auto              _flags = base_type::get_format_flags();
    auto              _disp  = get_display_unit();

    auto _val = get();

    ssv.setf(_flags);
    ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
    if(!std::get<0>(_disp).empty())
        ssv << " " << std::get<0>(_disp);

    ssr.setf(_flags);
    ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
    if(!std::get<1>(_disp).empty())
        ssr << " " << std::get<1>(_disp);

    ss << ssv.str() << ", " << ssr.str();
    ss << " wchar";
    return ss.str();
}

TIMEMORY_IO_INLINE
written_char::result_type
written_char::get() const
{
    auto val = base_type::load();

    double data  = std::get<0>(val);
    double delta = std::get<1>(val);

    delta /= static_cast<double>(std::nano::den);
    delta *= get_timing_unit();

    double rate = 0.0;
    if(delta != 0.0)
        rate = data / delta;

    if(laps > 0)
        rate *= laps;

    data /= std::get<0>(get_unit());
    rate /= std::get<0>(get_unit());

    if(!std::isfinite(rate))
        rate = 0.0;

    return result_type{ { data, rate } };
}

TIMEMORY_IO_INLINE
void
written_char::start()
{
    value = record();
}

TIMEMORY_IO_INLINE
void
written_char::stop()
{
    using namespace tim::component::operators;
    auto diff         = (record() - value);
    std::get<0>(diff) = std::abs(std::get<0>(diff));
    accum += (value = diff);
}

TIMEMORY_IO_INLINE
written_char::unit_type
written_char::get_unit()
{
    auto  _instance = this_type::unit();
    auto& _mem      = std::get<0>(_instance);
    auto& _rate     = std::get<1>(_instance);

    if(settings::memory_units().length() > 0)
        _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

    if(settings::timing_units().length() > 0)
    {
        auto _timing_val = std::get<1>(units::get_timing_unit(settings::timing_units()));
        _rate            = _mem / (_timing_val);
    }

    const auto factor = static_cast<double>(std::nano::den);
    unit_type  _tmp   = _instance;
    std::get<1>(_tmp) *= factor;

    return _tmp;
}

TIMEMORY_IO_INLINE
written_char::display_unit_type
written_char::get_display_unit()
{
    display_unit_type _instance = this_type::display_unit();
    auto&             _mem      = std::get<0>(_instance);
    auto&             _rate     = std::get<1>(_instance);

    if(settings::memory_units().length() > 0)
        _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

    if(settings::timing_units().length() > 0)
    {
        auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
        _rate      = mpl::apply<std::string>::join('/', _mem, _tval);
    }
    else if(settings::memory_units().length() > 0)
    {
        _rate = mpl::apply<std::string>::join('/', _mem, "sec");
    }

    return _instance;
}

TIMEMORY_IO_INLINE
void
written_char::sample()
{
    std::get<0>(accum) = std::get<0>(value) =
        std::max<int64_t>(std::get<0>(value), get_char_written());
}

TIMEMORY_IO_INLINE
void
written_char::sample(const cache_type& _cache)
{
    std::get<0>(accum) = std::get<0>(value) =
        std::max<int64_t>(std::get<0>(value), _cache.get_char_written());
}

TIMEMORY_IO_INLINE
written_char::value_type
written_char::record(const cache_type& _cache)
{
    return value_type{ { _cache.get_char_written(), get_timestamp() } };
}

TIMEMORY_IO_INLINE
void
written_char::start(const cache_type& _cache)
{
    value = record(_cache);
}

TIMEMORY_IO_INLINE
void
written_char::stop(const cache_type& _cache)
{
    using namespace tim::component::operators;
    auto diff         = (record(_cache) - value);
    std::get<0>(diff) = std::abs(std::get<0>(diff));
    accum += (value = diff);
}

//--------------------------------------------------------------------------------------//

TIMEMORY_IO_INLINE
std::string
read_bytes::description()
{
    return "Number of bytes which this process really did cause to be fetched from "
           "the storage layer";
}

TIMEMORY_IO_INLINE
read_bytes::unit_type
read_bytes::unit()
{
    return std::pair<double, double>{ units::megabyte,
                                      static_cast<double>(units::megabyte) / units::sec };
}

TIMEMORY_IO_INLINE
std::vector<std::string>
read_bytes::display_unit_array()
{
    return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                     std::get<1>(get_display_unit()) };
}

TIMEMORY_IO_INLINE
std::vector<std::string>
read_bytes::label_array()
{
    return std::vector<std::string>{ label(), "read_rate" };
}

TIMEMORY_IO_INLINE
read_bytes::display_unit_type
read_bytes::display_unit()
{
    return display_unit_type{ "MB", "MB/sec" };
}

TIMEMORY_IO_INLINE
read_bytes::unit_type
read_bytes::unit_array()
{
    return unit();
}

TIMEMORY_IO_INLINE
std::vector<std::string>
read_bytes::description_array()
{
    return std::vector<std::string>{ "Number of bytes read", "Rate of bytes read" };
}

TIMEMORY_IO_INLINE
int64_t
read_bytes::get_timestamp()
{
    return tim::get_clock_real_now<int64_t, std::nano>();
}

TIMEMORY_IO_INLINE
read_bytes::value_type
read_bytes::record()
{
    return value_type(get_bytes_read(), get_timestamp());
}

TIMEMORY_IO_INLINE
int64_t
read_bytes::get_timing_unit()
{
    auto _value = units::sec;
    if(settings::timing_units().length() > 0)
        _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
    return _value;
}

TIMEMORY_IO_INLINE
std::string
read_bytes::get_display() const
{
    std::stringstream ss;
    std::stringstream ssv;
    std::stringstream ssr;
    auto              _prec  = base_type::get_precision();
    auto              _width = base_type::get_width();
    auto              _flags = base_type::get_format_flags();
    auto              _disp  = get_display_unit();

    auto _val = get();

    ssv.setf(_flags);
    ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
    if(!std::get<0>(_disp).empty())
        ssv << " " << std::get<0>(_disp);

    ssr.setf(_flags);
    ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
    if(!std::get<1>(_disp).empty())
        ssr << " " << std::get<1>(_disp);

    ss << ssv.str() << ", " << ssr.str();
    ss << " read_bytes";
    return ss.str();
}

TIMEMORY_IO_INLINE
read_bytes::result_type
read_bytes::get() const
{
    auto val = base_type::load();

    double data  = std::get<0>(val);
    double delta = std::get<1>(val);

    delta /= static_cast<double>(std::nano::den);
    delta *= get_timing_unit();

    double rate = 0.0;
    if(delta != 0.0)
        rate = data / delta;

    if(laps > 0)
        rate *= laps;

    data /= std::get<0>(get_unit());
    rate /= std::get<0>(get_unit());

    if(!std::isfinite(rate))
        rate = 0.0;

    return result_type(data, rate);
}

TIMEMORY_IO_INLINE
void
read_bytes::start()
{
    value = record();
}

TIMEMORY_IO_INLINE
void
read_bytes::stop()
{
    using namespace tim::component::operators;
    auto diff         = (record() - value);
    std::get<0>(diff) = std::abs(std::get<0>(diff));
    accum += (value = diff);
}

TIMEMORY_IO_INLINE
read_bytes::unit_type
read_bytes::get_unit()
{
    auto  _instance = this_type::unit();
    auto& _mem      = std::get<0>(_instance);
    auto& _rate     = std::get<1>(_instance);

    if(settings::memory_units().length() > 0)
        _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

    if(settings::timing_units().length() > 0)
    {
        auto _timing_val = std::get<1>(units::get_timing_unit(settings::timing_units()));
        _rate            = _mem / (_timing_val);
    }

    const auto factor = static_cast<double>(std::nano::den);
    unit_type  _tmp   = _instance;
    std::get<1>(_tmp) *= factor;

    return _tmp;
}

TIMEMORY_IO_INLINE
read_bytes::display_unit_type
read_bytes::get_display_unit()
{
    display_unit_type _instance = this_type::display_unit();
    auto&             _mem      = std::get<0>(_instance);
    auto&             _rate     = std::get<1>(_instance);

    if(settings::memory_units().length() > 0)
        _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

    if(settings::timing_units().length() > 0)
    {
        auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
        _rate      = mpl::apply<std::string>::join('/', _mem, _tval);
    }
    else if(settings::memory_units().length() > 0)
    {
        _rate = mpl::apply<std::string>::join('/', _mem, "sec");
    }

    return _instance;
}

TIMEMORY_IO_INLINE
void
read_bytes::sample()
{
    std::get<0>(accum) = std::get<0>(value) =
        std::max<int64_t>(std::get<0>(value), get_bytes_read());
}

TIMEMORY_IO_INLINE
void
read_bytes::sample(const cache_type& _cache)
{
    std::get<0>(accum) = std::get<0>(value) =
        std::max<int64_t>(std::get<0>(value), _cache.get_bytes_read());
}

TIMEMORY_IO_INLINE
read_bytes::value_type
read_bytes::record(const cache_type& _cache)
{
    return value_type{ _cache.get_bytes_read(), get_timestamp() };
}

TIMEMORY_IO_INLINE
void
read_bytes::start(const cache_type& _cache)
{
    value = record(_cache);
}

TIMEMORY_IO_INLINE
void
read_bytes::stop(const cache_type& _cache)
{
    using namespace tim::component::operators;
    auto diff         = (record(_cache) - value);
    std::get<0>(diff) = std::abs(std::get<0>(diff));
    accum += (value = diff);
}

//--------------------------------------------------------------------------------------//

TIMEMORY_IO_INLINE
std::string
written_bytes::description()
{
    return "Number of bytes sent to the storage layer";
}

TIMEMORY_IO_INLINE
written_bytes::result_type
written_bytes::unit()
{
    return result_type{ { units::megabyte,
                          static_cast<double>(units::megabyte) / units::sec } };
}

TIMEMORY_IO_INLINE
std::vector<std::string>
written_bytes::display_unit_array()
{
    return std::vector<std::string>{ std::get<0>(get_display_unit()),
                                     std::get<1>(get_display_unit()) };
}

TIMEMORY_IO_INLINE
std::vector<std::string>
written_bytes::label_array()
{
    return std::vector<std::string>{ label(), "written_rate" };
}

TIMEMORY_IO_INLINE
written_bytes::display_unit_type
written_bytes::display_unit()
{
    return display_unit_type{ { "MB", "MB/sec" } };
}

TIMEMORY_IO_INLINE
written_bytes::unit_type
written_bytes::unit_array()
{
    return unit();
}

TIMEMORY_IO_INLINE
std::vector<std::string>
written_bytes::description_array()
{
    return std::vector<std::string>{ "Number of bytes written", "Rate of bytes written" };
}

TIMEMORY_IO_INLINE
int64_t
written_bytes::get_timestamp()
{
    return tim::get_clock_real_now<int64_t, std::nano>();
}

TIMEMORY_IO_INLINE
written_bytes::value_type
written_bytes::record()
{
    return value_type{ { get_bytes_written(), get_timestamp() } };
}

TIMEMORY_IO_INLINE
int64_t
written_bytes::get_timing_unit()
{
    auto _value = units::sec;
    if(settings::timing_units().length() > 0)
        _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
    return _value;
}

TIMEMORY_IO_INLINE
std::string
written_bytes::get_display() const
{
    std::stringstream ss;
    std::stringstream ssv;
    std::stringstream ssr;
    auto              _prec  = base_type::get_precision();
    auto              _width = base_type::get_width();
    auto              _flags = base_type::get_format_flags();
    auto              _disp  = get_display_unit();

    auto _val = get();

    ssv.setf(_flags);
    ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
    if(!std::get<0>(_disp).empty())
        ssv << " " << std::get<0>(_disp);

    ssr.setf(_flags);
    ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
    if(!std::get<1>(_disp).empty())
        ssr << " " << std::get<1>(_disp);

    ss << ssv.str() << ", " << ssr.str();
    ss << " write_bytes";
    return ss.str();
}

TIMEMORY_IO_INLINE
written_bytes::result_type
written_bytes::get() const
{
    auto val = base_type::load();

    double data  = std::get<0>(val);
    double delta = std::get<1>(val);

    delta /= static_cast<double>(std::nano::den);
    delta *= get_timing_unit();

    double rate = 0.0;
    if(delta != 0.0)
        rate = data / delta;

    if(laps > 0)
        rate *= laps;

    data /= std::get<0>(get_unit());
    rate /= std::get<0>(get_unit());

    if(!std::isfinite(rate))
        rate = 0.0;

    return result_type{ { data, rate } };
}

TIMEMORY_IO_INLINE
void
written_bytes::start()
{
    value = record();
}

TIMEMORY_IO_INLINE
void
written_bytes::stop()
{
    using namespace tim::component::operators;
    auto diff         = (record() - value);
    std::get<0>(diff) = std::abs(std::get<0>(diff));
    accum += (value = diff);
}

TIMEMORY_IO_INLINE
written_bytes::unit_type
written_bytes::get_unit()
{
    auto  _instance = this_type::unit();
    auto& _mem      = std::get<0>(_instance);
    auto& _rate     = std::get<1>(_instance);

    if(settings::memory_units().length() > 0)
        _mem = std::get<1>(units::get_memory_unit(settings::memory_units()));

    if(settings::timing_units().length() > 0)
    {
        auto _timing_val = std::get<1>(units::get_timing_unit(settings::timing_units()));
        _rate            = _mem / (_timing_val);
    }

    const auto factor = static_cast<double>(std::nano::den);
    unit_type  _tmp   = _instance;
    std::get<1>(_tmp) *= factor;

    return _tmp;
}

TIMEMORY_IO_INLINE
written_bytes::display_unit_type
written_bytes::get_display_unit()
{
    display_unit_type _instance = this_type::display_unit();
    auto&             _mem      = std::get<0>(_instance);
    auto&             _rate     = std::get<1>(_instance);

    if(settings::memory_units().length() > 0)
        _mem = std::get<0>(units::get_memory_unit(settings::memory_units()));

    if(settings::timing_units().length() > 0)
    {
        auto _tval = std::get<0>(units::get_timing_unit(settings::timing_units()));
        _rate      = mpl::apply<std::string>::join('/', _mem, _tval);
    }
    else if(settings::memory_units().length() > 0)
    {
        _rate = mpl::apply<std::string>::join('/', _mem, "sec");
    }

    return _instance;
}

TIMEMORY_IO_INLINE
void
written_bytes::sample()
{
    std::get<0>(accum) = std::get<0>(value) =
        std::max<int64_t>(std::get<0>(value), get_bytes_written());
}

TIMEMORY_IO_INLINE
void
written_bytes::sample(const cache_type& _cache)
{
    std::get<0>(accum) = std::get<0>(value) =
        std::max<int64_t>(std::get<0>(value), _cache.get_bytes_written());
}

TIMEMORY_IO_INLINE
written_bytes::value_type
written_bytes::record(const cache_type& _cache)
{
    return value_type{ { _cache.get_bytes_written(), get_timestamp() } };
}

TIMEMORY_IO_INLINE
void
written_bytes::start(const cache_type& _cache)
{
    value = record(_cache);
}

TIMEMORY_IO_INLINE
void
written_bytes::stop(const cache_type& _cache)
{
    using namespace tim::component::operators;
    auto diff         = (record(_cache) - value);
    std::get<0>(diff) = std::abs(std::get<0>(diff));
    accum += (value = diff);
}
}  // namespace component
}  // namespace tim
