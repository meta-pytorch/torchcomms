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

#ifndef TIMEMORY_COMPONENT_TIMESTAMP_TIMESTAMP_CPP_
#define TIMEMORY_COMPONENT_TIMESTAMP_TIMESTAMP_CPP_

#include "timemory/components/timestamp/types.hpp"

#if !defined(TIMEMORY_TIMESTAMP_COMPONENT_HEADER_MODE)
#    include "timemory/components/timestamp/timestamp.hpp"
#    define TIMEMORY_COMPONENT_TIMESTAMP_INLINE
#else
#    define TIMEMORY_COMPONENT_TIMESTAMP_INLINE inline
#endif

namespace tim
{
namespace component
{
TIMEMORY_COMPONENT_TIMESTAMP_INLINE
std::string
timestamp::label()
{
    return "timestamp";
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
std::string
timestamp::description()
{
    return "Provides a timestamp for every sample and/or phase";
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
timestamp::value_type
timestamp::record()
{
    return std::chrono::system_clock::now();
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
std::string
timestamp::as_string(const time_point_type& _tp)
{
    char _repr[64];
    std::memset(_repr, '\0', sizeof(_repr));
    std::time_t _value = std::chrono::system_clock::to_time_t(_tp);
    // alternative: "%c %Z"
    if(std::strftime(_repr, sizeof(_repr), "%a %b %d %T %Y %Z", std::localtime(&_value)))
        return std::string{ _repr };
    return std::string{};
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
int64_t
timestamp::count()
{
    return std::chrono::duration_cast<duration_type>(record().time_since_epoch()).count();
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
void
timestamp::sample()
{
    get_reference_ts();
    base_type::set_value(record());
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
void
timestamp::start()
{
    get_reference_ts();
    base_type::set_value(record());
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
timestamp::value_type
timestamp::get() const
{
    return base_type::get_value();
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
std::string
timestamp::get_display() const
{
    return as_string(base_type::get_value());
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
timestamp&
timestamp::operator+=(const timestamp& _rhs)
{
    return (*this = _rhs);
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
timestamp&
timestamp::operator/=(const timestamp&)
{
    return *this;
}

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
timestamp& timestamp::operator/=(int64_t) { return *this; }

TIMEMORY_COMPONENT_TIMESTAMP_INLINE
timestamp::value_type
timestamp::get_reference_ts()
{
    static auto _v = record();
    return _v;
}

}  // namespace component

namespace data
{
namespace base
{
template <>
TIMEMORY_COMPONENT_TIMESTAMP_INLINE void
stream_entry::construct<timestamp_value_t>(const timestamp_value_t& val)
{
    stringstream_t ss;
    ss.setf(m_format);
    ss << std::setprecision(m_precision) << component::timestamp::as_string(val) << " / "
       << val.time_since_epoch().count();
    m_value = ss.str();
    if(settings::max_width() > 0 && m_value.length() > (size_t) settings::max_width())
    {
        //
        //  don't truncate and add ellipsis if max width is really small
        //
        if(settings::max_width() > 20)
        {
            m_value = m_value.substr(0, settings::max_width() - 3);
            m_value += "...";
        }
        else if(settings::max_width() >= 0)
        {
            m_value = m_value.substr(0, settings::max_width());
        }
    }
}
}  // namespace base
}  // namespace data
}  // namespace tim

#endif  // TIMEMORY_COMPONENT_TIMESTAMP_TIMESTAMP_CPP_
