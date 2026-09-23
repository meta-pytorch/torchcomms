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

#ifndef TIMEMORY_COMPONENTS_PAPI_PAPI_VECTOR_CPP_
#define TIMEMORY_COMPONENTS_PAPI_PAPI_VECTOR_CPP_

#include "timemory/components/papi/macros.hpp"

#if defined(TIMEMORY_PAPI_SOURCE) && TIMEMORY_PAPI_SOURCE > 0
#    include "timemory/components/papi/papi_vector.hpp"
#elif !defined(TIMEMORY_PAPI_COMPONENT_HEADER_MODE) ||                                   \
    TIMEMORY_PAPI_COMPONENT_HEADER_MODE == 0
#    include "timemory/components/papi/papi_vector.hpp"
#endif

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
papi_vector::configure(papi_config* _cfg)
{
    if(_cfg && trait::runtime_enabled<this_type>::get())
    {
        auto&& _orig      = std::move(_cfg->initializer);
        _cfg->initializer = get_initializer();
        _cfg->initialize();
        _cfg->initializer = std::move(_orig);
    }
}

TIMEMORY_PAPI_INLINE
void
papi_vector::initialize(papi_config* _cfg)
{
    configure(_cfg);
}

TIMEMORY_PAPI_INLINE
void
papi_vector::shutdown(papi_config* _cfg)
{
    _cfg->finalize();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
papi_vector::thread_init()
{
    configure(common_type::get_config());
}

TIMEMORY_PAPI_INLINE
void
papi_vector::thread_finalize()
{
    shutdown(common_type::get_config());
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector::papi_vector()
: m_config{ common_type::get_config() }
{
    value.resize(size(), 0);
    accum.resize(size(), 0);
}

TIMEMORY_PAPI_INLINE
papi_vector::papi_vector(papi_config* _cfg)
: m_config{ _cfg }
, m_config_is_common{ _cfg == common_type::get_config() }
{
    value.resize(size(), 0);
    accum.resize(size(), 0);
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
size_t
papi_vector::size() const
{
    return (m_config) ? m_config->size : 0;
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector::value_type
papi_vector::record()
{
    value_type read_value(size(), 0);
    if(m_config && m_config->is_running)
        papi::read(m_config->event_set, read_value.data());
    return read_value;
}

//----------------------------------------------------------------------------------//
// sample
//
TIMEMORY_PAPI_INLINE
void
papi_vector::sample()
{
    if(!m_config)
        return;
    value = record();
}

//----------------------------------------------------------------------------------//
// start
//
TIMEMORY_PAPI_INLINE
void
papi_vector::start()
{
    if(!m_config)
        return;

    m_config->start();
    value.resize(size(), 0);
    accum.resize(size(), 0);
    value = record();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
void
papi_vector::stop()
{
    if(!m_config)
        return;

    using namespace tim::component::operators;
    value = (record() - value);
    accum += value;
    m_config->stop();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector&
papi_vector::operator+=(const papi_vector& rhs)
{
    value += rhs.value;
    accum += rhs.accum;
    return *this;
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector&
papi_vector::operator-=(const papi_vector& rhs)
{
    value -= rhs.value;
    accum -= rhs.accum;
    return *this;
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::string
papi_vector::label()
{
    const auto& _cfg       = common_type::get_config();
    auto        _event_set = (_cfg) ? _cfg->event_set : PAPI_NULL;
    return "papi_vector" + std::to_string(_event_set);
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::string
papi_vector::description()
{
    return "Dynamically allocated array of PAPI HW counters";
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
papi_vector::entry_type
papi_vector::get_display(int _idx) const
{
    return (_idx < static_cast<int>(accum.size())) ? accum.at(_idx) : entry_type{ 0 };
}

//----------------------------------------------------------------------------------//
// array of descriptions
//
TIMEMORY_PAPI_INLINE
std::vector<std::string>
papi_vector::label_array() const
{
    return (m_config) ? m_config->labels : std::vector<std::string>{};
}

//----------------------------------------------------------------------------------//
// array of labels
//
TIMEMORY_PAPI_INLINE
std::vector<std::string>
papi_vector::description_array() const
{
    return (m_config) ? m_config->descriptions : std::vector<std::string>{};
}

//----------------------------------------------------------------------------------//
// array of unit
//
TIMEMORY_PAPI_INLINE
std::vector<std::string>
papi_vector::display_unit_array() const
{
    return (m_config) ? m_config->display_units : std::vector<std::string>{};
}

//----------------------------------------------------------------------------------//
// array of unit values
//
TIMEMORY_PAPI_INLINE
std::vector<int64_t>
papi_vector::unit_array() const
{
    return (m_config) ? m_config->units : std::vector<int64_t>{};
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::string
papi_vector::get_display() const
{
    auto events = m_config->event_names;
    if(events.empty())
        return "";
    auto val          = load();
    auto _get_display = [&](std::ostream& os, size_type idx) {
        auto        _obj_value = val.at(idx);
        auto        _evt_type  = events.at(idx);
        std::string _label     = papi::get_event_info(_evt_type).short_descr;
        std::string _disp      = papi::get_event_info(_evt_type).units;
        auto        _prec      = base_type::get_precision();
        auto        _width     = base_type::get_width();
        auto        _flags     = base_type::get_format_flags();

        std::stringstream ss;
        std::stringstream ssv;
        std::stringstream ssi;
        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << _obj_value;
        if(!_disp.empty())
            ssv << " " << _disp;
        if(!_label.empty())
            ssi << " " << _label;
        ss << ssv.str() << ssi.str();
        os << ss.str();
    };

    std::stringstream ss;
    for(size_type i = 0; i < events.size(); ++i)
    {
        try
        {
            _get_display(ss, i);
            if(i + 1 < events.size())
                ss << ", ";
        } catch(std::exception& _e)
        {
            TIMEMORY_PRINTF_WARNING(stderr, "[papi_vector][%s] %s\n", __FUNCTION__,
                                    _e.what());
        }
    }
    return ss.str();
}

//----------------------------------------------------------------------------------//

TIMEMORY_PAPI_INLINE
std::ostream&
papi_vector::write(std::ostream& os) const
{
    if(size() == 0)
        return os;
    // output the metrics
    auto _value = get_display();
    auto _label = get_label();
    auto _disp  = display_unit();
    auto _prec  = get_precision();
    auto _width = get_width();
    auto _flags = get_format_flags();

    std::stringstream ss_value;
    std::stringstream ss_extra;
    ss_value.setf(_flags);
    ss_value << std::setw(_width) << std::setprecision(_prec) << _value;
    if(!_disp.empty())
    {
        ss_extra << " " << _disp;
    }
    else if(!_label.empty())
    {
        ss_extra << " " << _label;
    }
    os << ss_value.str() << ss_extra.str();
    return os;
}
}  // namespace component
}  // namespace tim

#endif
