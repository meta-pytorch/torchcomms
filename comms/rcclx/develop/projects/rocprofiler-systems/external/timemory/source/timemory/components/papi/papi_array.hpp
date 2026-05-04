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

#include "timemory/backends/papi.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/papi/backends.hpp"
#include "timemory/components/papi/papi_common.hpp"
#include "timemory/components/papi/papi_config.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
template <size_t MaxNumEvents>
struct papi_array
: public base<papi_array<MaxNumEvents>, std::array<long long, MaxNumEvents>>
, private papi_common<void>
{
    using size_type    = size_t;
    using event_list   = std::vector<int>;
    using value_type   = std::array<long long, MaxNumEvents>;
    using entry_type   = typename value_type::value_type;
    using this_type    = papi_array<MaxNumEvents>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using common_type  = papi_common<void>;

    static constexpr size_t event_count_max = MaxNumEvents;
    static const short      precision       = 3;
    static const short      width           = 8;

    template <typename Tp>
    using array_t = std::array<Tp, MaxNumEvents>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

    //----------------------------------------------------------------------------------//

    static auto& get_initializer()
    {
        static auto _v = common_type::initializer_t{};
        return _v;
    }
    static void configure(papi_config* _cfg = common_type::get_config())
    {
        if(_cfg && trait::runtime_enabled<this_type>::get())
        {
            auto&& _orig      = std::move(_cfg->initializer);
            _cfg->initializer = get_initializer();
            _cfg->initialize();
            _cfg->initializer = std::move(_orig);
        }
    }
    static void initialize(papi_config* _cfg = common_type::get_config())
    {
        configure(_cfg);
    }
    static void shutdown(papi_config* _cfg = common_type::get_config())
    {
        if(_cfg)
            _cfg->finalize();
    }

    static void thread_init() { configure(); }
    static void thread_finalize() { shutdown(); }

    //----------------------------------------------------------------------------------//

    papi_array()                      = default;
    ~papi_array()                     = default;
    papi_array(const papi_array&)     = default;
    papi_array(papi_array&&) noexcept = default;
    papi_array& operator=(const papi_array&) = default;
    papi_array& operator=(papi_array&&) noexcept = default;

    using base_type::load;

    size_t      size() const { return (m_config) ? m_config->size : 0; }
    const auto* get_config() const { return m_config; }

    //----------------------------------------------------------------------------------//

    static value_type record(int _event_set)
    {
        value_type read_value{};
        read_value.fill(0);
        if(_event_set != PAPI_NULL)
            papi::read(_event_set, read_value.data());
        return read_value;
    }

    static value_type record(papi_config* _cfg)
    {
        return record((_cfg) ? _cfg->event_set : PAPI_NULL);
    }

    static value_type record()
    {
        const auto& _cfg = common_type::get_config();
        return record((_cfg) ? _cfg->event_set : PAPI_NULL);
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp = double>
    std::vector<Tp> get() const
    {
        auto  values = std::vector<Tp>{};
        auto& _data  = load();
        values.reserve(_data.size());
        for(auto& itr : _data)
            values.emplace_back(itr);
        values.resize(size());
        return values;
    }

    //----------------------------------------------------------------------------------//
    // sample
    //
    void sample()
    {
        if(!m_config)
            return;

        value = record(m_config);
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        if(!m_config)
            return;

        m_config->start();
        value = record(m_config);
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        if(!m_config)
            return;

        using namespace tim::component::operators;
        value = (record(m_config) - value);
        accum += value;

        m_config->stop();
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        value += rhs.value;
        accum += rhs.accum;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator/=(size_t _val)
    {
        for(size_type i = 0; i < size(); ++i)
            accum[i] /= _val;
        for(size_type i = 0; i < size(); ++i)
            value[i] /= _val;
        return *this;
    }

    //----------------------------------------------------------------------------------//

protected:
    using base_type::accum;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;
    papi_config* m_config = common_type::get_config();

    friend struct base<this_type, value_type>;
    friend class impl::storage<this_type,
                               trait::uses_value_storage<this_type, value_type>::value>;

public:
    //==================================================================================//
    //
    //      data representation
    //
    //==================================================================================//

    static std::string label()
    {
        const auto& _cfg       = common_type::get_config();
        auto        _event_set = (_cfg) ? _cfg->event_set : PAPI_NULL;
        if(_event_set != PAPI_NULL)
            return "papi_array" + std::to_string(_event_set);
        return "papi_array";
    }

    static std::string description() { return "Fixed-size array of PAPI HW counters"; }

    entry_type get_display(int evt_type) const { return accum.at(evt_type); }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum));
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        auto _data = get<double>();
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("repr_data", _data),
           cereal::make_nvp("value", value), cereal::make_nvp("accum", accum),
           cereal::make_nvp("display", _data));
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    std::vector<std::string> label_array() const
    {
        const auto& _cfg = get_config();
        return (_cfg) ? _cfg->labels : std::vector<std::string>{};
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    std::vector<std::string> description_array() const
    {
        const auto& _cfg = get_config();
        return (_cfg) ? _cfg->descriptions : std::vector<std::string>{};
    }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    std::vector<std::string> display_unit_array() const
    {
        const auto& _cfg = get_config();
        return (_cfg) ? _cfg->display_units : std::vector<std::string>{};
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    std::vector<int64_t> unit_array() const
    {
        const auto& _cfg = get_config();
        return (_cfg) ? _cfg->units : std::vector<int64_t>{};
    }

    //----------------------------------------------------------------------------------//

    string_t get_display() const
    {
        auto events = m_config->event_names;
        if(events.empty())
            return "";
        auto val          = load();
        auto _get_display = [&](std::ostream& os, size_type idx) {
            auto     _obj_value = val[idx];
            auto     _evt_type  = events[idx];
            string_t _label     = papi::get_event_info(_evt_type).short_descr;
            string_t _disp      = papi::get_event_info(_evt_type).units;
            auto     _prec      = base_type::get_precision();
            auto     _width     = base_type::get_width();
            auto     _flags     = base_type::get_format_flags();

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
            _get_display(ss, i);
            if(i + 1 < events.size())
                ss << ", ";
        }
        return ss.str();
    }

    //----------------------------------------------------------------------------------//

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        if(obj.size() == 0)
            return os;
        // output the metrics
        auto _value = obj.get_display();
        auto _label = this_type::get_label();
        auto _disp  = this_type::display_unit();
        auto _prec  = this_type::get_precision();
        auto _width = this_type::get_width();
        auto _flags = this_type::get_format_flags();

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
};
}  // namespace component
}  // namespace tim
