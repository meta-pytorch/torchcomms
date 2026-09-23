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

#include "timemory/components/base.hpp"
#include "timemory/components/papi/backends.hpp"
#include "timemory/components/papi/papi_common.hpp"
#include "timemory/components/papi/papi_config.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types/fini.hpp"
#include "timemory/operations/types/init.hpp"
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
/// \struct tim::component::papi_rate_tuple
/// \tparam RateT Component whose value will be the divisor for all the hardware counters
/// \tparam EventTypes Compile-time constant list of PAPI event identifiers
///
/// \brief This component pairs a \ref tim::component::papi_tuple with a component which
/// will provide an interval over which the hardware counters will be reported, e.g. if
/// `RateT` is \ref tim::component::wall_clock, the reported values will be the
/// hardware-counters w.r.t. the wall-clock time. If `RateT` is \ref
/// tim::component::cpu_clock, the reported values will be the hardware counters w.r.t.
/// the cpu time.
///
/// \code{.cpp}
/// // the "Instructions" alias below explicitly collects the total instructions per
/// second,
/// // the number of load instructions per second, the number of store instructions per
/// second using Instructions = papi_rate_tuple<wall_clock, PAPI_TOT_INS, PAPI_LD_INS,
/// PAPI_SR_INS>;
///
/// Instructions inst{};
/// inst.start();
/// ...
/// inst.stop();
/// std::vector<double> data = inst.get();
///
/// \endcode
template <typename RateT, int... EventTypes>
struct papi_rate_tuple
: public base<papi_rate_tuple<RateT, EventTypes...>,
              std::pair<papi_tuple<EventTypes...>, RateT>>
, private papi_common<papi_tuple<EventTypes...>>
{
    static_assert(concepts::is_component<RateT>::value,
                  "Error! rate_type must be a component");

    using size_type                   = std::size_t;
    static const size_type num_events = sizeof...(EventTypes);

    using tuple_type   = papi_tuple<EventTypes...>;
    using rate_type    = RateT;
    using value_type   = std::pair<tuple_type, rate_type>;
    using this_type    = papi_rate_tuple<RateT, EventTypes...>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using common_type  = papi_common<tuple_type>;

    template <typename Tp>
    using array_t = std::array<Tp, num_events>;

    friend struct operation::record<common_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

public:
    static void configure() { tuple_type::configure(); }
    static void initialize() { tuple_type::initialize(); }
    static void finalize() { tuple_type::finalize(); }

    static void global_init()
    {
        operation::init<tuple_type>{}(
            operation::mode_constant<operation::init_mode::global>{});
        operation::init<rate_type>{}(
            operation::mode_constant<operation::init_mode::global>{});
    }
    static void global_finalize()
    {
        /*operation::fini<tuple_type>{}(
            operation::mode_constant<operation::fini_mode::global>{});
        operation::fini<rate_type>{}(
            operation::mode_constant<operation::fini_mode::global>{});*/
    }

    static void thread_init()
    {
        operation::init<tuple_type>{}(
            operation::mode_constant<operation::init_mode::thread>{});
        operation::init<rate_type>{}(
            operation::mode_constant<operation::init_mode::thread>{});
    }
    static void thread_finalize()
    {
        /*operation::fini<tuple_type>{}(
            operation::mode_constant<operation::fini_mode::thread>{});
        operation::fini<rate_type>{}(
            operation::mode_constant<operation::fini_mode::thread>{});*/
    }

    static std::string label()
    {
        return tuple_type::label() + "_" + properties<rate_type>::id() + "_rate";
    }
    static std::string description()
    {
        return std::string(
                   "Reports the rate of the given set of HW counters w.r.t. the ") +
               rate_type::label() + " component";
    }

public:
    TIMEMORY_DEFAULT_OBJECT(papi_rate_tuple)

    using base_type::load;

    void start()
    {
        value.first.start();
        value.second.start();
    }

    void stop()
    {
        value.second.stop();
        value.first.stop();
    }

    this_type& operator+=(const this_type& rhs)
    {
        value.first += rhs.value.first;
        value.second += rhs.value.second;
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        value.first -= rhs.value.first;
        value.second -= rhs.value.second;
        return *this;
    }

    template <typename Tp = double>
    auto get() const
    {
        auto _val = value.first.template get<Tp>();
        for(auto& itr : _val)
            itr /= value.second.get();
        return _val;
    }

    auto label_array() const
    {
        auto arr = value.first.label_array();
        for(auto& itr : arr)
            itr += " per " + rate_type::get_display_unit();
        return arr;
    }

    auto description_array() const
    {
        auto arr = value.first.description_array();
        for(auto& itr : arr)
            itr += " Rate";
        return arr;
    }

    auto display_unit_array() const
    {
        auto arr = value.first.label_array();
        for(auto& itr : arr)
        {
            if(itr.empty())
                itr = "1";
            itr += "/" + rate_type::display_unit();
        }
        return arr;
    }

    auto unit_array() const
    {
        auto _units = value.first.unit_array();
        for(auto& itr : _units)
            itr /= rate_type::unit();
        return _units;
    }

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        // output the metrics
        auto _value = obj.get();
        auto _label = obj.label_array();
        auto _disp  = obj.display_unit_array();
        auto _prec  = this_type::get_precision();
        auto _width = this_type::get_width();
        auto _flags = this_type::get_format_flags();

        for(size_t i = 0; i < _value.size(); ++i)
        {
            std::stringstream ss_value;
            std::stringstream ss_extra;
            ss_value.setf(_flags);
            ss_value << std::setw(_width) << std::setprecision(_prec) << _value.at(i);
            if(!_disp.at(i).empty())
            {
                ss_extra << " " << _disp.at(i);
            }
            else if(!_label.at(i).empty())
            {
                ss_extra << " " << _label.at(i);
            }
            os << ss_value.str() << ss_extra.str();
            if(i + 1 < _value.size())
                os << ", ";
        }
        return os;
    }

protected:
    using base_type::accum;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    friend struct base<this_type, value_type>;
    friend class impl::storage<this_type,
                               trait::uses_value_storage<this_type, value_type>::value>;
};
}  // namespace component
}  // namespace tim
