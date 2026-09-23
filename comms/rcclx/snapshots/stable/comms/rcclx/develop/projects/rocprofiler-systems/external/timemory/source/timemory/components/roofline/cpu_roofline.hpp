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

#include "timemory/backends/papi.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/papi/components.hpp"
#include "timemory/components/roofline/backends.hpp"
#include "timemory/components/roofline/types.hpp"
#include "timemory/components/timing/wall_clock.hpp"
#include "timemory/ert/configuration.hpp"
#include "timemory/ert/extern.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/demangle.hpp"

#include <array>
#include <memory>
#include <numeric>
#include <utility>

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
// this computes the numerator of the roofline for a given set of PAPI counters.
// e.g. for FLOPS roofline (floating point operations / second:
//
//  single precision:
//              cpu_roofline<float>
//
//  double precision:
//              cpu_roofline<double>
//
//  generic:
//              cpu_roofline<T, ...>
//
/// \struct tim::component::cpu_roofline
/// \tparam Types Variadic list of data types for roofline analysis
///
/// \brief Combines hardware counters and timers and executes the empirical roofline
/// toolkit during application termination to estimate the peak possible performance for
/// the machine
///
template <typename... Types>
struct cpu_roofline
: public base<cpu_roofline<Types...>, std::pair<std::vector<long long>, double>>
{
    static_assert(!is_one_of<gpu::fp16_t, std::tuple<Types...>>::value,
                  "Error! No CPU roofline support for gpu::fp16_t");

    using size_type    = std::size_t;
    using event_type   = std::vector<int>;
    using array_type   = std::vector<long long>;
    using data_type    = long long*;
    using value_type   = std::pair<array_type, double>;
    using this_type    = cpu_roofline<Types...>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using record_type  = std::function<value_type()>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

    using unit_type         = typename trait::units<this_type>::type;
    using display_unit_type = typename trait::units<this_type>::display_type;

    using device_t    = device::cpu;
    using count_type  = wall_clock;
    using ratio_t     = typename count_type::ratio_t;
    using types_tuple = std::tuple<Types...>;

    using ert_data_t     = ert::exec_data<count_type>;
    using ert_data_ptr_t = std::shared_ptr<ert_data_t>;

    // short-hand for variadic expansion
    template <typename Tp>
    using ert_config_type = ert::configuration<device_t, Tp, count_type>;
    template <typename Tp>
    using ert_counter_type = ert::counter<device_t, Tp, count_type>;
    template <typename Tp>
    using ert_executor_type = ert::executor<device_t, Tp, count_type>;
    template <typename Tp>
    using ert_callback_type = ert::callback<ert_executor_type<Tp>>;

    // variadic expansion for ERT types
    using ert_config_t   = std::tuple<ert_config_type<Types>...>;
    using ert_counter_t  = std::tuple<ert_counter_type<Types>...>;
    using ert_executor_t = std::tuple<ert_executor_type<Types>...>;
    using ert_callback_t = std::tuple<ert_callback_type<Types>...>;

    static_assert(std::tuple_size<ert_config_t>::value ==
                      std::tuple_size<types_tuple>::value,
                  "Error! ert_config_t size does not match types_tuple size!");

    using iterator       = typename array_type::iterator;
    using const_iterator = typename array_type::const_iterator;

    static const short precision = 3;
    static const short width     = 8;

    //----------------------------------------------------------------------------------//

    // collection mode, AI (arithmetic intensity) is the load/store: PAPI_LST_INS
    enum class MODE
    {
        OP,
        AI
    };

    //----------------------------------------------------------------------------------//

    using strvec_t          = std::vector<std::string>;
    using intvec_t          = std::vector<int>;
    using events_callback_t = std::function<intvec_t(const MODE&)>;

    //----------------------------------------------------------------------------------//
    /// replace this callback to add in custom HW counters
    static events_callback_t& get_events_callback()
    {
        static events_callback_t _instance = [](const MODE&) { return intvec_t{}; };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// set to false to suppress adding predefined enumerations
    static bool& use_predefined_enums()
    {
        static bool _instance = true;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static MODE& event_mode()
    {
        auto&& _get = []() {
            auto&& aslc = [](std::string str) {
                for(auto& itr : str)
                    itr = tolower(itr);
                return str;
            };
            // check the standard variable
            std::string _env = aslc(settings::cpu_roofline_mode());
            if(_env.empty())
                _env = aslc(settings::roofline_mode());
            auto _val =
                (_env == "op" || _env == "hw" || _env == "counters")
                    ? MODE::OP
                    : ((_env == "ai" || _env == "ac" || _env == "activity") ? MODE::AI
                                                                            : MODE::OP);
            if(settings::verbose() > 1 || settings::debug())
            {
                std::cerr << "[" << demangle<this_type>()
                          << "]> roofline mode: " << ((_val == MODE::OP) ? "op" : "ai")
                          << std::endl;
            }
            return _val;
        };

        static MODE _instance = _get();
        if(!is_configured())
            _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static ert_config_t& get_finalizer()
    {
        static ert_config_t _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static ert_data_ptr_t get_ert_data()
    {
        static ert_data_ptr_t _instance = std::make_shared<ert_data_t>();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static event_type get_events()
    {
        static auto _instance = []() {
            event_type _events;
            auto       _mode = event_mode();
            if(_mode == MODE::OP)
            {
                //
                // add in user callback events BEFORE presets based on type so that
                // the user can override the counters being used
                //
                auto _extra_events = get_events_callback()(_mode);
                for(const auto& itr : _extra_events)
                    _events.push_back(itr);

                //
                //  add some presets based on data types
                //
                if(use_predefined_enums())
                {
                    if(is_one_of<float, types_tuple>::value)
                        _events.push_back(PAPI_SP_OPS);
                    if(is_one_of<double, types_tuple>::value)
                        _events.push_back(PAPI_DP_OPS);
                }
            }
            else if(_mode == MODE::AI)
            {
                //
                //  add the load/store hardware counter
                //
                if(use_predefined_enums())
                {
                    _events.push_back(PAPI_LD_INS);
                    _events.push_back(PAPI_SR_INS);
                    _events.push_back(PAPI_LST_INS);
                    _events.push_back(PAPI_TOT_INS);
                }
                //
                // add in user callback events AFTER load/store so that load/store
                // instructions are always counted
                //
                auto _extra_events = get_events_callback()(_mode);
                for(const auto& itr : _extra_events)
                    _events.push_back(itr);
            }

            return _events;
        }();

        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void configure()
    {
        if(!is_configured())
        {
            if(settings::debug() || settings::verbose() > 1)
                TIMEMORY_PRINT_HERE("%s", "configuring cpu_roofline");

            // do this BEFORE setting is_configured to true to ensure mode is updated
            // properly
            auto _events    = get_events();
            is_configured() = true;
            for(auto&& itr : _events)
                papi_config::add_global_event(itr);
            papi_vector::configure();
        }
    }

    //----------------------------------------------------------------------------------//

    static void global_init()
    {
        if(settings::debug() || settings::verbose() > 2)
            TIMEMORY_PRINT_HERE("%s", "global initialization of cpu_roofline");
        configure();
    }

    //----------------------------------------------------------------------------------//

    static void thread_init()
    {
        if(settings::debug() || settings::verbose() > 2)
            TIMEMORY_PRINT_HERE("%s", "thread initialization of cpu_roofline");
        configure();
    }

    //----------------------------------------------------------------------------------//

    static void thread_finalize() {}

    //----------------------------------------------------------------------------------//

    template <typename Tp, typename FuncT>
    static void set_executor_callback(FuncT&& f)
    {
        ert_executor_type<Tp>::get_callback() = std::forward<FuncT>(f);
    }

    //----------------------------------------------------------------------------------//

    static void global_finalize(storage_type* _store)
    {
        // query environment for whether this is part of CI test
        // auto ci = get_env<bool>("CONTINUOUS_INTEGRATION", false);
        if(_store && _store->size() > 0)
        {
            // run roofline peak generation
            auto ert_config = get_finalizer();
            auto ert_data   = get_ert_data();
            mpl::apply<void>::access<ert_executor_t>(ert_config, ert_data);
            if(ert_data && (settings::verbose() > 1 || settings::debug()))
                std::cout << *(ert_data) << std::endl;
        }
    }

    //----------------------------------------------------------------------------------//

    template <typename Archive>
    static void extra_serialization(Archive& ar)
    {
        auto _ert_data = get_ert_data();
        if(!_ert_data)  // for input
            _ert_data = std::make_shared<ert_data_t>();
        ar(cereal::make_nvp("roofline", *_ert_data));
    }

    //----------------------------------------------------------------------------------//

    static std::string get_mode_string()
    {
        return (event_mode() == MODE::OP) ? "op" : "ai";
    }

    //----------------------------------------------------------------------------------//

    static std::string get_type_string()
    {
        return mpl::apply<std::string>::join('_', demangle(typeid(Types).name())...);
    }

    //----------------------------------------------------------------------------------//

    static unit_type unit()
    {
        return (event_mode() == MODE::OP) ? (1.0 / count_type::unit()) : 1.0;
    }

    //----------------------------------------------------------------------------------//

    display_unit_type display_unit()
    {
        auto _units = m_papi_vector->display_unit_array();
        _units.push_back(m_wall_clock->display_unit());
        return _units;
    }

    //----------------------------------------------------------------------------------//

    unit_type get_unit() { return unit(); }

    //----------------------------------------------------------------------------------//

    display_unit_type get_display_unit() { return display_unit(); }

    //----------------------------------------------------------------------------------//

    static std::string label()
    {
        if(settings::roofline_type_labels_cpu() || settings::roofline_type_labels())
        {
            return std::string("cpu_roofline_") + get_type_string() + "_" +
                   get_mode_string();
        }
        {
            return std::string("cpu_roofline_") + get_mode_string();
        }
    }

    //----------------------------------------------------------------------------------//

    static std::string description()
    {
        return "Model used to provide performance relative to the peak possible "
               "performance on a CPU architecture.";
    }

    //----------------------------------------------------------------------------------//

    value_type record()
    {
        auto hwcount  = m_papi_vector->record();
        auto duration = m_wall_clock->record();
        return value_type(hwcount, duration);
    }

public:
    cpu_roofline()
    : base_type()
    {
        configure();
        m_papi_vector                        = std::make_shared<papi_vector>();
        m_wall_clock                         = std::make_shared<wall_clock>();
        std::tie(value.second, accum.second) = std::make_pair(0, 0);
    }

    ~cpu_roofline()                           = default;
    cpu_roofline(const cpu_roofline& rhs)     = default;
    cpu_roofline(cpu_roofline&& rhs) noexcept = default;
    cpu_roofline& operator=(const cpu_roofline&) = default;
    cpu_roofline& operator=(cpu_roofline&&) noexcept = default;

    //----------------------------------------------------------------------------------//

    TIMEMORY_NODISCARD std::vector<double> get() const
    {
        auto _data = m_papi_vector->get();
        _data.push_back(m_wall_clock->get());
        return _data;
    }

    //----------------------------------------------------------------------------------//

    void start()
    {
        m_wall_clock->start();
        m_papi_vector->start();
        value = value_type{ m_papi_vector->get_value(), m_wall_clock->get_value() };
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        m_papi_vector->stop();
        m_wall_clock->stop();
        value = value_type{ m_papi_vector->get_value(), m_wall_clock->get_value() };
        accum += value_type{ m_papi_vector->get_accum(), m_wall_clock->get_accum() };
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        if(rhs.value.first.size() > value.first.size())
            value.first.resize(rhs.value.first.size());
        if(rhs.accum.first.size() > accum.first.size())
            accum.first.resize(rhs.accum.first.size());
        value += rhs.value;
        accum += rhs.accum;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        if(rhs.value.first.size() > value.first.size())
            value.first.resize(rhs.value.first.size());
        if(rhs.accum.first.size() > accum.first.size())
            accum.first.resize(rhs.accum.first.size());
        value -= rhs.value;
        accum -= rhs.accum;
        return *this;
    }

    using base_type::load;

protected:
    using base_type::accum;
    using base_type::laps;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    friend struct base<this_type, value_type>;
    friend class impl::storage<this_type,
                               trait::uses_value_storage<this_type, value_type>::value>;

public:
    //==================================================================================//
    //
    //      representation as a string
    //
    //==================================================================================//

    TIMEMORY_NODISCARD std::vector<double> get_display() const { return get(); }

    //----------------------------------------------------------------------------------//

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        using namespace tim::stl::ostream;

        // output the time
        auto&             _obj = obj.load();
        std::stringstream sst;
        auto              t_value = _obj.second;
        auto              t_label = count_type::get_label();
        auto              t_disp  = count_type::get_display_unit();
        auto              t_prec  = count_type::get_precision();
        auto              t_width = count_type::get_width();
        auto              t_flags = count_type::get_format_flags();

        sst.setf(t_flags);
        sst << std::setw(t_width) << std::setprecision(t_prec) << t_value;
        if(!t_disp.empty())
            sst << " " << t_disp;
        if(!t_label.empty())
            sst << " " << t_label;
        sst << ", ";

        auto _prec  = count_type::get_precision();
        auto _width = this_type::get_width();
        auto _flags = count_type::get_format_flags();

        // output the roofline metric
        auto _value = obj.get();
        auto _label = obj.label_array();
        auto _disp  = obj.display_unit_array();

#if defined(DEBUG)
        if(settings::debug())
        {
            std::cout << "value: " << _value << std::endl;
            std::cout << "label: " << _label << std::endl;
            std::cout << "displ: " << _disp << std::endl;
        }
#endif
        assert(_value.size() <= _label.size());
        assert(_value.size() <= _disp.size());

        auto n = _label.size();
        for(size_t i = 0; i < n; ++i)
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
            os << sst.str() << ss_value.str() << ss_extra.str();
            if(i + 1 < n)
                os << ", ";
        }

        return os;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        auto _disp  = get_display();
        auto labels = label_array();

        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("labels", labels),
           cereal::make_nvp("papi_vector", m_papi_vector));
        ar(cereal::make_nvp("value", value));
        ar(cereal::make_nvp("accum", accum));
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        auto _disp  = get_display();
        auto labels = label_array();

        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("display", _disp),
           cereal::make_nvp("mode", get_mode_string()),
           cereal::make_nvp("type", get_type_string()),
           cereal::make_nvp("labels", labels),
           cereal::make_nvp("papi_vector", m_papi_vector));

        auto data = get();
        ar.setNextName("repr_data");
        ar.startNode();
        auto litr = labels.begin();
        auto ditr = data.begin();
        for(; litr != labels.end() && ditr != data.end(); ++litr, ++ditr)
            ar(cereal::make_nvp(*litr, double(*ditr)));
        ar.finishNode();

        ar(cereal::make_nvp("value", value));
        ar(cereal::make_nvp("accum", accum));
        // ar(cereal::make_nvp("units", unit_array()));
        // ar(cereal::make_nvp("display_units", display_unit_array()));
    }

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    TIMEMORY_NODISCARD strvec_t label_array() const
    {
        strvec_t arr = m_papi_vector->label_array();
        arr.push_back("Runtime");
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    TIMEMORY_NODISCARD strvec_t description_array() const
    {
        strvec_t arr = m_papi_vector->description_array();
        arr.push_back("Runtime");
        return arr;
    }

    //----------------------------------------------------------------------------------//
    //
    TIMEMORY_NODISCARD strvec_t display_unit_array() const
    {
        strvec_t arr = m_papi_vector->display_unit_array();
        arr.push_back(count_type::get_display_unit());
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    TIMEMORY_NODISCARD std::vector<int64_t> unit_array() const
    {
        auto arr = m_papi_vector->unit_array();
        arr.push_back(count_type::get_unit());
        return arr;
    }

private:
    //----------------------------------------------------------------------------------//
    // these are needed after the global label array is destroyed
    //
    std::shared_ptr<papi_vector> m_papi_vector{ nullptr };
    std::shared_ptr<wall_clock>  m_wall_clock{ nullptr };

public:
    //----------------------------------------------------------------------------------//

    static void cleanup() {}

private:
    //----------------------------------------------------------------------------------//

    static bool& is_configured()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
