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

#include "timemory/components/base.hpp"
#include "timemory/components/cupti/components.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/roofline/backends.hpp"
#include "timemory/components/roofline/types.hpp"
#include "timemory/components/timing/wall_clock.hpp"
#include "timemory/ert/configuration.hpp"
#include "timemory/ert/extern.hpp"
#include "timemory/operations/types/fini.hpp"
#include "timemory/operations/types/init.hpp"
#include "timemory/settings/declaration.hpp"

#include <array>
#include <cassert>
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
//              gpu_roofline<float>
//
//  double precision:
//              gpu_roofline<double>
//
//
/// \struct tim::component::gpu_roofline
/// \tparam Types Variadic list of data types for roofline analysis
///
/// \brief Combines hardware counters and timers and executes the empirical roofline
/// toolkit during application termination to estimate the peak possible performance for
/// the machine
///
template <typename... Types>
struct gpu_roofline
: public base<gpu_roofline<Types...>, std::tuple<typename cupti_activity::value_type,
                                                 typename cupti_counters::value_type>>
{
    using value_type   = std::tuple<typename cupti_activity::value_type,
                                  typename cupti_counters::value_type>;
    using this_type    = gpu_roofline<Types...>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;

    using size_type     = std::size_t;
    using counters_type = cupti_counters;
    using activity_type = cupti_activity;
    using device_t      = device::gpu;
    using result_type   = std::vector<double>;
    using label_type    = std::vector<std::string>;
    using count_type    = wall_clock;
    using types_tuple   = std::tuple<Types...>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

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

    static const short precision = 3;
    static const short width     = 8;

    //----------------------------------------------------------------------------------//
    // collection mode, COUNTERS is the HW counting, ACTIVITY in the runtime measurements
    enum class MODE
    {
        COUNTERS,
        ACTIVITY
    };

    //----------------------------------------------------------------------------------//

    using strvec_t           = std::vector<std::string>;
    using events_callback_t  = std::function<strvec_t()>;
    using metrics_callback_t = events_callback_t;

    //----------------------------------------------------------------------------------//

    static events_callback_t& get_events_callback()
    {
        static events_callback_t _instance = []() { return strvec_t{}; };
        return _instance;
    }

    static metrics_callback_t& get_metrics_callback()
    {
        static metrics_callback_t _instance = []() { return strvec_t{}; };
        return _instance;
    }

public:
    //----------------------------------------------------------------------------------//

    static MODE& event_mode()
    {
        auto&& _get = [=]() {
            auto&& aslc = [](std::string str) {
                for(auto& itr : str)
                    itr = tolower(itr);
                return str;
            };

            // check the standard variable
            std::string _env = aslc(settings::gpu_roofline_mode());
            if(_env.empty())
                _env = aslc(settings::roofline_mode());
            return (_env == "op" || _env == "hw" || _env == "counters")
                       ? MODE::COUNTERS
                       : ((_env == "ai" || _env == "ac" || _env == "activity")
                              ? MODE::ACTIVITY
                              : MODE::COUNTERS);
        };

        static MODE _instance = _get();
        if(!is_configured())
            _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void configure(const MODE& _mode, int _device = 0)
    {
        if(is_configured())
            return;
        is_configured() = true;

        event_mode() = _mode;

        if(event_mode() == MODE::ACTIVITY)
        {
            get_labels() = { std::string("runtime") };
        }
        else
        {
            strvec_t events  = { "global_load", "global_store" };
            strvec_t metrics = { "ldst_executed" };
#if defined(TIMEMORY_USE_CUDA_HALF)
            if(is_one_of<cuda::fp16_t, types_tuple>::value)
            {
                metrics.push_back("flop_count_hp");
            }
#endif

            if(is_one_of<float, types_tuple>::value)
            {
                metrics.push_back("flop_count_sp");
            }

            if(is_one_of<double, types_tuple>::value)
            {
                metrics.push_back("flop_count_dp");
            }

            // integer
            if(mpl::is_one_of_integral<types_tuple>::value ||
               settings::instruction_roofline())
            {
                for(const auto* itr :
                    { "ipc", "inst_executed", "inst_integer", "inst_fp_64", "inst_fp_32",
                      "inst_fp_16", "local_load_transactions_per_request",
                      "local_store_transactions_per_request",
                      "shared_load_transactions_per_request",
                      "shared_store_transactions_per_request",
                      "gld_transactions_per_request", "gst_transactions_per_request",
                      "inst_executed_global_reductions", "inst_executed_global_stores",
                      "inst_executed_global_loads", "inst_executed_local_loads",
                      "inst_executed_local_stores", "inst_executed_shared_loads",
                      "inst_executed_shared_stores" })
                    metrics.emplace_back(itr);
            }

            // add in extra events
            auto _extra_events = get_events_callback()();
            for(const auto& itr : _extra_events)
                events.push_back(itr);

            // add in extra metrics
            auto _extra_metrics = get_metrics_callback()();
            for(const auto& itr : _extra_metrics)
                metrics.push_back(itr);

            auto _get_unique = [](const strvec_t& _vec) {
                std::set<std::string> _set;
                for(const auto& itr : _vec)
                    _set.insert(itr);
                strvec_t _ret;
                for(const auto& itr : _set)
                    _ret.push_back(itr);
                return _ret;
            };

            metrics = _get_unique(metrics);
            events  = _get_unique(events);

            counters_type::configure(_device, events, metrics);
            get_labels() = counters_type::label_array();
        }
    }

    //----------------------------------------------------------------------------------//

    static void configure()
    {
        if(!is_configured())
            configure(event_mode());
    }

    //----------------------------------------------------------------------------------//

    static std::string get_mode_string()
    {
        return (event_mode() == MODE::COUNTERS) ? "counters" : "activity";
    }

    //----------------------------------------------------------------------------------//

    static std::string get_type_string()
    {
        return mpl::apply<std::string>::join('_', demangle(typeid(Types).name())...);
    }

    //----------------------------------------------------------------------------------//

    static ert_config_t& get_finalizer()
    {
        static ert_config_t _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static ert_data_ptr_t& get_ert_data()
    {
        static ert_data_ptr_t _instance = std::make_shared<ert_data_t>();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void global_init()
    {
        if(event_mode() == MODE::ACTIVITY)
        {
            operation::init<activity_type>{}(
                operation::mode_constant<operation::init_mode::global>{});
        }
        else
        {
            operation::init<counters_type>{}(
                operation::mode_constant<operation::init_mode::global>{});
        }
    }

    //----------------------------------------------------------------------------------//

    template <typename Tp, typename FuncT>
    static void set_executor_callback(FuncT&& f)
    {
        ert_executor_type<Tp>::get_callback() = std::forward<FuncT>(f);
    }

    //----------------------------------------------------------------------------------//

    static void global_finalize(storage_type* _store)
    {
        // disable the activity/counters before running ERT
        if(event_mode() == MODE::ACTIVITY)
        {
            operation::fini<activity_type>{}(
                operation::mode_constant<operation::fini_mode::global>{});
        }
        else
        {
            operation::fini<counters_type>{}(
                operation::mode_constant<operation::fini_mode::global>{});
        }

        // run ERT
        if(_store && _store->size() > 0)
        {
            assert(_store->is_finalizing());
            // run roofline peak generation
            auto ert_config = get_finalizer();
            auto ert_data   = get_ert_data();
            mpl::apply<void>::access<ert_executor_t>(ert_config, ert_data);
            if(ert_data && (settings::verbose() > 0 || settings::debug()))
                std::cout << *(ert_data) << std::endl;
        }
    }

    //----------------------------------------------------------------------------------//

    static void thread_init() {}
    static void thread_finalize() {}

    //----------------------------------------------------------------------------------//

    template <typename Archive>
    static void extra_serialization(Archive& ar)
    {
        auto& _ert_data = get_ert_data();
        if(!_ert_data)  // for input
            _ert_data = std::make_shared<ert_data_t>();
        ar(cereal::make_nvp("roofline", *_ert_data));
    }

    //----------------------------------------------------------------------------------//

    static int64_t unit()
    {
        if(event_mode() == MODE::ACTIVITY)
            return activity_type::unit();
        return counters_type::unit();
    }

    static std::string label()
    {
        if(settings::roofline_type_labels_gpu() || settings::roofline_type_labels())
        {
            auto ret = std::string("gpu_roofline_") + get_type_string() + "_" +
                       get_mode_string();
            // erase consecutive underscores
            while(ret.find("__") != std::string::npos)
                ret.erase(ret.find("__"), 1);
            return ret;
        }

        return std::string("gpu_roofline_") + get_mode_string();
    }

    static std::string description()
    {
        return "Model used to provide performance relative to the peak possible "
               "performance on a GPU architecture.";
    }

    static std::string display_unit()
    {
        if(event_mode() == MODE::ACTIVITY)
            return activity_type::display_unit();
        return counters_type::display_unit();
    }

    //----------------------------------------------------------------------------------//

    static value_type record()
    {
        value_type tmp;
        switch(event_mode())
        {
            case MODE::ACTIVITY: std::get<0>(tmp) = activity_type::record(); break;
            case MODE::COUNTERS: std::get<1>(tmp) = counters_type::record(); break;
            default: break;
        }
        return tmp;
    }

private:
    //----------------------------------------------------------------------------------//

    static bool& is_configured()
    {
        static bool _instance = false;
        return _instance;
    }

public:
    gpu_roofline() { configure(); }
    ~gpu_roofline() = default;

    gpu_roofline(const gpu_roofline& rhs)
    : base_type(rhs)
    , m_data(rhs.m_data)
    {}

    gpu_roofline& operator=(const gpu_roofline& rhs)
    {
        if(this != &rhs)
        {
            base_type::operator=(rhs);
            m_data             = rhs.m_data;
        }
        return *this;
    }

    gpu_roofline(gpu_roofline&&) noexcept = default;
    gpu_roofline& operator=(gpu_roofline&&) noexcept = default;

    //----------------------------------------------------------------------------------//

    TIMEMORY_NODISCARD result_type get() const
    {
        switch(event_mode())
        {
            case MODE::ACTIVITY: return result_type({ m_data.activity->get() });
            case MODE::COUNTERS: return m_data.counters->get();
            default: break;
        }
        return result_type{};
    }

    //----------------------------------------------------------------------------------//

    void start()
    {
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                m_data.activity->start();
                std::get<0>(value) = m_data.activity->get_value();
                break;
            }
            case MODE::COUNTERS:
            {
                m_data.counters->start();
                std::get<1>(value) = m_data.counters->get_value();
                break;
            }
        }
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        using namespace tim::component::operators;
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                m_data.activity->stop();
                std::get<0>(accum) = m_data.activity->get_accum();
                std::get<0>(value) = m_data.activity->get_value();
                break;
            }
            case MODE::COUNTERS:
            {
                m_data.counters->stop();
                std::get<1>(accum) = m_data.counters->get_accum();
                std::get<1>(value) = m_data.counters->get_value();
                break;
            }
        }
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                *m_data.activity += *rhs.m_data.activity;
                std::get<0>(accum) = m_data.activity->get_accum();
                std::get<0>(value) = m_data.activity->get_value();
                break;
            }
            case MODE::COUNTERS:
            {
                *m_data.counters += *rhs.m_data.counters;
                std::get<1>(accum) = m_data.counters->get_accum();
                std::get<1>(value) = m_data.counters->get_value();
                break;
            }
        }
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                *m_data.activity -= *rhs.m_data.activity;
                std::get<0>(accum) = m_data.activity->get_accum();
                std::get<0>(value) = m_data.activity->get_value();
                break;
            }
            case MODE::COUNTERS:
            {
                *m_data.counters -= *rhs.m_data.counters;
                std::get<1>(accum) = m_data.counters->get_accum();
                std::get<1>(value) = m_data.counters->get_value();
                break;
            }
        }
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const value_type& rhs)
    {
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                *m_data.activity += std::get<0>(rhs);
                std::get<0>(accum) = m_data.activity->get_accum();
                std::get<0>(value) = m_data.activity->get_value();
                break;
            }
            case MODE::COUNTERS:
            {
                *m_data.counters += std::get<1>(rhs);
                std::get<1>(accum) = m_data.counters->get_accum();
                std::get<1>(value) = m_data.counters->get_value();
                break;
            }
        }
        return *this;
    }

    //----------------------------------------------------------------------------------//

    using activity_value_type = typename cupti_activity::value_type;
    using counters_value_type = typename cupti_counters::value_type;
    using secondary_type      = std::unordered_multimap<std::string, value_type>;

    //----------------------------------------------------------------------------------//

    TIMEMORY_NODISCARD secondary_type get_secondary() const
    {
        secondary_type ret;
        switch(event_mode())
        {
            case MODE::ACTIVITY:
            {
                auto&& _tmp = m_data.activity->get_secondary();
                for(auto&& itr : _tmp)
                {
                    ret.insert(
                        { itr.first, value_type{ itr.second, counters_value_type{} } });
                }
                break;
            }
            case MODE::COUNTERS:
            {
                auto&& _tmp = m_data.counters->get_secondary();
                for(auto&& itr : _tmp)
                {
                    ret.insert(
                        { itr.first, value_type{ activity_value_type{}, itr.second } });
                }
                break;
            }
        }
        return ret;
    }

    //----------------------------------------------------------------------------------//

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
    //
    TIMEMORY_NODISCARD string_t get_display() const
    {
        std::stringstream ss;
        if(event_mode() == MODE::COUNTERS)
        {
            return m_data.counters->get_display();
        }
        {
            ss << m_data.activity->get_display();
        }
        return ss.str();
    }

    //----------------------------------------------------------------------------------//
    //
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        os << as_string(obj.get_display());
        return os;
    }

    //----------------------------------------------------------------------------------//
    //
    static label_type label_array() { return this_type::get_labels(); }

    //----------------------------------------------------------------------------------//
    //
    static label_type display_unit_array()
    {
        const auto& _labels = get_labels();
        return label_type(_labels.size(), this_type::display_unit());
    }

private:
    //----------------------------------------------------------------------------------//
    //
    static string_t as_string(const string_t& _value)
    {
        auto _label = this_type::get_label();
        auto _disp  = this_type::get_display_unit();
        auto _prec  = this_type::get_precision();
        auto _width = this_type::get_width();
        auto _flags = this_type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;
        if(!_disp.empty() && !trait::custom_unit_printing<this_type>::value)
            ss_extra << " " << _disp;
        if(!_label.empty() && !trait::custom_label_printing<this_type>::value)
            ss_extra << " " << _label;

        std::stringstream ss;
        ss << ss_value.str() << ss_extra.str();
        return ss.str();
    }

private:
    static label_type& get_labels() { return *_get_labels(); }

    static label_type* _get_labels()
    {
        static auto _instance = std::make_unique<label_type>();
        return _instance.get();
    }

private:
    union cupti_data
    {
        cupti_activity* activity = nullptr;
        cupti_counters* counters;

        cupti_data()
        {
            switch(event_mode())
            {
                case MODE::ACTIVITY: activity = new cupti_activity(); break;
                case MODE::COUNTERS: counters = new cupti_counters(); break;
            }
        }

        ~cupti_data()
        {
            switch(event_mode())
            {
                case MODE::ACTIVITY: delete activity; break;
                case MODE::COUNTERS: delete counters; break;
            }
        }

        cupti_data(const cupti_data& rhs)
        {
            switch(event_mode())
            {
                case MODE::ACTIVITY: activity = new cupti_activity(*rhs.activity); break;
                case MODE::COUNTERS: counters = new cupti_counters(*rhs.counters); break;
            }
        }

        cupti_data(cupti_data&& rhs) noexcept
        {
            switch(event_mode())
            {
                case MODE::ACTIVITY:
                    activity = nullptr;
                    std::swap(activity, rhs.activity);
                    break;
                case MODE::COUNTERS:
                    counters = nullptr;
                    std::swap(counters, rhs.counters);
                    break;
            }
        }

        cupti_data& operator=(const cupti_data& rhs)
        {
            if(this == &rhs)
                return *this;
            switch(event_mode())
            {
                case MODE::ACTIVITY:
                    delete activity;
                    activity = new cupti_activity(*rhs.activity);
                    break;
                case MODE::COUNTERS:
                    delete counters;
                    counters = new cupti_counters(*rhs.counters);
                    break;
            }
            return *this;
        }

        cupti_data& operator=(cupti_data&& rhs) noexcept
        {
            if(this == &rhs)
                return *this;
            switch(event_mode())
            {
                case MODE::ACTIVITY:
                    delete activity;
                    activity = nullptr;
                    std::swap(activity, rhs.activity);
                    break;
                case MODE::COUNTERS:
                    delete counters;
                    counters = nullptr;
                    std::swap(counters, rhs.counters);
                    break;
            }
            return *this;
        }
    };

    cupti_data m_data;

public:
    //----------------------------------------------------------------------------------//

    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        auto _disp   = get_display();
        auto _data   = get();
        auto _labels = get_labels();

        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("display", _disp),
           cereal::make_nvp("mode", get_mode_string()),
           cereal::make_nvp("type", get_type_string()),
           cereal::make_nvp("labels", _labels));

        ar.setNextName("repr_data");
        ar.startNode();
        auto litr = _labels.begin();
        auto ditr = _data.begin();
        for(; litr != _labels.end() && ditr != _data.end(); ++litr, ++ditr)
            ar(cereal::make_nvp(*litr, *ditr));
        ar.finishNode();

        ar.setNextName("value");
        ar.startNode();
        ar.makeArray();
        if(event_mode() == MODE::ACTIVITY)
        {
            ar(std::get<0>(value));
        }
        else
        {
            ar(std::get<1>(value));
        }
        ar.finishNode();

        ar.setNextName("accum");
        ar.startNode();
        ar.makeArray();
        if(event_mode() == MODE::ACTIVITY)
        {
            ar(std::get<0>(accum));
        }
        else
        {
            ar(std::get<1>(accum));
        }
        ar.finishNode();
    }

    //----------------------------------------------------------------------------------//

    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        std::string              _disp;
        result_type              _data;
        std::vector<std::string> _labels;
        std::string              _mode_str;
        std::string              _type_str;

        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("display", _disp),
           cereal::make_nvp("mode", _mode_str), cereal::make_nvp("type", _type_str),
           cereal::make_nvp("labels", _labels));

        if(_mode_str == "counters")
        {
            event_mode() = MODE::COUNTERS;
        }
        else if(_mode_str == "activity")
        {
            event_mode() = MODE::ACTIVITY;
        }

        _data.resize(_labels.size());

        ar.setNextName("repr_data");
        ar.startNode();
        auto litr = _labels.begin();
        auto ditr = _data.begin();
        for(; litr != _labels.end() && ditr != _data.end(); ++litr, ++ditr)
            ar(cereal::make_nvp(*litr, *ditr));
        ar.finishNode();

        ar.setNextName("value");
        ar.startNode();
        if(event_mode() == MODE::ACTIVITY)
        {
            ar(std::get<0>(value));
        }
        else
        {
            ar(std::get<1>(value));
        }
        ar.finishNode();

        ar.setNextName("accum");
        ar.startNode();
        if(event_mode() == MODE::ACTIVITY)
        {
            ar(std::get<0>(accum));
        }
        else
        {
            ar(std::get<1>(accum));
        }
        ar.finishNode();
    }
};

}  // namespace component
}  // namespace tim
