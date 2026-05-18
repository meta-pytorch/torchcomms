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

/** \file cupti.hpp
 * \headerfile cupti_counters.hpp "timemory/cupti_counters.hpp"
 * Provides implementation of CUPTI routines.
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/cupti/backends.hpp"
#include "timemory/components/cupti/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/settings/declaration.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          CUPTI hardware counters component
//
//--------------------------------------------------------------------------------------//
/// \struct tim::component::cupti_counters
/// \brief NVprof-style hardware counters via the CUpti callback API. Collecting these
/// hardware counters has a higher overhead than the new CUpti Profiling API (\ref
/// tim::component::cupti_profiler). However, there are currently some issues with nesting
/// the Profiling API and it is currently recommended to use this component for NVIDIA
/// hardware counters in timemory. The callback API / NVprof is quite specific about
/// the distinction between an "event" and a "metric". For your convenience, timemory
/// removes this distinction and events can be specified arbitrarily as metrics and
/// vice-versa and this component will sort them into their appropriate category.
/// For the full list of the available events/metrics, use `timemory-avail -H` from the
/// command-line.
///
struct cupti_counters : public base<cupti_counters, cupti::profiler::results_t>
{
    // required aliases
    using value_type = cupti::profiler::results_t;
    using this_type  = cupti_counters;
    using base_type  = base<cupti_counters, value_type>;

    // custom aliases
    using size_type        = std::size_t;
    using string_t         = std::string;
    using kernel_data_t    = cupti::result;
    using entry_type       = typename value_type::value_type;
    using results_t        = cupti::profiler::results_t;
    using kernel_results_t = cupti::profiler::kernel_results_t;

    // short-hand for vectors
    using strvec_t  = std::vector<string_t>;
    using profptr_t = std::shared_ptr<cupti::profiler>;
    // a tuple of the <devices, events, metrics>
    using tuple_type = std::tuple<int, strvec_t, strvec_t>;
    // function for setting device, metrics, and events to record
    using event_func_t  = std::function<strvec_t()>;
    using metric_func_t = std::function<strvec_t()>;
    using device_func_t = std::function<int()>;
    // function for setting all of device, metrics, and events
    using get_initializer_t = std::function<tuple_type()>;

    static const short precision = 3;
    static const short width     = 8;

    static event_func_t& get_event_initializer()
    {
        static event_func_t _instance = []() {
            return delimit(settings::cupti_events());
        };
        return _instance;
    }

    static metric_func_t& get_metric_initializer()
    {
        static metric_func_t _instance = []() {
            return delimit(settings::cupti_metrics());
        };
        return _instance;
    }

    static device_func_t& get_device_initializer()
    {
        static device_func_t _instance = []() {
            if(cuda::device_count() < 1)
                return -1;
            return settings::cupti_device();
        };
        return _instance;
    }

    static get_initializer_t& get_initializer()
    {
        static get_initializer_t _instance = []() -> tuple_type {
            return tuple_type(get_device_initializer()(), get_event_initializer()(),
                              get_metric_initializer()());
        };
        return _instance;
    }

    static void configure()
    {
        if(_get_profiler().get() == nullptr)
            init();
    }

    /// explicitly configure for a device and set of events/metrics.
    static void configure(int device, const strvec_t& events,
                          const strvec_t& metrics = {})
    {
        get_initializer() = [=]() -> tuple_type {
            return tuple_type(device, events, metrics);
        };
        if(_get_profiler().get() == nullptr)
            init();
    }

    static void global_init() { configure(); }
    static void global_finalize() { clear(); }

    static const profptr_t& get_profiler() { return _get_profiler(); }
    static const strvec_t&  get_events() { return *_get_events(); }
    static const strvec_t&  get_metrics() { return *_get_metrics(); }
    static int              get_device() { return *_get_device(); }
    static const strvec_t&  get_labels() { return *_get_labels(); }

    cupti_counters()
    {
        configure();
        auto* _labels = _get_labels();
        if(_labels)
        {
            value.resize(_labels->size());
            accum.resize(_labels->size());
            for(size_type i = 0; i < _labels->size(); ++i)
            {
                value[i].name = (*_labels)[i];
                accum[i].name = (*_labels)[i];
            }
        }
    }

    ~cupti_counters()                         = default;
    cupti_counters(const cupti_counters&)     = default;
    cupti_counters(cupti_counters&&) noexcept = default;
    cupti_counters& operator                  =(const cupti_counters& rhs)
    {
        if(this != &rhs)
        {
            base_type::operator=(rhs);
            m_kernel_value     = rhs.m_kernel_value;
            m_kernel_accum     = rhs.m_kernel_accum;
        }
        return *this;
    }
    cupti_counters& operator=(cupti_counters&&) noexcept = default;

    static int64_t unit() { return 1; }
    // leave these empty
    static string_t label() { return "cupti_counters"; }
    static string_t description() { return "Hardware counters for the CUDA API"; }
    static string_t display_unit() { return ""; }

    static value_type record()
    {
        configure();
        value_type tmp;
        auto&      _profiler = _get_profiler();
        if(!_profiler || !_get_labels())
            return tmp;
        auto& _labels = *_get_labels();
        _profiler->stop();
        if(tmp.empty())
        {
            tmp = _profiler->get_events_and_metrics(_labels);
        }
        else if(tmp.size() == _labels.size())
        {
            auto ret = _profiler->get_events_and_metrics(_labels);
            for(size_t j = 0; j < _labels.size(); ++j)
                tmp[j] += ret[j];
        }
        else
        {
            TIMEMORY_PRINTF_WARNING(
                stderr, "Warning! mis-matched size in cupti_event::%s @ %s:%i\n",
                TIMEMORY_ERROR_FUNCTION_MACRO, __FILE__, __LINE__);
        }

        return tmp;
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        value           = record();
        auto& _profiler = _get_profiler();
        if(_profiler)
        {
            m_kernel_value = _profiler->get_kernel_events_and_metrics(*_get_labels());
            _profiler->start();
        }
    }

    void stop()
    {
        using namespace stl;
        using namespace tim::component::operators;

        value_type tmp       = record();
        auto&      _profiler = _get_profiler();
        if(!_profiler)
            return;

        kernel_results_t kernel_data =
            _profiler->get_kernel_events_and_metrics(*_get_labels());
        kernel_results_t kernel_tmp = kernel_data;

        if(accum.empty())
        {
            accum = tmp;
            for(size_type i = 0; i < tmp.size(); ++i)
                accum[i] -= value[i];
        }
        else
        {
            for(size_type i = 0; i < tmp.size(); ++i)
                accum[i] += (tmp[i] - value[i]);
        }

        for(size_t i = 0; i < m_kernel_value.size(); ++i)
            kernel_tmp[i].second -= m_kernel_value[i].second;
        for(size_t i = 0; i < kernel_tmp.size(); ++i)
        {
            if(i >= m_kernel_accum.size())
            {
                m_kernel_accum.resize(i + 1, kernel_tmp[i]);
            }
            else
            {
                m_kernel_accum[i].second += kernel_tmp[i].second;
            }
        }

        value          = std::move(tmp);
        m_kernel_value = std::move(kernel_data);
    }

    TIMEMORY_NODISCARD string_t get_display() const
    {
        auto _get_display = [&](std::ostream& os, const cupti::result& obj) {
            auto _label = obj.name;
            auto _prec  = base_type::get_precision();
            auto _width = base_type::get_width();
            auto _flags = base_type::get_format_flags();

            std::stringstream ss;
            std::stringstream ssv;
            std::stringstream ssi;
            ssv.setf(_flags);
            ssv << std::setw(_width) << std::setprecision(_prec);
            cupti::print(ssv, obj.data);
            if(!_label.empty())
                ssi << " " << _label;
            ss << ssv.str() << ssi.str();
            os << ss.str();
        };

        const auto&       _data = load();
        std::stringstream ss;
        for(size_type i = 0; i < _data.size(); ++i)
        {
            _get_display(ss, _data[i]);
            if(i + 1 < _data.size())
                ss << ", ";
        }
        return ss.str();
    }

    TIMEMORY_NODISCARD std::vector<double> get() const
    {
        std::vector<double> values;
        const auto&         _data = load();
        values.reserve(_data.size());
        for(const auto& itr : _data)
            values.push_back(cupti::get<double>(itr.data));
        return values;
    }

    using secondary_type = std::unordered_multimap<std::string, value_type>;

    TIMEMORY_NODISCARD secondary_type get_secondary() const
    {
        secondary_type _data;
        for(const auto& itr : m_kernel_accum)
            _data.insert({ itr.first, itr.second });
        return _data;
    }

    template <typename Tp>
    using array_t = std::vector<Tp>;

    //----------------------------------------------------------------------------------//
    // array of descriptions
    //
    static array_t<string_t> label_array()
    {
        array_t<string_t> arr;
        auto              contains = [&](const string_t& entry) {
            return std::find(arr.begin(), arr.end(), entry) != arr.end();
        };
        auto insert = [&](const string_t& entry) {
            if(!contains(entry))
                arr.push_back(entry);
        };
        auto* _labels = _get_labels();
        if(_labels)
        {
            for(const auto& itr : *_labels)
                insert(itr);
        }
        // auto profiler = get_profiler();
        // for(const auto& itr : profiler->get_event_names())
        //     insert(itr);
        // for(const auto& itr : profiler->get_metric_names())
        //     insert(itr);
        return arr;
    }

    //----------------------------------------------------------------------------------//
    // array of labels
    //
    static array_t<string_t> description_array() { return label_array(); }

    //----------------------------------------------------------------------------------//
    // array of unit
    //
    static array_t<string_t> display_unit_array()
    {
        return array_t<string_t>(get_labels().size(), "");
    }

    //----------------------------------------------------------------------------------//
    // array of unit values
    //
    static array_t<int64_t> unit_array()
    {
        return array_t<int64_t>(get_labels().size(), 1);
    }

    this_type& operator+=(const this_type& rhs)
    {
        auto _combine = [](value_type& _data, const value_type& _other) {
            auto& _labels = *_get_labels();
            if(_data.empty())
            {
                _data = _other;
            }
            else
            {
                for(size_type i = 0; i < _labels.size(); ++i)
                    _data[i] += _other[i];
            }
        };

        _combine(value, rhs.value);
        _combine(accum, rhs.accum);
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        auto _combine = [](value_type& _data, const value_type& _other) {
            auto& _labels = *_get_labels();
            // set to other
            if(_data.empty())
                _data = _other;
            // subtract other (if data was empty, will contain zero data)
            for(size_type i = 0; i < _labels.size(); ++i)
                _data[i] -= _other[i];
        };

        _combine(value, rhs.value);
        _combine(accum, rhs.accum);
        return *this;
    }

    this_type& operator+=(const results_t& rhs)
    {
        auto _combine = [](value_type& _data, const value_type& _other) {
            if(_data.empty())
            {
                _data = _other;
            }
            else
            {
                for(size_type i = 0; i < _other.size(); ++i)
                    _data[i] += _other[i];
            }
        };

        _combine(value, rhs);
        _combine(accum, rhs);

        return *this;
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto _get = [&](const value_type& _data) {
            std::vector<double> values;
            for(const auto& itr : _data)
                values.push_back(cupti::get<double>(itr.data));
            return values;
        };
        array_t<double> _disp  = _get(accum);
        array_t<double> _value = _get(value);
        array_t<double> _accum = _get(accum);
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("repr_data", _disp),
           cereal::make_nvp("value", _value), cereal::make_nvp("accum", _accum),
           cereal::make_nvp("display", _disp));
        // ar(cereal::make_nvp("units", unit_array()),
        //   cereal::make_nvp("display_units", display_unit_array()));
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    static void extra_serialization(Archive& ar)
    {
        auto& _devices = *_get_device();
        auto& _events  = *_get_events();
        auto& _metrics = *_get_metrics();
        auto& _labels  = *_get_labels();

        ar(cereal::make_nvp("devices", _devices), cereal::make_nvp("events", _events),
           cereal::make_nvp("metrics", _metrics), cereal::make_nvp("labels", _labels));
    }

    //----------------------------------------------------------------------------------//

private:
    template <typename Tp>
    struct writer
    {
        using const_iterator = typename Tp::const_iterator;
        Tp& obj;
        writer(Tp& _obj)
        : obj(_obj)
        {}

        TIMEMORY_NODISCARD const_iterator begin() const { return obj.begin(); }
        TIMEMORY_NODISCARD const_iterator end() const { return obj.end(); }

        friend std::ostream& operator<<(std::ostream& os, const writer<Tp>& _obj)
        {
            auto sz = std::distance(_obj.begin(), _obj.end());
            for(auto itr = _obj.begin(); itr != _obj.end(); ++itr)
            {
                auto idx = std::distance(_obj.begin(), itr);
                os << (*itr);
                if(idx + 1 < sz)
                    os << ", ";
            }
            return os;
        }
    };

    static profptr_t& _get_profiler()
    {
        static profptr_t _instance = profptr_t(nullptr);
        return _instance;
    }

    static strvec_t*& _get_events()
    {
        static strvec_t* _instance = new strvec_t();
        return _instance;
    }

    static strvec_t*& _get_metrics()
    {
        static strvec_t* _instance = new strvec_t();
        return _instance;
    }

    static int*& _get_device()
    {
        static int* _instance = new int(0);
        return _instance;
    }

    static strvec_t*& _get_labels()
    {
        static strvec_t* _instance = new strvec_t();
        return _instance;
    }

    static strvec_t generate_labels()
    {
        array_t<string_t> arr;
        auto              contains = [&](const string_t& entry) {
            return std::find(arr.begin(), arr.end(), entry) != arr.end();
        };
        auto insert = [&](const string_t& entry) {
            if(!contains(entry))
                arr.push_back(entry);
        };
        auto profiler = get_profiler();
        if(profiler)
        {
            for(const auto& itr : profiler->get_event_names())
                insert(itr);
            for(const auto& itr : profiler->get_metric_names())
                insert(itr);
        }
        return arr;
    }

    static strvec_t get_available_events(int devid)
    {
        return cupti::available_events(cupti::get_device(devid));
    }

    static strvec_t get_available_metrics(int devid)
    {
        return cupti::available_metrics(cupti::get_device(devid));
    }

    static tuple_type get_available(const tuple_type&, int);

    static void init()
    {
        auto _manager = manager::instance();
        if(!_manager || _manager->is_finalized() || _manager->is_finalizing())
            return;

        auto _init_cb = tim::get_env<bool>("TIMEMORY_CUPTI_INIT_CB", true);
        cupti::init_driver();
        if(_init_cb)
            cuda::device_sync();
        clear();

        auto& _profiler = _get_profiler();
        auto& _events   = *_get_events();
        auto& _metrics  = *_get_metrics();
        auto& _device   = *_get_device();
        auto& _labels   = *_get_labels();

        auto _init = get_initializer()();

        _device  = std::get<0>(_init);
        _events  = std::get<1>(_init);
        _metrics = std::get<2>(_init);

        using intset_t = std::set<int>;
        using strset_t = std::set<string_t>;

        intset_t _used_devs;
        strset_t _used_evts;
        strset_t _used_mets;

        auto  _dev_init = get_available(_init, _device);
        auto& _dev      = std::get<0>(_dev_init);

        // if < 0, no metrics or events available/specified
        if(_dev >= 0)
        {
            if(settings::debug())
                TIMEMORY_PRINTF(stderr,
                                "Creating CUPTI hardware profiler for device %i...\n",
                                _device);

            auto& _evt = std::get<1>(_dev_init);
            auto& _met = std::get<2>(_dev_init);

            if(!_evt.empty() || !_met.empty())
            {
                _profiler = std::make_shared<cupti::profiler>(_evt, _met, _dev, _init_cb);
                _used_devs.insert(_dev);
                for(const auto& itr : _evt)
                    _used_evts.insert(itr);
                for(const auto& itr : _met)
                    _used_mets.insert(itr);
                _labels = generate_labels();
            }
            else
            {
                static int _pass = 0;
                if(_pass++ > 0)
                    TIMEMORY_PRINTF(stderr,
                                    "[cupti_counters]> Warning! No events or metrics!\n");
            }
        }
        else
        {
            TIMEMORY_PRINTF(stderr, "[cupti_counters]> Warning! No devices available!\n");
        }

        if(!_used_devs.empty())
        {
            // if(settings::verbose() > 0 || settings::debug())
            {
                std::cout << "Devices : " << writer<intset_t>(_used_devs) << std::endl;
                std::cout << "Event   : " << writer<strset_t>(_used_evts) << std::endl;
                std::cout << "Metrics : " << writer<strset_t>(_used_mets) << std::endl;
                std::cout << "Labels  : " << writer<strvec_t>(_labels) << std::endl;
            }
        }
    }

    static void clear()
    {
        if(_get_metrics())
            _get_metrics()->clear();
        if(_get_events())
            _get_events()->clear();
        _get_profiler().reset();
    }

public:
    static void cleanup()
    {
        clear();
        delete _get_device();
        delete _get_events();
        delete _get_labels();
        delete _get_metrics();
        _get_device()  = nullptr;
        _get_events()  = nullptr;
        _get_labels()  = nullptr;
        _get_metrics() = nullptr;
    }

private:
    kernel_results_t m_kernel_value;
    kernel_results_t m_kernel_accum;
};

//--------------------------------------------------------------------------------------//

inline cupti_counters::tuple_type
cupti_counters::get_available(const tuple_type& _init, int devid)
{
    if(devid < 0 || devid >= cuda::device_count())
    {
        int ndev = cuda::device_count();
        TIMEMORY_PRINTF(stderr,
                        "[cupti_counters]> Invalid device id: %i. # devices: %i...\n",
                        devid, ndev);
        return tuple_type(-1, strvec_t(), strvec_t());
    }

    strvec_t _events  = std::get<1>(_init);
    strvec_t _metrics = std::get<2>(_init);

    auto _tmp_init = get_initializer()();

    if(_events.empty())
        _events = std::get<1>(_tmp_init);

    // provide defaults events
    if(_events.empty())
    {
        // _events = { "active_warps", "active_cycles", "global_load", "global_store" };
    }

    if(_metrics.empty())
        _metrics = std::get<2>(_tmp_init);

    // provide default metrics
    if(_metrics.empty())
    {
        //_metrics = { "inst_per_warp", "branch_efficiency", "gld_efficiency",
        //             "gst_efficiency", "warp_execution_efficiency" };
    }

    const auto& _avail_events = get_available_events(devid);
    const auto& _avail_metric = get_available_metrics(devid);

    std::set<std::string> _discarded_events{};
    std::set<std::string> _discarded_metrics{};

    bool _discard = true;

    // handle events
    auto _not_event = [&_avail_events, &_discarded_events,
                       &_discard](const string_t& evt) {
        bool nf = (std::find(std::begin(_avail_events), std::end(_avail_events), evt) ==
                   std::end(_avail_events));
        if(nf && _discard)
            _discarded_events.insert(evt);
        return nf;
    };

    // handle metrics
    auto _not_metric = [&_avail_metric, &_discarded_metrics,
                        &_discard](const string_t& met) {
        bool nf = (std::find(std::begin(_avail_metric), std::end(_avail_metric), met) ==
                   std::end(_avail_metric));
        if(nf && _discard)
            _discarded_metrics.insert(met);
        return nf;
    };

    // do the removals
    _events.erase(std::remove_if(std::begin(_events), std::end(_events), _not_event),
                  std::end(_events));

    _metrics.erase(std::remove_if(std::begin(_metrics), std::end(_metrics), _not_metric),
                   std::end(_metrics));

    // turn off discarding
    _discard = false;

    // check to see if any requested events are actually metrics
    for(const auto& itr : _discarded_events)
    {
        bool is_metric = !(_not_metric(itr));
        if(is_metric)
        {
            _metrics.push_back(itr);
        }
        else
        {
            TIMEMORY_PRINTF(
                stderr,
                "[cupti_counters]> Removing unavailable event '%s' on device %i...\n",
                itr.c_str(), devid);
        }
    }

    // check to see if any requested metrics are actually events
    for(const auto& itr : _discarded_metrics)
    {
        bool is_event = !(_not_event(itr));
        if(is_event)
        {
            _events.push_back(itr);
        }
        else
        {
            TIMEMORY_PRINTF(
                stderr,
                "[cupti_counters]> Removing unavailable metric '%s' on device "
                "%i...\n",
                itr.c_str(), devid);
        }
    }

    // determine total
    return tuple_type(devid, _events, _metrics);
}

//--------------------------------------------------------------------------------------//

using cupti_event = cupti_counters;

//--------------------------------------------------------------------------------------//

}  // namespace component

}  // namespace tim
