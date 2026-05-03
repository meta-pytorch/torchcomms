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

#ifndef TIMEMORY_COMPONENTS_PAPI_PAPI_CONFIG_CPP_
#    define TIMEMORY_COMPONENTS_PAPI_PAPI_CONFIG_CPP_
#endif

#ifndef TIMEMORY_COMPONENTS_PAPI_PAPI_CONFIG_HPP_
#    include "timemory/components/papi/papi_config.hpp"
#    define TIMEMORY_PAPI_CONFIG_INLINE
#elif !defined(TIMEMORY_PAPI_SOURCE)
#    define TIMEMORY_PAPI_CONFIG_INLINE inline
#else
#    define TIMEMORY_PAPI_CONFIG_INLINE
#endif

#include "timemory/backends/papi.hpp"
#include "timemory/components/papi/backends.hpp"
#include "timemory/components/papi/macros.hpp"
#include "timemory/components/papi/papi_event_vector.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/defines.h"
#include "timemory/macros/attributes.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/macros.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
namespace component
{
TIMEMORY_PAPI_CONFIG_INLINE std::set<int>&
                            papi_config::global_events()
{
    static auto _v = std::set<int>{};
    return _v;
}

TIMEMORY_PAPI_CONFIG_INLINE void
papi_config::overflow_handler(int evt_set, void* address, long long overflow_vector,
                              void* context)
{
    TIMEMORY_PRINTF(stderr, "[papi_config]> Overflow at %p! bit=0x%llx \n", evt_set,
                    address, overflow_vector);
    consume_parameters(context);
}

TIMEMORY_PAPI_CONFIG_INLINE bool
papi_config::add_global_event(int evt)
{
    return global_events().emplace(evt).second;
}

TIMEMORY_PAPI_CONFIG_INLINE bool
papi_config::remove_global_event(int evt)
{
    return (global_events().erase(evt) > 0);
}

TIMEMORY_PAPI_CONFIG_INLINE
papi_config::papi_config(std::string _v)
: config_string{ std::move(_v) }
{}

TIMEMORY_PAPI_CONFIG_INLINE
papi_config::papi_config(std::set<int> _v)
: config_codes{ std::move(_v) }
{}

TIMEMORY_PAPI_CONFIG_INLINE
papi_config::papi_config(constructor_t _v)
: config_callback{ std::move(_v) }
{}

TIMEMORY_PAPI_CONFIG_INLINE
papi_config::papi_config(std::string _string, std::set<int> _codes, constructor_t _cb)
: config_string{ std::move(_string) }
, config_codes{ std::move(_codes) }
, config_callback{ std::move(_cb) }
{}

TIMEMORY_PAPI_CONFIG_INLINE
int
papi_config::default_verbosity()
{
    return (settings::papi_quiet()) ? -1
                                    : ((settings::debug()) ? 16 : settings::verbose());
}

// clang-format off
TIMEMORY_PAPI_CONFIG_INLINE
papi_event_vector  // clang-format on
papi_config::default_initializer(const papi_config* _cfg)
{
    if(!trait::runtime_enabled<papi_config>::get())
        return papi_event_vector{};

    if(!papi::working())
        return papi_event_vector{};

    const auto& events_str = _cfg->config_string;

    if(events_str.empty() && _cfg->config_codes.empty())
        return papi_event_vector{};

    // don't delimit colons!
    papi_event_vector events_str_list = delimit(events_str, "\"',; ");
    papi_event_vector events_list     = {};

    auto _query_event = [](const std::string& _evt) {
        auto _v = papi::query_event(_evt);
        (void) papi::get_event_info(_evt);
        return _v;
    };

    auto _codes = _cfg->config_codes;
    // if the configuration is not fixed (i.e. static), add in the global events
    if(!_cfg->fixed)
    {
        for(auto itr : global_events())
            _codes.emplace(itr);
    }

    for(const auto& itr : _codes)
    {
        std::string _itr_str = papi::get_event_info(itr).symbol;
        if(!_query_event(_itr_str))
            continue;
        auto fitr = std::find(events_list.begin(), events_list.end(), _itr_str);
        if(fitr == events_list.end())
            events_list.push_back(_itr_str);
    }

    for(const auto& itr : events_str_list)
    {
        if(itr.length() == 0)
            continue;

        if(_cfg->verbosity >= 8)
        {
            TIMEMORY_PRINTF(stderr, "[papi_config] Querying event '%s'...\n",
                            itr.c_str());
        }

        if(!_query_event(itr))
        {
            std::stringstream _ss{};
            log::stream(_ss, log::color::warning())
                << "[papi_config] Event '" << itr << "' not valid";
            if(_cfg->fail_on_error)
            {
                TIMEMORY_EXCEPTION(_ss.str());
            }
            else if(_cfg->verbosity >= 0)
            {
                TIMEMORY_PRINTF(stderr, "%s\n", _ss.str().c_str());
            }
            continue;
        }

        auto fitr = std::find(events_list.begin(), events_list.end(), itr);
        if(fitr == events_list.end())
        {
            if(_cfg->verbosity >= 3)
            {
                TIMEMORY_PRINTF(stderr,
                                "[papi_config] Successfully queried event '%s'...\n",
                                itr.c_str());
            }
            events_list.push_back(itr);
        }
        else
        {
            if(_cfg->verbosity >= 4)
            {
                TIMEMORY_PRINTF(stderr, "[papi_config] Event '%s' already exists...\n",
                                itr.c_str());
            }
        }
    }

    return events_list;
}

// clang-format off
TIMEMORY_PAPI_CONFIG_INLINE
papi_config::event_list  // clang-format on
papi_config::initialize() const
{
    if(!trait::runtime_enabled<papi_config>::get() || is_finalized)
        return event_list{};

    std::unique_lock<std::mutex> _lk{ m_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();

    if(is_initialized)
        return event_names;

    papi::init();
    papi::register_thread();

    if(config_callback)
        config_callback(make_mutable(*this));

    // get the event list
    auto _events = (initializer) ? initializer() : default_initializer(this);

    // convert papi_event_vector to real vectors of strings and ints
    make_mutable(event_names) = _events;
    make_mutable(event_codes) = _events;

    if(!event_names.empty())
    {
        if(verbosity >= 2)
            TIMEMORY_PRINT_HERE("configuring %zu papi event_names", event_names.size());
        auto& _event_set   = make_mutable(event_set);
        auto& _event_names = make_mutable(event_names);
        papi::create_event_set(&_event_set, multiplexing);
        papi::add_events(event_set, _event_names.data(), event_names.size());
        make_mutable(is_initialized) = papi::working();
        make_mutable(size)           = event_names.size();
    }

    if(is_initialized)
    {
        auto& _units         = make_mutable(units);
        auto& _labels        = make_mutable(labels);
        auto& _descriptions  = make_mutable(descriptions);
        auto& _display_units = make_mutable(display_units);

        _units.clear();
        _labels.clear();
        _descriptions.clear();
        _display_units.clear();

        for(const auto& itr : event_names)
        {
            papi::event_info_t _info = papi::get_event_info(itr);

            std::string _name = {};
            if(!_info.modified_short_descr)
                _name = _info.short_descr;
            if(_name.empty())
                _name = _info.symbol;
            if(_name.empty())
                _name = itr;

            size_t n = std::string::npos;
            while((n = _name.find("L/S")) != std::string::npos)
                _name.replace(n, 3, "Loads_Stores");
            while((n = _name.find('/')) != std::string::npos)
                _name.replace(n, 1, "_per_");
            while((n = _name.find(' ')) != std::string::npos)
                _name.replace(n, 1, "_");
            while((n = _name.find("__")) != std::string::npos)
                _name.replace(n, 2, "_");

            _units.emplace_back(1);
            _labels.emplace_back(_name);
            _descriptions.emplace_back(_info.long_descr);
            _display_units.emplace_back(_info.units);
        }
    }

    if(!event_names.empty() && !is_initialized)
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(verbosity >= 0,
                                        "Warning! Configuring %i papi event_names failed",
                                        (int) event_names.size());
    }

    return _events;
}

// clang-format off
TIMEMORY_PAPI_CONFIG_INLINE
void  // clang-format on
papi_config::finalize() const
{
    if(!is_initialized || is_finalized)
        return;

    if(event_set != PAPI_NULL && !event_names.empty())
    {
        while(m_inflight > 0)
            make_mutable(this)->stop();
        auto& _event_names = make_mutable(event_names);
        papi::remove_events(event_set, _event_names.data(), event_names.size());
        papi::destroy_event_set(event_set);
    }

    make_mutable(is_finalized) = true;
    papi::unregister_thread();
}

// clang-format off
TIMEMORY_PAPI_CONFIG_INLINE
void  // clang-format on
papi_config::start()
{
    if(is_finalized)
        return;

    auto _idx = m_inflight.fetch_add(1, std::memory_order_relaxed);
    if(_idx == 0 && !is_initialized)
        initialize();
    else if(_idx > 0)
        return;

    if(is_initialized)
    {
        make_mutable(is_running) = true;
        papi::start(event_set);
    }
}

// clang-format off
TIMEMORY_PAPI_CONFIG_INLINE
void  // clang-format on
papi_config::stop()
{
    if(is_finalized)
        return;

    auto _idx = m_inflight.fetch_add(-1, std::memory_order_relaxed);
    if(_idx > 1)
        return;

    if(is_running)
    {
        make_mutable(is_running) = false;
        value_type _data(event_names.size(), 0);
        papi::stop(event_set, _data.data());
    }
}
}  // namespace component
}  // namespace tim
