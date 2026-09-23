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

#ifndef TIMEMORY_COMPONENTS_PAPI_PAPI_CONFIG_HPP_
#    define TIMEMORY_COMPONENTS_PAPI_PAPI_CONFIG_HPP_
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
struct papi_config
{
public:
    using size_type     = size_t;
    using event_list    = papi_event_vector;
    using value_type    = std::vector<long long>;
    using initializer_t = std::function<event_list()>;
    using constructor_t = std::function<void(papi_config&)>;

    static int        default_verbosity();
    static bool       add_global_event(int evt);
    static bool       remove_global_event(int evt);
    static event_list default_initializer(const papi_config*);
    static void overflow_handler(int evt_set, void* address, long long overflow_vector,
                                 void* context);

    papi_config() = default;
    explicit papi_config(std::string _v);
    explicit papi_config(std::set<int> _v);
    explicit papi_config(constructor_t _ctor);
    papi_config(std::string _string, std::set<int> _codes, constructor_t _ctor);

    ~papi_config()                      = default;
    papi_config(const papi_config&)     = delete;
    papi_config(papi_config&&) noexcept = delete;
    papi_config& operator=(const papi_config&) = delete;
    papi_config& operator=(papi_config&&) noexcept = delete;

    event_list initialize() const;
    void       finalize() const;
    void       start();
    void       stop();
    int64_t    get_inflight() const;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int);

    // these fields are meant to be modified internally
    const bool is_initialized = false;
    const bool is_finalized   = false;
    const bool is_running     = false;

    // these fields can/should be modified by the creator of the instance
    // before initialize() is called
    bool          fixed           = false;
    bool          multiplexing    = settings::papi_multiplexing();
    bool          fail_on_error   = settings::papi_fail_on_error();
    int           verbosity       = default_verbosity();
    initializer_t initializer     = {};
    std::string   config_string   = {};  //
    std::set<int> config_codes    = {};
    constructor_t config_callback = {};

    // these fields are set during initialization
    const int                      event_set     = PAPI_NULL;
    const size_t                   size          = 0;
    const std::vector<int>         event_codes   = {};
    const std::vector<std::string> event_names   = {};
    const std::vector<int64_t>     units         = {};
    const std::vector<std::string> display_units = {};
    const std::vector<std::string> labels        = {};
    const std::vector<std::string> descriptions  = {};

private:
    template <typename Tp>
    static inline Tp& make_mutable(const Tp&);
    template <typename Tp>
    static inline Tp*     make_mutable(const Tp*);
    static std::set<int>& global_events();

    std::atomic<int64_t> m_inflight{ 0 };
    mutable std::mutex   m_mutex{};
};

template <typename Archive>
inline void
papi_config::serialize(Archive& ar, const unsigned int)
{
    ar(cereal::make_nvp("multiplexing", multiplexing),
       cereal::make_nvp("config_string", config_string),
       cereal::make_nvp("config_codes", config_codes),
       cereal::make_nvp("event_codes", event_codes),
       cereal::make_nvp("event_names", event_names), cereal::make_nvp("labels", labels),
       cereal::make_nvp("descriptions", descriptions), cereal::make_nvp("units", units),
       cereal::make_nvp("display_units", display_units));
}

template <typename Tp>
inline Tp&
papi_config::make_mutable(const Tp& _v)
{
    return const_cast<Tp&>(_v);
}

template <typename Tp>
inline Tp*
papi_config::make_mutable(const Tp* _v)
{
    return const_cast<Tp*>(_v);
}
}  // namespace component
}  // namespace tim

#if !defined(TIMEMORY_COMPONENTS_PAPI_PAPI_CONFIG_CPP_)
#    if defined(TIMEMORY_PAPI_COMPONENT_HEADER_MODE) &&                                  \
        TIMEMORY_PAPI_COMPONENT_HEADER_MODE > 0
#        include "timemory/components/papi/papi_config.cpp"
#    endif
#endif
