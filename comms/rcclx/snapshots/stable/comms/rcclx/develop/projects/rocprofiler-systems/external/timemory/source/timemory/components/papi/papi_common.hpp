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
#include "timemory/components/papi/backends.hpp"
#include "timemory/components/papi/macros.hpp"
#include "timemory/components/papi/papi_config.hpp"
#include "timemory/components/papi/papi_config_factory.hpp"
#include "timemory/components/papi/papi_event_vector.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/defines.h"
#include "timemory/macros/attributes.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/delimit.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                          Common PAPI configuration
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct TIMEMORY_VISIBLE papi_common : private policy::instance_tracker<papi_common<Tp>>
{
public:
    using this_type     = papi_common<Tp>;
    using size_type     = size_t;
    using event_list    = papi_event_vector;
    using factory_type  = papi_config_factory<Tp>;
    using initializer_t = decltype(std::declval<papi_config>().initializer);
    using tracker_type  = policy::instance_tracker<this_type>;

    static papi_config*             get_config();
    static initializer_t            get_initializer();
    static bool                     is_initialized();
    static bool                     is_finalized();
    static void                     initialize();
    static void                     finalize();
    static size_t                   size();
    static int                      event_set();
    static bool                     is_fixed();
    static bool                     is_running();
    static std::vector<int>         event_codes();
    static std::vector<std::string> event_names();

    TIMEMORY_DEFAULT_OBJECT(papi_common)

protected:
    using tracker_type::m_thr;
};

template <typename Tp>
papi_config*
papi_common<Tp>::get_config()
{
    static auto* _v = factory_type{}();
    return _v;
}

template <typename Tp>
typename papi_common<Tp>::initializer_t
papi_common<Tp>::get_initializer()
{
    const auto& _cfg = get_config();
    if(_cfg)
        return _cfg->initializer;
    return []() { return std::invoke_result_t<initializer_t>{}; };
}

template <typename Tp>
inline bool
papi_common<Tp>::is_initialized()
{
    return (get_config() != nullptr && get_config()->is_initialized);
}

template <typename Tp>
inline bool
papi_common<Tp>::is_finalized()
{
    return (get_config() != nullptr && get_config()->is_finalized);
}

template <typename Tp>
inline void
papi_common<Tp>::initialize()
{
    const auto& _cfg = get_config();
    if(_cfg)
        _cfg->initialize();
}

template <typename Tp>
inline void
papi_common<Tp>::finalize()
{
    const auto& _cfg = get_config();
    if(_cfg)
        _cfg->finalize();
}

template <typename Tp>
inline size_t
papi_common<Tp>::size()
{
    if(get_config())
        return get_config()->size;
    return 0;
}

template <typename Tp>
inline int
papi_common<Tp>::event_set()
{
    if(get_config())
        return get_config()->event_set;
    return PAPI_NULL;
}

template <typename Tp>
inline bool
papi_common<Tp>::is_fixed()
{
    return (get_config() != nullptr && get_config()->fixed);
}

template <typename Tp>
inline std::vector<int>
papi_common<Tp>::event_codes()
{
    const auto& _cfg = get_config();
    if(_cfg)
        return _cfg->event_codes;
    return std::vector<int>{};
}

template <typename Tp>
inline std::vector<std::string>
papi_common<Tp>::event_names()
{
    const auto& _cfg = get_config();
    if(_cfg)
        return _cfg->event_names;
    return std::vector<std::string>{};
}

template <typename Tp>
inline bool
papi_common<Tp>::is_running()
{
    return (get_config() != nullptr && get_config()->is_running);
}
}  // namespace component
}  // namespace tim
