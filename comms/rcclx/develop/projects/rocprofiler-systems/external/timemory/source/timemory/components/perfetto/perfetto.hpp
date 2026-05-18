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

#include "timemory/components/base.hpp"  // for base
#include "timemory/components/perfetto/backends.hpp"
#include "timemory/components/perfetto/policy.hpp"
#include "timemory/components/perfetto/types.hpp"
#include "timemory/mpl/concepts.hpp"  // for concepts::is_component
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"  // for derivation_types

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//                      Component declaration
//
//--------------------------------------------------------------------------------------//

namespace component
{
/// \struct tim::component::perfetto_trace
/// \brief Component providing perfetto implementation
//
struct perfetto_trace : base<perfetto_trace, void>
{
    using TracingInitArgs = backend::perfetto::tracing_init_args;
    using TracingSession  = backend::perfetto::tracing_session;

    using value_type      = void;
    using base_type       = base<perfetto_trace, value_type>;
    using strset_t        = std::unordered_set<std::string>;
    using strset_iterator = typename strset_t::iterator;
    using initializer_t   = std::function<void()>;
    using finalizer_t     = std::function<void()>;

    constexpr perfetto_trace() = default;

    struct config;

    static std::string label();
    static std::string description();
    static void        global_init();
    static void        global_finalize();

    static config&        get_config();
    static initializer_t& get_initializer();
    static finalizer_t&   get_finalizer();

    template <typename Tp, typename... Args>
    void store(const char*, Tp, Args&&...,
               enable_if_t<std::is_integral<Tp>::value, int> = 0);

    template <typename Tp, typename... Args>
    void store(Tp, Args&&..., enable_if_t<std::is_integral<Tp>::value, int> = 0);

    void        set_prefix(const char*);
    void        start();
    static void start(const char*);
    static void stop();

    template <typename ApiT, typename... Args>
    static void start(type_list<ApiT>, const char*, Args&&...);

    template <typename ApiT, typename... Args>
    static void stop(type_list<ApiT>, Args&&...);

    struct config
    {
        friend struct perfetto_trace;

        using session_t                = std::unique_ptr<TracingSession>;
        bool            in_process     = true;
        bool            system_backend = false;
        TracingInitArgs init_args      = {};

    private:
        session_t session{ nullptr };
    };

private:
    static TracingInitArgs& get_tracing_init_args();
    const char*             m_prefix = nullptr;
};

}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                      perfetto_trace component function defs
//
//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Args>
void
tim::component::perfetto_trace::store(Tp _val, Args&&... _args,
                                      enable_if_t<std::is_integral<Tp>::value, int>)
{
    if(m_prefix)
        backend::perfetto::trace_counter(
            policy::perfetto_category<TIMEMORY_PERFETTO_API>{}, m_prefix, _val,
            std::forward<Args>(_args)...);
}

template <typename Tp, typename... Args>
void
tim::component::perfetto_trace::store(const char* _label, Tp _val, Args&&... _args,
                                      enable_if_t<std::is_integral<Tp>::value, int>)
{
    backend::perfetto::trace_counter(policy::perfetto_category<TIMEMORY_PERFETTO_API>{},
                                     _label, _val, std::forward<Args>(_args)...);
}

template <typename ApiT, typename... Args>
void
tim::component::perfetto_trace::start(tim::type_list<ApiT>, const char* _label,
                                      Args&&... _args)
{
    backend::perfetto::trace_event_start(policy::perfetto_category<ApiT>{}, _label,
                                         std::forward<Args>(_args)...);
}

template <typename ApiT, typename... Args>
void
tim::component::perfetto_trace::stop(tim::type_list<ApiT>, Args&&... _args)
{
    backend::perfetto::trace_event_stop(policy::perfetto_category<ApiT>{},
                                        std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_PERFETTO_COMPONENT_HEADER_MODE) &&                                  \
    TIMEMORY_PERFETTO_COMPONENT_HEADER_MODE > 0
#    include "timemory/components/perfetto/perfetto.cpp"
#endif
