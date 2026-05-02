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

#include "timemory/components/perfetto/policy.hpp"
#include "timemory/components/perfetto/types.hpp"

#include <type_traits>

// perfetto forward decl types
namespace perfetto
{
struct TracingInitArgs;
struct TracingSession;
struct CounterTrack;
}  // namespace perfetto

// perfetto header
#if defined(TIMEMORY_USE_PERFETTO)
#    include <perfetto.h>
#endif

// provides a constexpr "category"
#if !defined(TIMEMORY_PERFETTO_API)
#    define TIMEMORY_PERFETTO_API ::tim::project::timemory
#endif

// allow including file to override by defining before inclusion
#if !defined(TIMEMORY_PERFETTO_CATEGORIES)
#    define TIMEMORY_PERFETTO_CATEGORIES                                                 \
        perfetto::Category("timemory").SetDescription("Events from the timemory API")
#endif

#if defined(TIMEMORY_USE_PERFETTO) && !defined(TIMEMORY_PERFETTO_CATEGORIES_DEFINED)
PERFETTO_DEFINE_CATEGORIES(TIMEMORY_PERFETTO_CATEGORIES);
#endif

namespace tim
{
namespace backend
{
namespace perfetto
{
template <typename... Args>
void
unused_args(Args&&...)
{}

#if defined(TIMEMORY_USE_PERFETTO)

using namespace ::perfetto;
using tracing_init_args = ::perfetto::TracingInitArgs;
using tracing_session   = ::perfetto::TracingSession;

#else

struct tracing_init_args
{};
struct tracing_session
{};

#endif

template <typename ApiT, typename Tp, typename... Args>
void
trace_counter(policy::perfetto_category<ApiT>, const char*, Tp, Args&&...,
              std::enable_if_t<std::is_integral<Tp>::value, int> = 0);

template <typename ApiT, typename... Args>
void
trace_event_start(policy::perfetto_category<ApiT>, const char*, Args&&...);

template <typename ApiT, typename... Args>
void
trace_event_stop(policy::perfetto_category<ApiT>, Args&&...);

}  // namespace perfetto
}  // namespace backend
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                      perfetto backend functions (call macros)
//
//--------------------------------------------------------------------------------------//

template <typename ApiT, typename Tp, typename... Args>
void
tim::backend::perfetto::trace_counter(policy::perfetto_category<ApiT> _category,
                                      const char* _label, Tp _val, Args&&... _args,
                                      std::enable_if_t<std::is_integral<Tp>::value, int>)
{
#if defined(TIMEMORY_USE_PERFETTO)
    TRACE_COUNTER(_category(), _label, _val, std::forward<Args>(_args)...);
#else
    unused_args(_category, _label, _val, _args...);
#endif
}

template <typename ApiT, typename... Args>
void
tim::backend::perfetto::trace_event_start(policy::perfetto_category<ApiT> _category,
                                          const char* _label, Args&&... _args)
{
#if defined(TIMEMORY_USE_PERFETTO)
    TRACE_EVENT_BEGIN(_category(), perfetto::StaticString(_label),
                      std::forward<Args>(_args)...);
#else
    unused_args(_category, _label, _args...);
#endif
}

template <typename ApiT, typename... Args>
void
tim::backend::perfetto::trace_event_stop(policy::perfetto_category<ApiT> _category,
                                         Args&&... _args)
{
#if defined(TIMEMORY_USE_PERFETTO)
    TRACE_EVENT_END(_category(), std::forward<Args>(_args)...);
#else
    unused_args(_category, _args...);
#endif
}
