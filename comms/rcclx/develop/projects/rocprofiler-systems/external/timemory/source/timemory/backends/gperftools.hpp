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

#include <iostream>
#include <sstream>
#include <string>

#if defined(TIMEMORY_USE_GPERFTOOLS) || defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
#    include <gperftools/heap-profiler.h>
#    if !defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
#        define TIMEMORY_USE_GPERFTOOLS_TCMALLOC
#    endif
#endif

#if defined(TIMEMORY_USE_GPERFTOOLS) || defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
#    include <gperftools/profiler.h>
#    if !defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
#        define TIMEMORY_USE_GPERFTOOLS_PROFILER
#    endif
#endif

//======================================================================================//
//
//      GOOGLE PERF-TOOLS
//
//======================================================================================//

namespace tim
{
namespace gperftools
{
//--------------------------------------------------------------------------------------//

template <typename... Types>
void
consume_parameters(Types&&...)
{}

//--------------------------------------------------------------------------------------//

namespace cpu
{
//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)

struct _ProfilerState
{
    int         enabled          = 0;  /* Is profiling currently enabled? */
    uint64_t    start_time       = 0;  /* If enabled, when was profiling started? */
    std::string profile_name     = {}; /* Name of profile file being written, or '\0' */
    int         samples_gathered = 0;  /* Number of samples gathered so far (or 0) */
};

struct _ProfilerOptions
{
    /* Filter function and argument.
     *
     * If filter_in_thread is not NULL, when a profiling tick is delivered
     * the profiler will call:
     *
     *   (*filter_in_thread)(filter_in_thread_arg)
     *
     * If it returns nonzero, the sample will be included in the profile.
     * Note that filter_in_thread runs in a signal handler, so must be
     * async-signal-safe.
     *
     * A typical use would be to set up filter results for each thread
     * in the system before starting the profiler, then to make
     * filter_in_thread be a very simple function which retrieves those
     * results in an async-signal-safe way.  Retrieval could be done
     * using thread-specific data, or using a shared data structure that
     * supports async-signal-safe lookups.
     */
    int (*filter_in_thread)(void* arg);
    void* filter_in_thread_arg;
};

using state_t   = _ProfilerState;
using options_t = _ProfilerOptions;

#else

using state_t   = ProfilerState;
using options_t = ProfilerOptions;

#endif

//--------------------------------------------------------------------------------------//

inline int
profiler_start(const std::string& name, options_t* options = nullptr)
{
#if defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
    return ProfilerStartWithOptions(name.c_str(), options);
#else
    consume_parameters(name, options);
    return 0;
#endif
}

//--------------------------------------------------------------------------------------//

inline void
profiler_stop()
{
#if defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
    ProfilerStop();
#endif
}

//--------------------------------------------------------------------------------------//

inline void
profiler_flush()
{
#if defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
    ProfilerFlush();
#endif
}

//--------------------------------------------------------------------------------------//

inline void
register_thread()
{
#if defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
    ProfilerRegisterThread();
#endif
}

//--------------------------------------------------------------------------------------//

inline state_t
get_state()
{
    state_t _state;
#if defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
    ProfilerGetCurrentState(&_state);
#endif
    return _state;
}

//--------------------------------------------------------------------------------------//

inline bool
is_running()
{
    return (get_state().enabled != 0) ? true : false;
}

//--------------------------------------------------------------------------------------//

}  // namespace cpu

//======================================================================================//

namespace heap
{
//--------------------------------------------------------------------------------------//

inline int
profiler_start(const std::string& name)
{
#if defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
    HeapProfilerStart(name.c_str());
#else
    consume_parameters(name);
#endif
    return 0;
}

//--------------------------------------------------------------------------------------//

inline void
profiler_stop()
{
#if defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
    HeapProfilerStop();
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
is_running()
{
#if defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
    return (IsHeapProfilerRunning() == 0) ? false : true;
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

inline void
profiler_flush(const std::string& reason)
{
#if defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
    HeapProfilerDump(reason.c_str());
#else
    consume_parameters(reason);
#endif
}

//--------------------------------------------------------------------------------------//

inline std::string
get_profile()
{
#if defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
    char*             prof = GetHeapProfile();
    std::stringstream ss;
    ss << prof;
    free(prof);
    return ss.str();
#else
    return "";
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace heap

//======================================================================================//

inline void
profiler_stop()
{
    cpu::profiler_stop();
    heap::profiler_stop();
}

//--------------------------------------------------------------------------------------//
}  // namespace gperftools

}  // namespace tim
