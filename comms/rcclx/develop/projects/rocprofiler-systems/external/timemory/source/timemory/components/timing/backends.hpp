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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "timemory/macros/attributes.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <ratio>

#if defined(TIMEMORY_UNIX)

#    include <pthread.h>
#    include <sys/times.h>
#    include <unistd.h>

#elif defined(TIMEMORY_WINDOWS)
//
//  Windows does not have tms definition
//

// without this, windows will define macros for min and max
#    include <sys/timeb.h>
#    include <sys/types.h>

#    if defined(TIMEMORY_USE_WINSOCK)
#        include <winsock.h>

EXTERN_C inline int
gettimeofday(struct timeval* t, void* timezone) noexcept
{
    struct _timeb timebuffer;
#        if defined(_WIN64)
    _ftime64(&timebuffer);
#        elif defined(_WIN32)
    _ftime(&timebuffer);
#        endif
    t->tv_sec  = timebuffer.time;
    t->tv_usec = 1000 * timebuffer.millitm;
    return 0;
}
#    endif

#    define __need_clock_t

#    include <processthreadsapi.h>
#    include <time.h>

// Structure describing CPU time used by a process and its children.
struct tms
{
    clock_t tms_utime;   // User CPU time
    clock_t tms_stime;   // System CPU time
    clock_t tms_cutime;  // User CPU time of dead children
    clock_t tms_cstime;  // System CPU time of dead children
};

// Store the CPU time used by this process and all its
// dead children (and their dead children) in BUFFER.
// Return the elapsed real time, or (clock_t) -1 for errors.
// All times are in CLK_TCKths of a second.

EXTERN_C inline clock_t
times(struct tms* __buffer) noexcept
{
    __buffer->tms_utime  = clock();
    __buffer->tms_stime  = 0;
    __buffer->tms_cstime = 0;
    __buffer->tms_cutime = 0;
    return __buffer->tms_utime;
}

using suseconds_t = long long;

enum clockid_t
{
    CLOCK_REALTIME,
    CLOCK_MONOTONIC,
    CLOCK_MONOTONIC_RAW,
    CLOCK_THREAD_CPUTIME_ID,
    CLOCK_PROCESS_CPUTIME_ID
};

EXTERN_C inline LARGE_INTEGER
get_filetime_offset() noexcept
{
    SYSTEMTIME    s;
    FILETIME      f;
    LARGE_INTEGER t;

    s.wYear         = 1970;
    s.wMonth        = 1;
    s.wDay          = 1;
    s.wHour         = 0;
    s.wMinute       = 0;
    s.wSecond       = 0;
    s.wMilliseconds = 0;
    SystemTimeToFileTime(&s, &f);
    t.QuadPart = f.dwHighDateTime;
    t.QuadPart <<= 32;
    t.QuadPart |= f.dwLowDateTime;
    return (t);
}

EXTERN_C inline int
clock_gettime(clockid_t, struct timespec* tv) noexcept
{
    LARGE_INTEGER        t;
    FILETIME             f;
    double               microseconds;
    static LARGE_INTEGER offset;
    static double        frequencyToMicroseconds;
    static int           initialized           = 0;
    static BOOL          usePerformanceCounter = 0;

    if(!initialized)
    {
        LARGE_INTEGER performanceFrequency;
        initialized           = 1;
        usePerformanceCounter = QueryPerformanceFrequency(&performanceFrequency);
        if(usePerformanceCounter)
        {
            QueryPerformanceCounter(&offset);
            frequencyToMicroseconds = (double) performanceFrequency.QuadPart / 1000000.;
        }
        else
        {
            offset                  = get_filetime_offset();
            frequencyToMicroseconds = 10.;
        }
    }
    if(usePerformanceCounter)
        QueryPerformanceCounter(&t);
    else
    {
        GetSystemTimeAsFileTime(&f);
        t.QuadPart = f.dwHighDateTime;
        t.QuadPart <<= 32;
        t.QuadPart |= f.dwLowDateTime;
    }

    t.QuadPart -= offset.QuadPart;
    microseconds = (double) t.QuadPart / frequencyToMicroseconds;
    t.QuadPart   = microseconds;
    tv->tv_sec   = t.QuadPart / 1000000;
    tv->tv_nsec  = t.QuadPart % 1000000;
    return (0);
}

#endif

namespace tim
{
//--------------------------------------------------------------------------------------//

TIMEMORY_INLINE int64_t
clock_tick() noexcept
{
#if defined(TIMEMORY_WINDOWS)
    return CLOCKS_PER_SEC;
#else
    static int64_t _val = ::sysconf(_SC_CLK_TCK);
    return _val;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Precision, typename Ret = int64_t>
TIMEMORY_INLINE Ret
clock_tick() noexcept
{
    return static_cast<Ret>(Precision::den) / static_cast<Ret>(clock_tick());
}

//--------------------------------------------------------------------------------------//
// general struct for the differnt clock_gettime functions
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_now(clockid_t clock_id) noexcept
{
    constexpr Tp factor = Precision::den / static_cast<Tp>(std::nano::den);
#if defined(TIMEMORY_MACOS)
    return clock_gettime_nsec_np(clock_id) * factor;
#else
    struct timespec ts;
    clock_gettime(clock_id, &ts);
    return (ts.tv_sec * std::nano::den + ts.tv_nsec) * factor;
#endif
}

//--------------------------------------------------------------------------------------//
// the system's real time (i.e. wall time) clock, expressed as the amount of time since
// the epoch.
template <typename Tp = double, typename Precision = std::ratio<1>,
          typename ClockT = std::chrono::steady_clock>
TIMEMORY_INLINE Tp
get_clock_real_now() noexcept
{
    using clock_type    = ClockT;
    using duration_type = std::chrono::duration<typename clock_type::rep, Precision>;

    // return get_clock_now<Tp, Precision>(CLOCK_REALTIME);
    return std::chrono::duration_cast<duration_type>(clock_type::now().time_since_epoch())
        .count();
}

//--------------------------------------------------------------------------------------//
// clock that increments monotonically, tracking the time since an arbitrary point,
// and will continue to increment while the system is asleep.
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_monotonic_now() noexcept
{
    return get_clock_now<Tp, Precision>(CLOCK_MONOTONIC);
}

//--------------------------------------------------------------------------------------//
// clock that increments monotonically, tracking the time since an arbitrary point like
// CLOCK_MONOTONIC.  However, this clock is unaffected by frequency or time adjustments.
// It should not be compared to other system time sources.
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_monotonic_raw_now() noexcept
{
    return get_clock_now<Tp, Precision>(CLOCK_MONOTONIC_RAW);
}

//--------------------------------------------------------------------------------------//
// this clock measures the CPU time within the current thread (excludes sibling/child
// threads)
// clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling
// thread.
#if defined(TIMEMORY_WINDOWS)
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_thread_now() noexcept
{
    auto _get_thr_time = [](const FILETIME& kernel_time,
                            const FILETIME& user_time) -> Tp {
        // values are provided in 100 ns units
        constexpr Tp   factor = Precision::den / static_cast<Tp>(std::nano::den / 100);
        ULARGE_INTEGER kernel;
        ULARGE_INTEGER user;
        kernel.HighPart = kernel_time.dwHighDateTime;
        kernel.LowPart  = kernel_time.dwLowDateTime;
        user.HighPart   = user_time.dwHighDateTime;
        user.LowPart    = user_time.dwLowDateTime;
        return (kernel.QuadPart + user.QuadPart) * factor;
    };
    HANDLE   this_thread = GetCurrentThread();
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    GetThreadTimes(this_thread, &creation_time, &exit_time, &kernel_time, &user_time);
    return _get_thr_time(kernel_time, user_time);
}
#else
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_thread_now() noexcept
{
    return get_clock_now<Tp, Precision>(CLOCK_THREAD_CPUTIME_ID);
}
#endif

//--------------------------------------------------------------------------------------//
// this clock measures the CPU time within the current process (excludes child processes)
// clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling
// process.
#if defined(TIMEMORY_WINDOWS)
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_process_now() noexcept
{
    auto _get_proc_time = [](const FILETIME& kernel_time,
                             const FILETIME& user_time) -> Tp {
        // values are provided in 100 ns units
        constexpr Tp   factor = Precision::den / static_cast<Tp>(std::nano::den / 100);
        ULARGE_INTEGER kernel;
        ULARGE_INTEGER user;
        kernel.HighPart = kernel_time.dwHighDateTime;
        kernel.LowPart  = kernel_time.dwLowDateTime;
        user.HighPart   = user_time.dwHighDateTime;
        user.LowPart    = user_time.dwLowDateTime;
        return (kernel.QuadPart + user.QuadPart) * factor;
    };
    HANDLE   this_proc = GetCurrentProcess();
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    GetProcessTimes(this_proc, &creation_time, &exit_time, &kernel_time, &user_time);
    return _get_proc_time(kernel_time, user_time);
}
#else
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_process_now() noexcept
{
    return get_clock_now<Tp, Precision>(CLOCK_PROCESS_CPUTIME_ID);
}
#endif

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in user-mode
// for this process and child processes
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_user_now() noexcept
{
#if defined(TIMEMORY_WINDOWS)
    auto _get_user_time = [](const FILETIME& user_time) -> Tp {
        // values are provided in 100 ns units
        constexpr Tp   factor = Precision::den / static_cast<Tp>(std::nano::den / 100);
        ULARGE_INTEGER user;
        user.HighPart = user_time.dwHighDateTime;
        user.LowPart  = user_time.dwLowDateTime;
        return (user.QuadPart * factor);
    };
    HANDLE   this_proc = GetCurrentProcess();
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    GetProcessTimes(this_proc, &creation_time, &exit_time, &kernel_time, &user_time);
    return _get_user_time(user_time);
#else
    tms _tms;
    ::times(&_tms);
    return (_tms.tms_utime + _tms.tms_cutime) * static_cast<Tp>(clock_tick<Precision>());
#endif
}

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in kernel-mode
// for this process and child processes
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_system_now() noexcept
{
#if defined(TIMEMORY_WINDOWS)
    auto _get_kern_time = [](const FILETIME& kernel_time) -> Tp {
        // values are provided in 100 ns units
        constexpr Tp   factor = Precision::den / static_cast<Tp>(std::nano::den / 100);
        ULARGE_INTEGER kernel;
        kernel.HighPart = kernel_time.dwHighDateTime;
        kernel.LowPart  = kernel_time.dwLowDateTime;
        return (kernel.QuadPart * factor);
    };
    HANDLE   this_proc = GetCurrentProcess();
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    GetProcessTimes(this_proc, &creation_time, &exit_time, &kernel_time, &user_time);
    return _get_kern_time(kernel_time);
#else
    tms _tms;
    ::times(&_tms);
    return (_tms.tms_stime + _tms.tms_cstime) * static_cast<Tp>(clock_tick<Precision>());
#endif
}

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in both user- and kernel- mode
// for this process and child processes
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_clock_cpu_now() noexcept
{
#if defined(TIMEMORY_WINDOWS)
    return get_clock_process_now<Tp, Precision>();
#else
    tms _tms;
    ::times(&_tms);
    return (_tms.tms_utime + _tms.tms_cutime + _tms.tms_stime + _tms.tms_cstime) *
           static_cast<Tp>(clock_tick<Precision>());
#endif
}

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in user-mode
// for this process only
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_self_clock_user_now() noexcept
{
    tms _tms;
    ::times(&_tms);
    return _tms.tms_utime * static_cast<Tp>(clock_tick<Precision>());
}

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in kernel-mode
// for this process only
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_self_clock_system_now() noexcept
{
    tms _tms;
    ::times(&_tms);
    return _tms.tms_stime * static_cast<Tp>(clock_tick<Precision>());
}

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in both user- and kernel- mode
// for this process only
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_self_clock_cpu_now() noexcept
{
    return (clock() * static_cast<Tp>(Precision::den)) / static_cast<Tp>(CLOCKS_PER_SEC);
}

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in user-mode
// for this process only
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_child_clock_user_now() noexcept
{
    tms _tms;
    ::times(&_tms);
    return _tms.tms_cutime * static_cast<Tp>(clock_tick<Precision>());
}

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in kernel-mode
// for this process only
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_child_clock_system_now() noexcept
{
    tms _tms;
    ::times(&_tms);
    return _tms.tms_cstime * static_cast<Tp>(clock_tick<Precision>());
}

//--------------------------------------------------------------------------------------//
// this function extracts only the CPU time spent in both user- and kernel- mode
// for this process only
template <typename Tp = double, typename Precision = std::ratio<1>>
TIMEMORY_INLINE Tp
get_child_clock_cpu_now() noexcept
{
    tms _tms;
    ::times(&_tms);
    return (_tms.tms_cutime + _tms.tms_cstime) * static_cast<Tp>(clock_tick<Precision>());
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
