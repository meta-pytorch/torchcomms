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

#include "timemory/backends/process.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#if defined(TIMEMORY_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(TIMEMORY_MACOS)
#        include <libproc.h>
#        include <mach/mach.h>
#        if !defined(TIMEMORY_RUSAGE_THREAD)
#            define TIMEMORY_RUSAGE_THREAD 0
#        endif
#    else
#        if defined(RUSAGE_THREAD)
#            if !defined(TIMEMORY_RUSAGE_THREAD)
#                define TIMEMORY_RUSAGE_THREAD 1
#            endif
#        else
#            if !defined(TIMEMORY_RUSAGE_THREAD)
#                define TIMEMORY_RUSAGE_THREAD 0
#            endif
#        endif
#    endif
#elif defined(TIMEMORY_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#    if !defined(WIN32_LEAN_AND_MEAN)
#        define WIN32_LEAN_AND_MEAN
#    endif
#    if !defined(TIMEMORY_RUSAGE_THREAD)
#        define TIMEMORY_RUSAGE_THREAD 0
#    endif
#    include <stdio.h>
#    include <windows.h>
//
#    include <psapi.h>
#else
#    error "Cannot define get_peak_rss() or get_page_rss() for an unknown OS."
#endif

//======================================================================================//
//
namespace tim
{
#if defined(TIMEMORY_UNIX)

#    if !defined(TIMEMORY_RUSAGE_THREAD) || TIMEMORY_RUSAGE_THREAD < 1
#        define RUSAGE_THREAD RUSAGE_SELF
#    endif

using rusage_type_t = decltype(RUSAGE_SELF);

#elif defined(TIMEMORY_WINDOWS)

enum RUSAGE_TYPE
{
    RUSAGE_SELF,
    RUSAGE_CHILDREN,
    RUSAGE_THREAD,
};

using rusage_type_t = RUSAGE_TYPE;

#endif

inline rusage_type_t&
get_rusage_type()
{
    static auto instance = RUSAGE_THREAD;
    return instance;
}

inline pid_t&
get_rusage_pid()
{
    return process::get_target_id();
}

inline void
check_rusage_call(int ret, const char* _func)
{
#if defined(DEBUG)
    if(ret != 0)
        TIMEMORY_PRINTF_WARNING(
            stderr, "Warning! rusage call in '%s' returned a non-zero error code: %i\n",
            _func, ret);
#else
    (void) ret;
    (void) _func;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
struct rusage_cache
{
    explicit rusage_cache(rusage_type_t _t = get_rusage_type()) { update(_t); }
    ~rusage_cache() = default;

    rusage_cache(const rusage_cache&) = delete;
    rusage_cache& operator=(const rusage_cache&) = delete;

    rusage_cache(rusage_cache&&) noexcept = default;
    rusage_cache& operator=(rusage_cache&&) noexcept = default;

    inline void update(rusage_type_t _t)
    {
#if defined(TIMEMORY_UNIX)
        check_rusage_call(getrusage(_t, &m_data), __FUNCTION__);
#else
        (void) _t;
#endif
    }

    inline int64_t get_peak_rss() const;
    inline int64_t get_num_io_in() const;
    inline int64_t get_num_io_out() const;
    inline int64_t get_num_minor_page_faults() const;
    inline int64_t get_num_major_page_faults() const;
    inline int64_t get_num_voluntary_context_switch() const;
    inline int64_t get_num_priority_context_switch() const;
    inline int64_t get_user_mode_time() const;
    inline int64_t get_kernel_mode_time() const;

    friend std::ostream& operator<<(std::ostream& _os, const rusage_cache& _rc)
    {
        _os << "peak_rss: " << _rc.get_peak_rss()
            << ", kernel mode time: " << _rc.get_kernel_mode_time()
            << ", user mode time: " << _rc.get_user_mode_time()
            << ", io in: " << _rc.get_num_io_in() << ", io out: " << _rc.get_num_io_out()
            << ", major page faults: " << _rc.get_num_major_page_faults()
            << ", minor page faults: " << _rc.get_num_minor_page_faults()
            << ", prio ctx switch: " << _rc.get_num_priority_context_switch()
            << ", vol ctx switch: " << _rc.get_num_voluntary_context_switch();
        return _os;
    }

#if defined(TIMEMORY_UNIX)
private:
    using rusage_t = struct rusage;

    rusage_t m_data;
#endif
};
//
//--------------------------------------------------------------------------------------//
//
int64_t get_peak_rss(rusage_type_t = get_rusage_type());
int64_t get_stack_rss(rusage_type_t = get_rusage_type());
int64_t get_data_rss(rusage_type_t = get_rusage_type());
int64_t get_num_swap(rusage_type_t = get_rusage_type());
int64_t get_num_io_in(rusage_type_t = get_rusage_type());
int64_t get_num_io_out(rusage_type_t = get_rusage_type());
int64_t get_num_minor_page_faults(rusage_type_t = get_rusage_type());
int64_t get_num_major_page_faults(rusage_type_t = get_rusage_type());
int64_t get_num_messages_sent(rusage_type_t = get_rusage_type());
int64_t get_num_messages_received(rusage_type_t = get_rusage_type());
int64_t get_num_signals(rusage_type_t = get_rusage_type());
int64_t get_num_voluntary_context_switch(rusage_type_t = get_rusage_type());
int64_t get_num_priority_context_switch(rusage_type_t = get_rusage_type());
int64_t get_user_mode_time(rusage_type_t = get_rusage_type());
int64_t get_kernel_mode_time(rusage_type_t = get_rusage_type());

int64_t
get_page_rss();
int64_t
get_virt_mem();
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_peak_rss() const
{
#if defined(TIMEMORY_WINDOWS)
    return tim::get_peak_rss();
#else
#    if defined(TIMEMORY_MACOS)
    // Darwin reports in bytes
    return static_cast<int64_t>(m_data.ru_maxrss);
#    else
    // Linux reports in kilobytes
    return static_cast<int64_t>(units::kilobyte * m_data.ru_maxrss);
#    endif
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_num_io_in() const
{
#if defined(TIMEMORY_UNIX)
    return static_cast<int64_t>(m_data.ru_inblock);
#else
    return tim::get_num_io_in();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_num_io_out() const
{
#if defined(TIMEMORY_UNIX)
    return static_cast<int64_t>(m_data.ru_oublock);
#else
    return tim::get_num_io_out();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_num_minor_page_faults() const
{
#if defined(TIMEMORY_UNIX)
    return static_cast<int64_t>(m_data.ru_minflt);
#else
    return tim::get_num_minor_page_faults();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_num_major_page_faults() const
{
#if defined(TIMEMORY_UNIX)
    return static_cast<int64_t>(m_data.ru_majflt);
#else
    return tim::get_num_major_page_faults();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_num_voluntary_context_switch() const
{
#if defined(TIMEMORY_UNIX)
    return static_cast<int64_t>(m_data.ru_nvcsw);
#else
    return tim::get_num_voluntary_context_switch();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_num_priority_context_switch() const
{
#if defined(TIMEMORY_UNIX)
    return static_cast<int64_t>(m_data.ru_nivcsw);
#else
    return tim::get_num_priority_context_switch();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_user_mode_time() const
{
#if defined(TIMEMORY_UNIX)
    constexpr int64_t MSEC = 1000000;
    return static_cast<int64_t>(m_data.ru_utime.tv_sec * MSEC + m_data.ru_utime.tv_usec);
#else
    return tim::get_user_mode_time();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
rusage_cache::get_kernel_mode_time() const
{
#if defined(TIMEMORY_UNIX)
    constexpr int64_t MSEC = 1000000;
    return static_cast<int64_t>(m_data.ru_stime.tv_sec * MSEC + m_data.ru_stime.tv_usec);
#else
    return tim::get_kernel_mode_time();
#endif
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//======================================================================================//

// Returns the peak (maximum so far) resident set size (physical
// memory use) measured in bytes, or zero if the value cannot be
// determined on this OS.
//
inline int64_t
tim::get_peak_rss(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

// Darwin reports in bytes, Linux reports in kilobytes
#    if defined(TIMEMORY_MACOS)
    constexpr int64_t _units = 1;
#    else
    constexpr int64_t _units = units::kilobyte;
#    endif

    return static_cast<int64_t>(_units * _usage.ru_maxrss);

#elif defined(TIMEMORY_WINDOWS)
    (void) _rutype;
    DWORD                   processID = GetCurrentProcessId();
    HANDLE                  hProcess;
    PROCESS_MEMORY_COUNTERS pmc;

    hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, TRUE, processID);
    if(NULL == hProcess)
        return static_cast<int64_t>(0);

    int64_t nsize = 0;
    if(GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc)))
        nsize = static_cast<int64_t>(pmc.PeakWorkingSetSize);

    CloseHandle(hProcess);
    return nsize;
#else
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_stack_rss(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    const int64_t _units = units::kilobyte * units::clocks_per_sec;
    return static_cast<int64_t>(_units * _usage.ru_isrss);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_data_rss(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
#    if defined(TIMEMORY_MACOS)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    const int64_t _units = units::kilobyte * units::clocks_per_sec;
    return static_cast<int64_t>(_units * _usage.ru_idrss);

#    else  // Linux
    (void) _rutype;

    std::string fstatm = [&]() {
        std::stringstream fio;
        fio << "/proc/" << get_rusage_pid() << "/statm";
        return fio.str();
    }();
    int64_t       drss_size = 0;
    std::ifstream ifs;
    ifs.open(fstatm.c_str());
    if(ifs)
    {
        static int64_t dummy = 0;
        ifs >> dummy >> dummy >> dummy >> dummy >> dummy >> drss_size;
    }
    ifs.close();
    return static_cast<int64_t>(drss_size * units::get_page_size());
#    endif
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_swap(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_nswap);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_io_in(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_inblock);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_io_out(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_oublock);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_minor_page_faults(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_minflt);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_major_page_faults(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_majflt);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_messages_sent(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_msgsnd);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_messages_received(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_msgrcv);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_signals(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_nsignals);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_voluntary_context_switch(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_nvcsw);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_num_priority_context_switch(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    return static_cast<int64_t>(_usage.ru_nivcsw);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_user_mode_time(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    constexpr int64_t MSEC = 1000000;
    return static_cast<int64_t>(_usage.ru_utime.tv_sec * MSEC + _usage.ru_utime.tv_usec);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_kernel_mode_time(rusage_type_t _rutype)
{
#if defined(TIMEMORY_UNIX)
    struct rusage _usage;
    check_rusage_call(getrusage(_rutype, &_usage), __FUNCTION__);

    constexpr int64_t MSEC = 1000000;
    return static_cast<int64_t>(_usage.ru_stime.tv_sec * MSEC + _usage.ru_stime.tv_usec);
#else
    (void) _rutype;
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//
// Returns the current resident set size (physical memory use) measured
// in bytes, or zero if the value cannot be determined on this OS.
//
inline int64_t
tim::get_page_rss()
{
#if defined(TIMEMORY_UNIX)
#    if defined(TIMEMORY_MACOS)
    // OSX
    // kern_return_t kret;
    // task_t task;
    // kret = task_for_pid(mach_task_self(), get_rusage_pid(), &task);

    struct mach_task_basic_info info;
    mach_msg_type_number_t      infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if(task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info,
                 &infoCount) != KERN_SUCCESS)
    {
        TIMEMORY_PRINTF_WARNING(stderr,
                                "Warning! %s@'%s':%i :: task_info(...) != KERN_SUCCESS\n",
                                __FUNCTION__, __FILE__, __LINE__);
        return static_cast<int64_t>(0);
    }
    // Darwin reports in bytes
    return static_cast<int64_t>(info.resident_size);

#    else  // Linux

    int64_t           rss = 0;
    std::stringstream fio;
    fio << "/proc/" << get_rusage_pid() << "/statm";
    FILE* fp = fopen(fio.str().c_str(), "r");
    if(fp && fscanf(fp, "%*s%ld", &rss) == 1)
    {
        fclose(fp);
        return static_cast<int64_t>(rss * units::get_page_size());
    }

    if(fp)
        fclose(fp);

    return static_cast<int64_t>(0);

#    endif
#elif defined(TIMEMORY_WINDOWS)
    DWORD                   processID = GetCurrentProcessId();
    HANDLE                  hProcess;
    PROCESS_MEMORY_COUNTERS pmc;

    hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processID);
    if(NULL == hProcess)
        return static_cast<int64_t>(0);

    int64_t nsize = 0;
    if(GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc)))
        nsize = static_cast<int64_t>(pmc.WorkingSetSize);

    CloseHandle(hProcess);
    return nsize;
#else
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//

inline int64_t
tim::get_virt_mem()
{
#if defined(TIMEMORY_UNIX)
#    if defined(TIMEMORY_MACOS)
    // OSX
    // kern_return_t kret;
    // task_t task;
    // kret = task_for_pid(mach_task_self(), get_rusage_pid(), &task);

    struct mach_task_basic_info info;
    mach_msg_type_number_t      infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if(task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info,
                 &infoCount) != KERN_SUCCESS)
    {
        TIMEMORY_PRINTF_WARNING(stderr,
                                "Warning! %s@'%s':%i :: task_info(...) != KERN_SUCCESS\n",
                                __FUNCTION__, __FILE__, __LINE__);
        return static_cast<int64_t>(0);
    }
    // Darwin reports in bytes
    return static_cast<int64_t>(info.virtual_size);

#    else  // Linux

#        if defined(TIMEM_DEBUG)
    if(get_env("TIMEM_DEBUG", false))
        TIMEMORY_PRINTF(stderr, "[%s@%s:%i] using pid %li\n", __func__, __FILE__,
                        __LINE__, (long int) get_rusage_pid());
#        endif

    std::string fstatm = [&]() {
        std::stringstream fio;
        fio << "/proc/" << get_rusage_pid() << "/statm";
        return fio.str();
    }();
    int64_t       vm_size = 0;
    std::ifstream ifs;
    ifs.open(fstatm.c_str());
    if(ifs)
        ifs >> vm_size;
    ifs.close();
    return static_cast<int64_t>(vm_size * units::get_page_size());

#    endif
#elif defined(TIMEMORY_WINDOWS)
    return static_cast<int64_t>(0);
#else
    return static_cast<int64_t>(0);
#endif
}

//======================================================================================//
