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

#include "timemory/log/logger.hpp"
#include "timemory/log/macros.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/process/process.hpp"
#include "timemory/process/threading.hpp"
#include "timemory/sampling/trigger.hpp"

#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(TIMEMORY_LINUX)
#    include <linux/perf_event.h>
#    include <sys/ioctl.h>
#endif

namespace tim
{
namespace sampling
{
/// \struct tim::sampling::overflow
/// \brief A very simple trigger for when signals are delivered when some HW counter
/// overflows, e.g. via Linux Perf
struct overflow : public trigger
{
    using state_functor_t = bool (*)(int, pid_t, long, int64_t);

    overflow(int _signal, state_functor_t, state_functor_t, state_functor_t,
             int64_t = threading::get_id(), long = threading::get_sys_tid());

    ~overflow() override { stop(); }

    overflow(const overflow&) = delete;
    overflow& operator=(const overflow&) = delete;

    overflow(overflow&& rhs) noexcept;
    overflow& operator=(overflow&& rhs) noexcept;

    bool initialize() override;
    bool start() override;
    bool stop() override;

    friend std::ostream& operator<<(std::ostream& _os, const overflow& _v)
    {
        return (_os << _v.as_string());
    }

private:
    std::string as_string() const;

private:
    state_functor_t m_initer  = nullptr;
    state_functor_t m_starter = nullptr;
    state_functor_t m_stopper = nullptr;
};
//
inline overflow::overflow(int _signal, state_functor_t _initer, state_functor_t _starter,
                          state_functor_t _stopper, int64_t _tim_tid, long _sys_tid)
: trigger{ _signal, process::get_id(), _tim_tid, _sys_tid }
, m_initer{ _initer }
, m_starter{ _starter }
, m_stopper{ _stopper }
{}

inline overflow::overflow(overflow&& rhs) noexcept
: trigger{ std::move(rhs) }
, m_initer{ rhs.m_initer }
, m_starter{ rhs.m_starter }
, m_stopper{ rhs.m_stopper }
{
    rhs.m_initer  = nullptr;
    rhs.m_starter = nullptr;
    rhs.m_stopper = nullptr;
}

inline overflow&
overflow::operator=(overflow&& rhs) noexcept
{
    if(this == &rhs)
        return *this;

    trigger::operator=(std::move(rhs));
    m_initer         = rhs.m_initer;
    m_starter        = rhs.m_starter;
    m_stopper        = rhs.m_stopper;
    rhs.m_initer     = nullptr;
    rhs.m_starter    = nullptr;
    rhs.m_stopper    = nullptr;
    return *this;
}

inline bool
overflow::initialize()
{
    if(m_is_active)
        return false;

    if(m_initer)
    {
        TIMEMORY_REQUIRE(m_initer(m_signal, m_pid, m_sys_tid, m_tim_tid))
            << "Failed to init perf event: " << *this << " (errno: " << strerror(errno)
            << ")";
    }

    m_initialized = true;
    return true;
}

inline bool
overflow::start()
{
    if(m_is_active)
        return false;

    initialize();

    if(m_starter)
    {
        TIMEMORY_REQUIRE(m_starter(m_signal, m_pid, m_sys_tid, m_tim_tid))
            << "Failed to start perf event: " << *this << " (errno: " << strerror(errno)
            << ")";
        m_is_active = true;
    }

    return m_is_active;
}

inline bool
overflow::stop()
{
    if(m_initialized && m_pid == process::get_id())
    {
        if(m_stopper)
        {
            TIMEMORY_REQUIRE(m_stopper(m_signal, m_pid, m_sys_tid, m_tim_tid))
                << "Failed to stop perf event: " << *this
                << " (errno: " << strerror(errno) << ")";
            m_is_active = false;
        }
    }

    return !m_is_active;
}

inline std::string
overflow::as_string() const
{
    std::stringstream _os;
    _os << std::boolalpha;
    _os << "pid=" << m_pid << ", tid=" << m_tim_tid << ", sys_tid=" << m_sys_tid
        << ", signal=" << m_signal << ", init=" << m_initialized
        << ", is_active=" << m_is_active;
    return _os.str();
}
}  // namespace sampling
}  // namespace tim
