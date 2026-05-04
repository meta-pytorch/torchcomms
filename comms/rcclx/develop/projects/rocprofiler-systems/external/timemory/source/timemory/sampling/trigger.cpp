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

#ifndef TIMEMORY_SAMPLING_TRIGGER_CPP_
#define TIMEMORY_SAMPLING_TRIGGER_CPP_

#if !defined(TIMEMORY_SAMPLING_TRIGGER_HPP_)
#    include "timemory/sampling/trigger.hpp"
#    define TIMEMORY_SAMPLING_INLINE
#else
#    define TIMEMORY_SAMPLING_INLINE inline
#endif

namespace tim
{
namespace sampling
{
// clang-format off
TIMEMORY_SAMPLING_INLINE
// clang-format on
trigger::trigger(int _signal, pid_t _pid, int64_t _tim_tid, long _sys_tid)
: m_signal{ _signal }
, m_pid{ _pid }
, m_sys_tid{ _sys_tid }
, m_tim_tid{ _tim_tid }
{}

// clang-format off
TIMEMORY_SAMPLING_INLINE
// clang-format on
trigger::trigger(trigger&& rhs) noexcept
: m_initialized{ rhs.m_initialized }
, m_is_active{ rhs.m_is_active }
, m_signal{ rhs.m_signal }
, m_pid{ rhs.m_pid }
, m_sys_tid{ rhs.m_sys_tid }
, m_tim_tid{ rhs.m_tim_tid }
{
    rhs.m_initialized = false;
    rhs.m_is_active   = false;
}

// clang-format off
TIMEMORY_SAMPLING_INLINE
// clang-format on
trigger&
trigger::operator=(trigger&& rhs) noexcept
{
    if(this == &rhs)
        return *this;

    m_initialized     = rhs.m_initialized;
    m_is_active       = rhs.m_is_active;
    m_signal          = rhs.m_signal;
    m_pid             = rhs.m_pid;
    m_sys_tid         = rhs.m_sys_tid;
    m_tim_tid         = rhs.m_tim_tid;
    rhs.m_initialized = false;
    rhs.m_is_active   = false;
    return *this;
}

// clang-format off
TIMEMORY_SAMPLING_INLINE
// clang-format on
void
trigger::set_signal(int _v)
{
    if(!m_is_active)
        m_signal = _v;
    TIMEMORY_PREFER(!m_is_active)
        << "trigger::" << __FUNCTION__ << " ignored. trigger already active\n";
}

// clang-format off
TIMEMORY_SAMPLING_INLINE
// clang-format on
std::string
trigger::as_string() const
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

#endif  // TIMEMORY_SAMPLING_TRIGGER_CPP_
