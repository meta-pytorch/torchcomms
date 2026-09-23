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

#ifndef TIMEMORY_SAMPLING_TRIGGER_HPP_
#    define TIMEMORY_SAMPLING_TRIGGER_HPP_
#endif

#include "timemory/log/logger.hpp"
#include "timemory/log/macros.hpp"
#include "timemory/process/process.hpp"
#include "timemory/process/threading.hpp"

namespace tim
{
namespace sampling
{
using sigaction_t = struct sigaction;

/// \struct tim::sampling::trigger
/// \brief this is the virtual base class for the objects which cause signals to be
/// delivered. For example, the \ref tim::sampling::timer triggers the delivery of a
/// signal at regular intervals based on a timer.
struct trigger
{
    trigger(int, pid_t = process::get_id(), int64_t = threading::get_id(),
            long = threading::get_sys_tid());

    virtual ~trigger() = default;

    trigger(const trigger&) = delete;
    trigger& operator=(const trigger&) = delete;

    trigger(trigger&& rhs) noexcept;
    trigger& operator=(trigger&& rhs) noexcept;

    virtual bool initialize() = 0;
    virtual bool start()      = 0;
    virtual bool stop()       = 0;

    virtual bool is_active() const { return m_is_active; }
    virtual bool is_initialized() const { return m_initialized; }

    auto signal() const { return m_signal; }
    auto get_pid() const { return m_pid; }
    auto get_tid() const { return m_tim_tid; }
    auto get_sys_tid() const { return m_sys_tid; }

    void set_signal(int);

protected:
    std::string as_string() const;

protected:
    mutable bool m_initialized = false;
    mutable bool m_is_active   = false;
    int          m_signal      = -1;
    int          m_pid         = process::get_id();
    long         m_sys_tid     = threading::get_sys_tid();
    int64_t      m_tim_tid     = threading::get_id();
};
}  // namespace sampling
}  // namespace tim

#include "timemory/sampling/trigger.cpp"
