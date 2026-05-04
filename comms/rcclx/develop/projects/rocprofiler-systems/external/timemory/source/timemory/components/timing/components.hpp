//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include "timemory/components/base/declaration.hpp"
#include "timemory/components/timing/backends.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/components/timing/wall_clock.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include <utility>

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
/// \struct tim::component::system_clock
/// \brief this component extracts only the CPU time spent in kernel-mode.
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct system_clock : public base<system_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<system_clock, value_type>;

    static std::string label() { return "system_clock"; }
    static std::string description() { return "CPU time spent in kernel-mode"; }
    static value_type  record() noexcept
    {
        return tim::get_clock_system_now<int64_t, ratio_t>();
    }
    double get() const noexcept
    {
        return load() / static_cast<double>(base_type::get_unit());
    }
    double get_display() const noexcept { return get(); }
    void   start() noexcept { value = record(); }
    void   stop() noexcept
    {
        value = (record() - value);
        accum += value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::user_clock
/// \brief this component extracts only the CPU time spent in user-mode.
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct user_clock : public base<user_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<user_clock, value_type>;

    static std::string label() { return "user_clock"; }
    static std::string description() { return "CPU time spent in user-mode"; }
    static value_type  record() noexcept
    {
        return tim::get_clock_user_now<int64_t, ratio_t>();
    }
    double get() const noexcept
    {
        return load() / static_cast<double>(base_type::get_unit());
    }
    double get_display() const noexcept { return get(); }
    void   start() noexcept { value = record(); }
    void   stop() noexcept
    {
        value = (record() - value);
        accum += value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::cpu_clock
/// \brief this component extracts only the CPU time spent in both user- and kernel- mode.
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct cpu_clock : public base<cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<cpu_clock, value_type>;

    static std::string label() { return "cpu_clock"; }
    static std::string description()
    {
        return "Total CPU time spent in both user- and kernel-mode";
    }
    static value_type record() noexcept
    {
        return tim::get_clock_cpu_now<int64_t, ratio_t>();
    }
    double get() const noexcept
    {
        return load() / static_cast<double>(base_type::get_unit());
    }
    double get_display() const noexcept { return get(); }
    void   start() noexcept { value = record(); }
    void   stop() noexcept
    {
        value = (record() - value);
        accum += value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::monotonic_clock
/// \brief clock that increments monotonically, tracking the time since an arbitrary
/// point, and will continue to increment while the system is asleep.
struct monotonic_clock : public base<monotonic_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<monotonic_clock, value_type>;

    static std::string label() { return "monotonic_clock"; }
    static std::string description()
    {
        return "Wall-clock timer which will continue to increment even while the system "
               "is asleep";
    }
    static value_type record()
    {
        return tim::get_clock_monotonic_now<int64_t, ratio_t>();
    }
    double get() const noexcept
    {
        return load() / static_cast<double>(base_type::get_unit());
    }
    double get_display() const noexcept { return get(); }
    void   start() noexcept { value = record(); }
    void   stop() noexcept
    {
        value = (record() - value);
        accum += value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::monotonic_raw_clock
/// \brief clock that increments monotonically, tracking the time since an arbitrary point
/// like CLOCK_MONOTONIC.  However, this clock is unaffected by frequency or time
/// adjustments. It should not be compared to other system time sources.
struct monotonic_raw_clock : public base<monotonic_raw_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<monotonic_raw_clock, value_type>;

    static std::string label() { return "monotonic_raw_clock"; }
    static std::string description()
    {
        return "Wall-clock timer unaffected by frequency or time adjustments in system "
               "time-of-day clock";
    }
    static value_type record()
    {
        return tim::get_clock_monotonic_raw_now<int64_t, ratio_t>();
    }
    double get() const noexcept
    {
        return load() / static_cast<double>(base_type::get_unit());
    }
    double get_display() const noexcept { return get(); }
    void   start() noexcept { value = record(); }
    void   stop() noexcept
    {
        value = (record() - value);
        accum += value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::thread_cpu_clock
/// \brief this clock measures the CPU time within the current thread (excludes
/// sibling/child threads).
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct thread_cpu_clock : public base<thread_cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<thread_cpu_clock, value_type>;

    static std::string label() { return "thread_cpu_clock"; }
    static std::string description() { return "CPU-clock timer for the calling thread"; }
    static value_type  record() noexcept
    {
        return tim::get_clock_thread_now<int64_t, ratio_t>();
    }
    double get() const noexcept
    {
        return load() / static_cast<double>(base_type::get_unit());
    }
    double get_display() const noexcept { return get(); }
    void   start() noexcept { value = record(); }
    void   stop() noexcept
    {
        value = (record() - value);
        accum += value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::process_cpu_clock
/// \brief this clock measures the CPU time within the current process (excludes child
/// processes).
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct process_cpu_clock : public base<process_cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<process_cpu_clock, value_type>;

    static std::string label() { return "process_cpu_clock"; }
    static std::string description()
    {
        return "CPU-clock timer for the calling process (all threads)";
    }
    static value_type record() noexcept
    {
        return tim::get_clock_process_now<int64_t, ratio_t>();
    }
    double get() const noexcept
    {
        return load() / static_cast<double>(base_type::get_unit());
    }
    double get_display() const noexcept { return get(); }
    void   start() noexcept { value = record(); }
    void   stop() noexcept
    {
        value = (record() - value);
        accum += value;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::cpu_util
/// \brief  this computes the CPU utilization percentage for the calling process and child
/// processes.
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct cpu_util : public base<cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<cpu_util, value_type>;
    using this_type  = cpu_util;

    static std::string label() { return "cpu_util"; }
    static std::string description()
    {
        return "Percentage of CPU-clock time divided by wall-clock time";
    }
    static value_type record()
    {
        return value_type(cpu_clock::record(), wall_clock::record());
    }
    double get() const noexcept
    {
        const auto& _data = load();
        double      denom = (_data.second > 0) ? _data.second : 1;
        double      numer = (_data.second > 0) ? _data.first : 0;
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serialization() const noexcept { return get_display(); }
    double get_display() const noexcept { return get(); }

    void start() noexcept
    {
        if(!m_derive)
            value = record();
    }

    void stop() noexcept
    {
        using namespace tim::component::operators;
        if(!m_derive)
        {
            value = (record() - value);
            accum += value;
        }
    }

    this_type& operator+=(const this_type& rhs) noexcept
    {
        accum += rhs.accum;
        value += rhs.value;
        return *this;
    }

    this_type& operator-=(const this_type& rhs) noexcept
    {
        accum -= rhs.accum;
        value -= rhs.value;
        return *this;
    }

    bool assemble(const wall_clock* wc, const cpu_clock* cc) noexcept
    {
        if(wc && cc)
            m_derive = true;
        return m_derive;
    }

    bool assemble(const wall_clock* wc, const user_clock* uc,
                  const system_clock* sc) noexcept
    {
        if(wc && uc && sc)
            m_derive = true;
        return m_derive;
    }

    bool derive(const wall_clock* wc, const cpu_clock* cc) noexcept
    {
        if(m_derive && wc && cc)
        {
            value.first  = cc->get_value();
            value.second = wc->get_value();
            accum += value;
            return true;
        }
        return false;
    }

    bool derive(const wall_clock* wc, const user_clock* uc,
                const system_clock* sc) noexcept
    {
        if(m_derive && wc && uc && sc)
        {
            value.first  = uc->get_value() + sc->get_value();
            value.second = wc->get_value();
            accum += value;
            return true;
        }
        return false;
    }

    bool is_derived() const noexcept { return m_derive; }

private:
    bool m_derive = false;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::process_cpu_util
/// \brief this computes the CPU utilization percentage for ONLY the calling process
/// (excludes child processes).
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct process_cpu_util : public base<process_cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<process_cpu_util, value_type>;
    using this_type  = process_cpu_util;

    static std::string label() { return "process_cpu_util"; }
    static std::string description()
    {
        return "Percentage of CPU-clock time divided by wall-clock time for calling "
               "process (all threads)";
    }
    static value_type record()
    {
        return value_type(process_cpu_clock::record(), wall_clock::record());
    }
    double get() const noexcept
    {
        const auto& _data = load();
        double      denom = (_data.second > 0) ? _data.second : 1;
        double      numer = (_data.second > 0) ? _data.first : 0;
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serialization() const noexcept { return get_display(); }
    double get_display() const noexcept { return get(); }
    void   start() noexcept
    {
        if(!m_derive)
            value = record();
    }
    void stop() noexcept
    {
        using namespace tim::component::operators;
        if(!m_derive)
        {
            value = (record() - value);
            accum += value;
        }
    }

    this_type& operator+=(const this_type& rhs) noexcept
    {
        accum += rhs.accum;
        value += rhs.value;
        return *this;
    }

    this_type& operator-=(const this_type& rhs) noexcept
    {
        accum -= rhs.accum;
        value -= rhs.value;
        return *this;
    }

    bool assemble(const wall_clock* wc, const process_cpu_clock* cc) noexcept
    {
        if(wc && cc)
            m_derive = true;
        return m_derive;
    }

    bool derive(const wall_clock* wc, const process_cpu_clock* cc) noexcept
    {
        if(m_derive && wc && cc)
        {
            value.first  = cc->get_value();
            value.second = wc->get_value();
            accum += value;
            return true;
        }
        return false;
    }

    bool is_derived() const noexcept { return m_derive; }

private:
    bool m_derive = false;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::thread_cpu_util
/// \brief this computes the CPU utilization percentage for ONLY the calling thread
/// (excludes sibling and child threads).
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct thread_cpu_util : public base<thread_cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<thread_cpu_util, value_type>;
    using this_type  = thread_cpu_util;

    static std::string label() { return "thread_cpu_util"; }
    static std::string description()
    {
        return "Percentage of CPU-clock time divided by wall-clock time for calling "
               "thread";
    }
    static value_type record()
    {
        return value_type(thread_cpu_clock::record(), wall_clock::record());
    }
    double get() const noexcept
    {
        const auto& _data = load();
        double      denom = (_data.second > 0) ? _data.second : 1;
        double      numer = (_data.second > 0) ? _data.first : 0;
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }
    double serialization() const noexcept { return get_display(); }
    double get_display() const noexcept { return get(); }
    void   start() noexcept
    {
        if(!m_derive)
            value = record();
    }
    void stop() noexcept
    {
        using namespace tim::component::operators;
        if(!m_derive)
        {
            value = (record() - value);
            accum += value;
        }
    }

    this_type& operator+=(const this_type& rhs) noexcept
    {
        accum += rhs.accum;
        value += rhs.value;
        return *this;
    }

    this_type& operator-=(const this_type& rhs) noexcept
    {
        accum -= rhs.accum;
        value -= rhs.value;
        return *this;
    }

    bool assemble(const wall_clock* wc, const thread_cpu_clock* cc) noexcept
    {
        if(wc && cc)
            m_derive = true;
        return m_derive;
    }

    bool derive(const wall_clock* wc, const thread_cpu_clock* cc) noexcept
    {
        if(m_derive && wc && cc)
        {
            value.first  = cc->get_value();
            value.second = wc->get_value();
            accum += value;
            return true;
        }
        return false;
    }

    bool is_derived() const noexcept { return m_derive; }

private:
    bool m_derive = false;
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
