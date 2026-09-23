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

#include "timemory/components/macros.hpp"
#include "timemory/components/timing/backends.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/components/timing/wall_clock.hpp"

TIMEMORY_DECLARE_COMPONENT(child_system_clock)
TIMEMORY_DECLARE_COMPONENT(child_user_clock)
TIMEMORY_DECLARE_COMPONENT(child_cpu_clock)
TIMEMORY_DECLARE_COMPONENT(child_cpu_util)
//
TIMEMORY_STATISTICS_TYPE(component::child_system_clock, double)
TIMEMORY_STATISTICS_TYPE(component::child_user_clock, double)
TIMEMORY_STATISTICS_TYPE(component::child_cpu_clock, double)
TIMEMORY_STATISTICS_TYPE(component::child_cpu_util, double)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::child_system_clock,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::child_user_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::child_cpu_clock, true_type)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::child_system_clock,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::child_user_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::child_cpu_clock, true_type)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::child_cpu_util, true_type)
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_flamegraph, component::child_system_clock,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_flamegraph, component::child_user_clock,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_flamegraph, component::child_cpu_clock, true_type)
//
namespace tim
{
namespace trait
{
//
template <>
struct derivation_types<component::child_cpu_util>
{
    using type = type_list<type_list<component::wall_clock, component::child_cpu_clock>,
                           type_list<component::wall_clock, component::child_user_clock,
                                     component::child_system_clock>>;
};
//
}  // namespace trait
//
namespace component
{
/// \struct tim::component::child_system_clock
/// \brief Similar to \ref system_clock except only reports values for child processes
struct child_system_clock : public base<child_system_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<child_system_clock, value_type>;

    static std::string label() { return "sys"; }
    static std::string description() { return "CPU time spent in kernel-mode"; }
    static value_type  record() noexcept
    {
        return tim::get_child_clock_system_now<int64_t, ratio_t>();
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
/// \struct tim::component::child_user_clock
/// \brief Similar to \ref user_clock except only reports values for child processes
struct child_user_clock : public base<child_user_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<child_user_clock, value_type>;

    static std::string label() { return "user"; }
    static std::string description() { return "CPU time spent in user-mode"; }
    static value_type  record() noexcept
    {
        return tim::get_child_clock_user_now<int64_t, ratio_t>();
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
/// \struct tim::component::child_cpu_clock
/// \brief Similar to \ref cpu_clock except only reports values for child processes
struct child_cpu_clock : public base<child_cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<child_cpu_clock, value_type>;

    static std::string label() { return "cpu"; }
    static std::string description()
    {
        return "Total CPU time spent in both user- and kernel-mode";
    }
    static value_type record() noexcept
    {
        return tim::get_child_clock_cpu_now<int64_t, ratio_t>();
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
/// \struct tim::component::child_cpu_util
/// \brief Similar to \ref cpu_util except only reports values for child processes
struct child_cpu_util : public base<child_cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<child_cpu_util, value_type>;

    static std::string label() { return "cpu_util"; }
    static std::string description()
    {
        return "Percentage of CPU-clock time divided by wall-clock time";
    }
    static value_type record()
    {
        return value_type(child_cpu_clock::record(), wall_clock::record());
    }

    double get() const noexcept
    {
        auto   _data = base_type::load();
        double denom = (_data.second > 0) ? _data.second : 1;
        double numer = (_data.second > 0) ? _data.first : 0;
        return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
    }

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

public:
    bool is_derived() const noexcept { return m_derive; }

    bool assemble(const wall_clock* wc, const child_cpu_clock* cc) noexcept
    {
        if(wc && cc)
            m_derive = true;
        return m_derive;
    }

    bool assemble(const wall_clock* wc, const child_user_clock* uc,
                  const child_system_clock* sc) noexcept
    {
        if(wc && uc && sc)
            m_derive = true;
        return m_derive;
    }

    bool derive(const wall_clock* wc, const child_cpu_clock* cc) noexcept
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

    bool derive(const wall_clock* wc, const child_user_clock* uc,
                const child_system_clock* sc) noexcept
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

public:
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

private:
    bool m_derive = false;
};
//
}  // namespace component
}  // namespace tim
