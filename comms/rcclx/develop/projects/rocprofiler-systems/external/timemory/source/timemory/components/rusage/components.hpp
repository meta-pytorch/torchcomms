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

#include "timemory/components/base.hpp"
#include "timemory/components/rusage/backends.hpp"
#include "timemory/components/rusage/types.hpp"
#include "timemory/components/timing/backends.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          Resource Usage types
//
//--------------------------------------------------------------------------------------//
/// \struct tim::component::peak_rss
/// \brief
/// this struct extracts the high-water mark (or a change in the high-water mark) of
/// the resident set size (RSS). Which is current amount of memory in RAM.
/// When used on a system with swap enabled, this value may fluctuate but should not
/// on an HPC system.
//
struct peak_rss : public base<peak_rss>
{
    static std::string label() { return "peak_rss"; }
    static std::string description()
    {
        return "Measures changes in the high-water mark for the amount of memory "
               "allocated in RAM. May fluctuate if swap is enabled";
    }
    static value_type record() { return get_peak_rss(); }
    double            get() const
    {
        auto val = base_type::load();
        return val / static_cast<double>(base_type::get_unit());
    }
    double get_display() const { return get(); }

    void start() { value = record(); }
    void stop()
    {
        value = (record() - value);
        accum += value;
    }

    /// sample a measurement
    void sample() { accum = value = std::max<int64_t>(value, record()); }

    /// sample a measurement from cached data
    void sample(const cache_type& _cache)
    {
        accum = value = std::max<int64_t>(value, record(_cache));
    }

    /// read the value from cached data
    static value_type record(const cache_type& _cache) { return _cache.get_peak_rss(); }

    /// start a measurement using the cached data
    void start(const cache_type& _cache) { value = record(_cache); }

    /// stop a measurement using the cached data
    void stop(const cache_type& _cache)
    {
        auto tmp   = record(_cache);
        auto delta = tmp - value;
        accum      = std::max(static_cast<const value_type&>(accum), delta);
        value      = tmp;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::page_rss
/// \brief
/// this struct measures the resident set size (RSS) currently allocated in pages of
/// memory. Unlike the peak_rss, this value will fluctuate as memory gets freed and
/// allocated
//
struct page_rss : public base<page_rss, int64_t>
{
    using value_type  = int64_t;
    using result_type = double;
    using this_type   = page_rss;
    using base_type   = base<this_type, value_type>;

    static std::string label() { return "page_rss"; }
    static std::string description()
    {
        return "Amount of memory allocated in pages of memory. Unlike peak_rss, value "
               "will fluctuate as memory is freed/allocated";
    }
    static value_type record() { return get_page_rss(); }
    double            get() const
    {
        auto val = base_type::load();
        return val / static_cast<double>(base_type::get_unit());
    }
    double get_display() const { return get(); }
    void   start() { value = record(); }
    void   stop()
    {
        value = (record() - value);
        accum += value;
    }
    void sample() { accum = value = std::max<int64_t>(value, record()); }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::num_io_in
/// \brief
/// the number of times the file system had to perform input.
//
struct num_io_in : public base<num_io_in>
{
    using value_type = int64_t;
    using base_type  = base<num_io_in>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "io_in"; }
    static std::string description()
    {
        return "Number of times the filesystem had to perform input";
    }
    static value_type record() { return get_num_io_in(); }
    value_type        get() const
    {
        auto val = base_type::load();
        return val;
    }
    value_type get_display() const { return get(); }
    void       start() { value = record(); }
    void       stop()
    {
        value = (record() - value);
        accum += value;
    }

    static value_type record(const cache_type& _cache) { return _cache.get_num_io_in(); }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        auto tmp = record(_cache);
        accum += (tmp - value);
        value = tmp;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::num_io_out
/// \brief
/// the number of times the file system had to perform output.
//
struct num_io_out : public base<num_io_out>
{
    using value_type = int64_t;
    using base_type  = base<num_io_out>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "io_out"; }
    static std::string description()
    {
        return "Number of times the filesystem had to perform output";
    }
    static value_type record() { return get_num_io_out(); }
    value_type        get() const
    {
        auto val = base_type::load();
        return val;
    }
    value_type get_display() const { return get(); }
    void       start() { value = record(); }
    void       stop()
    {
        value = (record() - value);
        accum += value;
    }

    static value_type record(const cache_type& _cache) { return _cache.get_num_io_out(); }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        auto tmp = record(_cache);
        accum += (tmp - value);
        value = tmp;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::num_minor_page_faults
/// \brief
/// the number of page faults serviced without any I/O activity; here I/O activity is
/// avoided by reclaiming a page frame from the list of pages awaiting reallocation.
//
struct num_minor_page_faults : public base<num_minor_page_faults>
{
    using value_type = int64_t;
    using base_type  = base<num_minor_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "minor_page_faults"; }
    static std::string description()
    {
        return "Number of page faults serviced without any I/O activity via 'reclaiming' "
               "a page frame from the list of pages awaiting reallocation";
    }
    static value_type record() { return get_num_minor_page_faults(); }
    value_type        get() const
    {
        auto val = base_type::load();
        return val;
    }
    value_type get_display() const { return get(); }
    void       start() { value = record(); }
    void       stop()
    {
        value = (record() - value);
        accum += value;
    }

    static value_type record(const cache_type& _cache)
    {
        return _cache.get_num_minor_page_faults();
    }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        auto tmp = record(_cache);
        accum += (tmp - value);
        value = tmp;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::num_major_page_faults
/// \brief
/// the number of page faults serviced that required I/O activity.
//
struct num_major_page_faults : public base<num_major_page_faults>
{
    using value_type = int64_t;
    using base_type  = base<num_major_page_faults>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "major_page_faults"; }
    static std::string description()
    {
        return "Number of page faults serviced that required I/O activity";
    }
    static value_type record() { return get_num_major_page_faults(); }
    value_type        get() const
    {
        auto val = base_type::load();
        return val;
    }
    value_type get_display() const { return get(); }
    void       start() { value = record(); }
    void       stop()
    {
        value = (record() - value);
        accum += value;
    }

    static value_type record(const cache_type& _cache)
    {
        return _cache.get_num_major_page_faults();
    }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        auto tmp = record(_cache);
        accum += (tmp - value);
        value = tmp;
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::voluntary_context_switch
/// \brief
/// the number of times a context switch resulted due to a process voluntarily giving up
/// the processor before its time slice was completed (usually to await availability of a
/// resource).
//
struct voluntary_context_switch : public base<voluntary_context_switch>
{
    using value_type = int64_t;
    using base_type  = base<voluntary_context_switch>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "vol_cxt_swch"; }
    static std::string description()
    {
        return "Number of context switches due to a process voluntarily giving up the "
               "processor before its time slice was completed";
    }
    static value_type record() { return get_num_voluntary_context_switch(); }
    value_type        get_display() const
    {
        auto val = base_type::load();
        return val;
    }
    value_type get() const { return get_display(); }
    void       start() { value = record(); }
    void       stop()
    {
        value = (record() - value);
        accum += value;
    }

    static value_type record(const cache_type& _cache)
    {
        return _cache.get_num_voluntary_context_switch();
    }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        auto tmp = record(_cache);
        accum += (tmp - value);
        value = tmp;
    }
};

using vol_cxt_switch = voluntary_context_switch;

//--------------------------------------------------------------------------------------//
/// \struct tim::component::priority_context_switch
/// \brief
/// the number of times a context switch resulted due to a higher priority process
/// becoming runnable or because the current process exceeded its time slice
//
struct priority_context_switch : public base<priority_context_switch>
{
    using value_type = int64_t;
    using base_type  = base<priority_context_switch>;

    static const short                   precision    = 0;
    static const short                   width        = 3;
    static const std::ios_base::fmtflags format_flags = {};

    static std::string label() { return "prio_cxt_swch"; }
    static std::string description()
    {
        return "Number of context switch due to higher priority process becoming runnable"
               " or because the current process exceeded its time slice";
    }
    static value_type record() { return get_num_priority_context_switch(); }
    value_type        get_display() const
    {
        auto val = base_type::load();
        return val;
    }
    value_type get() const { return get_display(); }
    void       start() { value = record(); }
    void       stop()
    {
        value = (record() - value);
        accum += value;
    }

    static value_type record(const cache_type& _cache)
    {
        return _cache.get_num_priority_context_switch();
    }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        auto tmp = record(_cache);
        accum += (tmp - value);
        value = tmp;
    }
};

using prio_cxt_switch = priority_context_switch;

//--------------------------------------------------------------------------------------//
/// \struct tim::component::virtual_memory
/// \brief
/// this struct extracts the virtual memory usage
//
struct virtual_memory : public base<virtual_memory>
{
    using value_type = int64_t;
    using base_type  = base<virtual_memory, value_type>;

    static std::string label() { return "virtual_memory"; }
    static std::string description() { return "Records the change in virtual memory"; }
    static value_type  record() { return get_virt_mem(); }
    double             get() const
    {
        auto val = base_type::load();
        return val / static_cast<double>(base_type::get_unit());
    }
    double get_display() const { return get(); }
    void   start() { value = record(); }
    void   stop()
    {
        value = (record() - value);
        accum += value;
    }
    void sample() { accum = value = std::max<int64_t>(value, record()); }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::user_mode_time
/// \brief This is the total amount of time spent executing in user mode
//
struct user_mode_time : public base<user_mode_time, int64_t>
{
    using ratio_t    = std::micro;
    using value_type = int64_t;
    using this_type  = user_mode_time;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "user_mode_time"; }
    static std::string description()
    {
        return "CPU time spent executing in user mode (via rusage)";
    }
    static value_type record() { return get_user_mode_time(); }

    double get_display() const { return get(); }
    double get() const { return load() / static_cast<double>(base_type::get_unit()); }

    void start() { value = record(); }

    void stop()
    {
        auto tmp = record();
        if(tmp > value)
        {
            value = (tmp - value);
            accum += value;
        }
    }

    static value_type record(const cache_type& _cache)
    {
        return _cache.get_user_mode_time();
    }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        auto tmp = record(_cache);
        if(tmp > value)
        {
            accum += (tmp - value);
            value = tmp;
        }
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::kernel_mode_time
/// \brief This is the total amount of time spent executing in kernel mode
//
struct kernel_mode_time : public base<kernel_mode_time, int64_t>
{
    using ratio_t    = std::micro;
    using value_type = int64_t;
    using this_type  = kernel_mode_time;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "kernel_mode_time"; }
    static std::string description()
    {
        return "CPU time spent executing in kernel mode (via rusage)";
    }
    static value_type record() { return get_kernel_mode_time(); }

    double get_display() const { return get(); }
    double get() const { return load() / static_cast<double>(base_type::get_unit()); }

    void start() { value = record(); }

    void stop()
    {
        auto tmp = record();
        if(tmp > value)
        {
            value = (tmp - value);
            accum += value;
        }
    }

    static value_type record(const cache_type& _cache)
    {
        return _cache.get_kernel_mode_time();
    }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        auto tmp = record(_cache);
        if(tmp > value)
        {
            accum += (tmp - value);
            value = tmp;
        }
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::current_peak_rss
/// \brief
/// this struct extracts the absolute value of high-water mark of the resident set size
/// (RSS) at start and stop points. RSS is current amount of memory in RAM.
//
struct current_peak_rss : public base<current_peak_rss, std::pair<int64_t, int64_t>>
{
    using unit_type         = std::pair<int64_t, int64_t>;
    using display_unit_type = std::pair<std::string, std::string>;
    using result_type       = std::pair<double, double>;
    using this_type         = current_peak_rss;

    static std::string label() { return "current_peak_rss"; }
    static std::string description()
    {
        return "Absolute value of high-water mark of memory allocation in RAM";
    }

    static value_type record() { return value_type{ get_peak_rss(), 0 }; }

    void start() { value = record(); }

    void stop()
    {
        using namespace tim::component::operators;
        value = value_type{ value.first, record().first };
        accum = std::max(accum, value);
    }

    static value_type record(const cache_type& _cache)
    {
        return value_type{ _cache.get_peak_rss(), 0 };
    }

    void start(const cache_type& _cache) { value = record(_cache); }

    void stop(const cache_type& _cache)
    {
        value = value_type{ value.first, record(_cache).first };
        accum = std::max(accum, value);
    }

public:
    std::string get_display() const
    {
        std::stringstream ss;
        std::stringstream ssv;
        std::stringstream ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = get_display_unit();

        auto _val = get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << ", " << ssr.str();
        return ss.str();
    }

    result_type get() const
    {
        result_type data = base_type::load();
        data.first /= get_unit().first;
        data.second /= get_unit().second;
        return data;
    }

    static std::pair<double, double> unit()
    {
        return std::pair<double, double>{ units::megabyte, units::megabyte };
    }

    static std::vector<std::string> display_unit_array()
    {
        return std::vector<std::string>{ get_display_unit().first,
                                         get_display_unit().second };
    }

    static std::vector<std::string> label_array()
    {
        return std::vector<std::string>{ "start peak rss", " stop peak rss" };
    }

    static display_unit_type display_unit() { return display_unit_type{ "MB", "MB" }; }

    static std::pair<double, double> unit_array() { return unit(); }

    static std::vector<std::string> description_array()
    {
        return std::vector<std::string>{ "Resident set size at start",
                                         "Resident set size at stop" };
    }

    static unit_type get_unit()
    {
        static auto  _instance = this_type::unit();
        static auto& _mem      = _instance;

        if(settings::memory_units().length() > 0)
        {
            _mem.first  = std::get<1>(units::get_memory_unit(settings::memory_units()));
            _mem.second = std::get<1>(units::get_memory_unit(settings::memory_units()));
        }

        return _mem;
    }

    static display_unit_type get_display_unit()
    {
        static display_unit_type _instance = this_type::display_unit();
        static auto&             _mem      = _instance;

        if(settings::memory_units().length() > 0)
        {
            _mem.first  = std::get<0>(units::get_memory_unit(settings::memory_units()));
            _mem.second = std::get<0>(units::get_memory_unit(settings::memory_units()));
        }

        return _mem;
    }
};

//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
