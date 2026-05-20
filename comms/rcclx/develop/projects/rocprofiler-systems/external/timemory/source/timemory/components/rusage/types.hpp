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
#include "timemory/enum.h"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_COMPONENT(peak_rss)
TIMEMORY_DECLARE_COMPONENT(page_rss)
TIMEMORY_DECLARE_COMPONENT(num_io_in)
TIMEMORY_DECLARE_COMPONENT(num_io_out)
TIMEMORY_DECLARE_COMPONENT(num_minor_page_faults)
TIMEMORY_DECLARE_COMPONENT(num_major_page_faults)
TIMEMORY_DECLARE_COMPONENT(voluntary_context_switch)
TIMEMORY_DECLARE_COMPONENT(priority_context_switch)
TIMEMORY_DECLARE_COMPONENT(virtual_memory)
TIMEMORY_DECLARE_COMPONENT(user_mode_time)
TIMEMORY_DECLARE_COMPONENT(kernel_mode_time)
TIMEMORY_DECLARE_COMPONENT(current_peak_rss)
//
// Fully deprecated
//
TIMEMORY_DECLARE_COMPONENT(num_swap)
TIMEMORY_DECLARE_COMPONENT(stack_rss)
TIMEMORY_DECLARE_COMPONENT(data_rss)
TIMEMORY_DECLARE_COMPONENT(num_msg_sent)
TIMEMORY_DECLARE_COMPONENT(num_msg_recv)
TIMEMORY_DECLARE_COMPONENT(num_signals)

namespace tim
{
namespace resource_usage
{
namespace alias
{
using pair_dd_t = std::pair<double, double>;
}  // namespace alias
}  // namespace resource_usage
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//

TIMEMORY_SET_COMPONENT_API(component::peak_rss, project::timemory, category::memory,
                           category::resource_usage, os::agnostic)

TIMEMORY_SET_COMPONENT_API(component::current_peak_rss, project::timemory,
                           category::memory, category::resource_usage, os::agnostic)

TIMEMORY_SET_COMPONENT_API(component::page_rss, project::timemory, category::memory,
                           category::resource_usage, os::agnostic)

TIMEMORY_SET_COMPONENT_API(component::virtual_memory, project::timemory, category::memory,
                           category::resource_usage, os::supports_linux)

TIMEMORY_SET_COMPONENT_API(component::num_io_in, project::timemory, category::io,
                           category::resource_usage, os::supports_unix)

TIMEMORY_SET_COMPONENT_API(component::num_io_out, project::timemory, category::io,
                           category::resource_usage, os::supports_unix)

TIMEMORY_SET_COMPONENT_API(component::num_minor_page_faults, project::timemory,
                           category::resource_usage, os::supports_unix)

TIMEMORY_SET_COMPONENT_API(component::num_major_page_faults, project::timemory,
                           category::resource_usage, os::supports_unix)

TIMEMORY_SET_COMPONENT_API(component::voluntary_context_switch, project::timemory,
                           category::resource_usage, os::supports_unix)

TIMEMORY_SET_COMPONENT_API(component::priority_context_switch, project::timemory,
                           category::resource_usage, os::supports_unix)

TIMEMORY_SET_COMPONENT_API(component::user_mode_time, project::timemory, category::timing,
                           os::supports_unix)

TIMEMORY_SET_COMPONENT_API(component::kernel_mode_time, project::timemory,
                           category::timing, os::supports_unix)

//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_STATISTICS_TYPE(component::peak_rss, double)
TIMEMORY_STATISTICS_TYPE(component::page_rss, double)
TIMEMORY_STATISTICS_TYPE(component::num_io_in, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_io_out, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_minor_page_faults, int64_t)
TIMEMORY_STATISTICS_TYPE(component::num_major_page_faults, int64_t)
TIMEMORY_STATISTICS_TYPE(component::voluntary_context_switch, int64_t)
TIMEMORY_STATISTICS_TYPE(component::priority_context_switch, int64_t)
TIMEMORY_STATISTICS_TYPE(component::virtual_memory, double)
TIMEMORY_STATISTICS_TYPE(component::user_mode_time, double)
TIMEMORY_STATISTICS_TYPE(component::kernel_mode_time, double)
TIMEMORY_STATISTICS_TYPE(component::current_peak_rss, resource_usage::alias::pair_dd_t)

//--------------------------------------------------------------------------------------//
//
//                              REPORT SUM
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(report_sum, component::current_peak_rss, false_type)

//--------------------------------------------------------------------------------------//
//
//                              FILE SAMPLER
//
//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_LINUX) || (defined(TIMEMORY_UNIX) && !defined(TIMEMORY_MACOS))

TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(file_sampler, component::virtual_memory, true_type)

#endif

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(custom_unit_printing, component::current_peak_rss,
                               true_type)

//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::stack_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::data_rss, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_recv, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_msg_sent, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_signals, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_swap, false_type)

//
//      WINDOWS (non-UNIX)
//
#if !defined(TIMEMORY_UNIX)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_in, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_out, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_major_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_minor_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::virtual_memory, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_mode_time, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::kernel_mode_time, false_type)
//
#endif

//
//      UNIX
//
#if defined(TIMEMORY_UNIX)

/// \macro TIMEMORY_USE_UNMAINTAINED_RUSAGE
/// \brief This macro enables the globally disable rusage structures that are
/// unmaintained by the Linux kernel and are zero on macOS
///
#    if !defined(TIMEMORY_USE_UNMAINTAINED_RUSAGE) && defined(TIMEMORY_MACOS)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_in, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::num_io_out, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::virtual_memory, false_type)

#    endif  // !defined(TIMEMORY_USE_UNMAINTAINED_RUSAGE) && defined(TIMEMORY_MACOS)

#endif

//--------------------------------------------------------------------------------------//
//
//                                  ECHO ENABLED
//
//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_WINDOWS)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::current_peak_rss, false_type)
#endif

//--------------------------------------------------------------------------------------//
//
//                              IS TIMING CATEGORY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::user_mode_time, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::kernel_mode_time, true_type)

//--------------------------------------------------------------------------------------//
//
//                              IS MEMORY CATEGORY
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::peak_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_io_in, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_io_out, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_minor_page_faults,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::num_major_page_faults,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::voluntary_context_switch,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::priority_context_switch,
                               true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, component::virtual_memory, true_type)

//--------------------------------------------------------------------------------------//
//
//                                 REPORT UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::num_minor_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::num_major_page_faults, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::voluntary_context_switch,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::priority_context_switch,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::num_io_in, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::num_io_out, false_type)

//--------------------------------------------------------------------------------------//
//
//                              USES TIMING UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::user_mode_time, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::kernel_mode_time, true_type)

//--------------------------------------------------------------------------------------//
//
//                              USES MEMORY UNITS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::peak_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::page_rss, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, component::virtual_memory, true_type)

//--------------------------------------------------------------------------------------//
//
//                              RUSAGE_CACHE
//
//--------------------------------------------------------------------------------------//

namespace tim
{
struct rusage_cache;
struct rusage_cache_type
{
    using type = rusage_cache;
};
}  // namespace tim

TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::peak_rss, rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::num_io_in, rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::num_io_out, rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::num_minor_page_faults, rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::num_major_page_faults, rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::voluntary_context_switch,
                               rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::priority_context_switch,
                               rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::user_mode_time, rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::kernel_mode_time, rusage_cache_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(cache, component::current_peak_rss, rusage_cache_type)

//--------------------------------------------------------------------------------------//
//
//                              UNITS SPECIALIZATIONS
//
//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
//--------------------------------------------------------------------------------------//

template <>
struct units<component::current_peak_rss>
{
    using type         = std::pair<double, double>;
    using display_type = std::pair<std::string, std::string>;
};

//--------------------------------------------------------------------------------------//
}  // namespace trait
}  // namespace tim

//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(peak_rss, TIMEMORY_PEAK_RSS, "peak_rss", "memory_hwm",
                                 "memory_high_water_mark")
//
TIMEMORY_PROPERTY_SPECIALIZATION(page_rss, TIMEMORY_PAGE_RSS, "page_rss", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(num_io_in, TIMEMORY_NUM_IO_IN, "num_io_in", "io_in")
//
TIMEMORY_PROPERTY_SPECIALIZATION(num_io_out, TIMEMORY_NUM_IO_OUT, "num_io_out", "io_out")
//
TIMEMORY_PROPERTY_SPECIALIZATION(num_minor_page_faults, TIMEMORY_NUM_MINOR_PAGE_FAULTS,
                                 "num_minor_page_faults", "minor_page_faults")
//
TIMEMORY_PROPERTY_SPECIALIZATION(num_major_page_faults, TIMEMORY_NUM_MAJOR_PAGE_FAULTS,
                                 "num_major_page_faults", "major_page_faults")
//
TIMEMORY_PROPERTY_SPECIALIZATION(voluntary_context_switch,
                                 TIMEMORY_VOLUNTARY_CONTEXT_SWITCH,
                                 "voluntary_context_switch", "voluntary_context_switches",
                                 "vol_ctx_switch", "vol_ctx_switches")
//
TIMEMORY_PROPERTY_SPECIALIZATION(priority_context_switch,
                                 TIMEMORY_PRIORITY_CONTEXT_SWITCH,
                                 "priority_context_switch", "priority_context_switches",
                                 "prio_ctx_switch", "prio_ctx_switches")
//
TIMEMORY_PROPERTY_SPECIALIZATION(virtual_memory, TIMEMORY_VIRTUAL_MEMORY,
                                 "virtual_memory", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_mode_time, TIMEMORY_USER_MODE_TIME,
                                 "user_mode_time", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(kernel_mode_time, TIMEMORY_KERNEL_MODE_TIME,
                                 "kernel_mode_time", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(current_peak_rss, TIMEMORY_CURRENT_PEAK_RSS,
                                 "current_peak_rss", "current_memory_hwm",
                                 "current_memory_high_water_mark")
//
//======================================================================================//
