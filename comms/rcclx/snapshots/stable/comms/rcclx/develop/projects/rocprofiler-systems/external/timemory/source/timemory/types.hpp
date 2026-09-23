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

/** \file types.hpp
 * \headerfile types.hpp "timemory/types.hpp"
 * Provides collection types
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/types.hpp"
#include "timemory/config/types.hpp"
#include "timemory/containers/types.hpp"
#include "timemory/enum.h"
#include "timemory/environment/types.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/manager/types.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/plotting/types.hpp"
#include "timemory/runtime/types.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/variadic/types.hpp"

#include <tuple>

//--------------------------------------------------------------------------------------//
//
// clang-format off
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_COMPONENT_TYPES)
#   define TIMEMORY_COMPONENT_TYPES             \
    component::allinea_map,                     \
    component::caliper_marker,                  \
    component::caliper_config,                  \
    component::caliper_loop_marker,             \
    component::cpu_clock,                       \
    component::cpu_roofline_dp_flops,           \
    component::cpu_roofline_flops,              \
    component::cpu_roofline_sp_flops,           \
    component::cpu_util,                        \
    component::craypat_counters,                \
    component::craypat_flush_buffer,            \
    component::craypat_heap_stats,              \
    component::craypat_record,                  \
    component::craypat_region,                  \
    component::cuda_event,                      \
    component::cuda_profiler,                   \
    component::cupti_activity,                  \
    component::cupti_counters,                  \
    component::cupti_pcsampling,                \
    component::current_peak_rss,                \
    component::data_tracker_integer,            \
    component::data_tracker_unsigned,           \
    component::data_tracker_floating,           \
    component::gperftools_cpu_profiler,         \
    component::gperftools_heap_profiler,        \
    component::gpu_roofline_dp_flops,           \
    component::gpu_roofline_flops,              \
    component::gpu_roofline_hp_flops,           \
    component::gpu_roofline_sp_flops,           \
    component::hip_event,                       \
    component::kernel_mode_time,                \
    component::likwid_marker,                   \
    component::likwid_nvmarker,                 \
    component::malloc_gotcha,                   \
    component::memory_allocations,              \
    component::monotonic_clock,                 \
    component::monotonic_raw_clock,             \
    component::network_stats,                   \
    component::num_io_in,                       \
    component::num_io_out,                      \
    component::num_major_page_faults,           \
    component::num_minor_page_faults,           \
    component::nvtx_marker,                     \
    component::ompt_native_handle,              \
    component::page_rss,                        \
    component::papi_array_t,                    \
    component::papi_vector,                     \
    component::peak_rss,                        \
    component::perfetto_trace,                  \
    component::printer,                         \
    component::priority_context_switch,         \
    component::process_cpu_clock,               \
    component::process_cpu_util,                \
    component::read_bytes,                      \
    component::read_char,                       \
    component::roctx_marker,                    \
    component::system_clock,                    \
    component::tau_marker,                      \
    component::thread_cpu_clock,                \
    component::thread_cpu_util,                 \
    component::timestamp,                       \
    component::trip_count,                      \
    component::user_clock,                      \
    component::user_global_bundle,              \
    component::user_mode_time,                  \
    component::virtual_memory,                  \
    component::voluntary_context_switch,        \
    component::vtune_event,                     \
    component::vtune_frame,                     \
    component::vtune_profiler,                  \
    component::wall_clock,                      \
    component::written_bytes,                   \
    component::written_char
#endif
//
//--------------------------------------------------------------------------------------//
//
// clang-format on
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_MINIMAL_TUPLE_TYPES)
#    define TIMEMORY_MINIMAL_TUPLE_TYPES                                                 \
        component::wall_clock, component::cpu_clock, component::cpu_util,                \
            component::peak_rss, component::user_global_bundle
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_FULL_TUPLE_TYPES)
#    define TIMEMORY_FULL_TUPLE_TYPES                                                    \
        component::wall_clock, component::system_clock, component::user_clock,           \
            component::cpu_util, component::peak_rss, component::user_global_bundle
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_MINIMAL_LIST_TYPES)
#    define TIMEMORY_MINIMAL_LIST_TYPES                                                  \
        component::papi_vector*, component::cuda_event*, component::nvtx_marker*,        \
            component::cupti_activity*, component::cupti_counters*,                      \
            component::cpu_roofline_flops*, component::gpu_roofline_flops*
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_FULL_LIST_TYPES)
#    define TIMEMORY_FULL_LIST_TYPES                                                     \
        component::caliper*, component::tau_marker*, component::papi_vector*,            \
            component::cuda_event*, component::nvtx_marker*, component::cupti_activity*, \
            component::cupti_counters*, component::cpu_roofline_flops*,                  \
            component::gpu_roofline_flops*
#endif

namespace tim
{
//
#if defined(DEBUG) && !defined(TIMEMORY_IGNORE_NATIVE_SIZE_MISMATCH)
//
static_assert((TIMEMORY_NATIVE_COMPONENTS_END -
               TIMEMORY_NATIVE_COMPONENT_INTERNAL_SIZE) ==
                  std::tuple_size<std::tuple<TIMEMORY_COMPONENT_TYPES>>::value,
              "Error! Number of components defined in macro TIMEMORY_COMPONENT_TYPES "
              "does not equal the enumeration value of TIMEMORY_NATIVE_COMPONENTS_END "
              "minus TIMEMORY_NATIVE_COMPONENT_INTERNAL_SIZE. It is likely that one of "
              "these was updated and the other was not");
//
#endif
//
//--------------------------------------------------------------------------------------//
//
// using complete_types_t = type_list<TIMEMORY_COMPONENT_TYPES>;
//
//--------------------------------------------------------------------------------------//
//
// using complete_tuple_t           = std::tuple<TIMEMORY_COMPONENT_TYPES>;
// using complete_component_list_t  = component_list<TIMEMORY_COMPONENT_TYPES>;
// using complete_component_tuple_t = component_tuple<TIMEMORY_COMPONENT_TYPES>;
// using complete_auto_list_t       = auto_list<TIMEMORY_COMPONENT_TYPES>;
// using complete_auto_tuple_t      = auto_tuple<TIMEMORY_COMPONENT_TYPES>;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct complete_types;
//
template <size_t... Idx>
struct complete_types<std::index_sequence<Idx...>>
{
    using type =
        convert_t<mpl::non_placeholder_t<type_list<component::enumerator_t<Idx>...>>,
                  type_list<>>;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct available_types;
//
template <size_t... Idx>
struct available_types<std::index_sequence<Idx...>>
{
    using type =
        convert_t<mpl::available_t<
                      mpl::non_placeholder_t<type_list<component::enumerator_t<Idx>...>>>,
                  type_list<>>;
};
//
//--------------------------------------------------------------------------------------//
//
using complete_types_t =
    typename complete_types<std::make_index_sequence<TIMEMORY_COMPONENTS_END>>::type;
//
using available_types_t =
    typename available_types<std::make_index_sequence<TIMEMORY_COMPONENTS_END>>::type;
//
//--------------------------------------------------------------------------------------//
//
using complete_tuple_t           = convert_t<complete_types_t, std::tuple<>>;
using complete_component_list_t  = convert_t<complete_types_t, component_list<>>;
using complete_component_tuple_t = convert_t<complete_types_t, component_tuple<>>;
using complete_auto_list_t       = convert_t<complete_types_t, auto_list<>>;
using complete_auto_tuple_t      = convert_t<complete_types_t, auto_tuple<>>;
//
//--------------------------------------------------------------------------------------//
//
using available_tuple_t           = convert_t<available_types_t, std::tuple<>>;
using available_component_list_t  = convert_t<available_types_t, component_list<>>;
using available_component_tuple_t = convert_t<available_types_t, component_tuple<>>;
using available_auto_list_t       = convert_t<available_types_t, auto_list<>>;
using available_auto_tuple_t      = convert_t<available_types_t, auto_tuple<>>;
//
//--------------------------------------------------------------------------------------//
//
//  backwards-compatibility
//
using complete_list_t  = complete_component_list_t;
using available_list_t = available_component_list_t;
//
//--------------------------------------------------------------------------------------//
//
using global_bundle_t = component_tuple<component::user_global_bundle>;
using ompt_bundle_t   = component_tuple<component::user_ompt_bundle>;
using mpip_bundle_t   = component_tuple<component::user_mpip_bundle>;
using ncclp_bundle_t  = component_tuple<component::user_ncclp_bundle>;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_LIBRARY_TYPE)
#    define TIMEMORY_LIBRARY_TYPE tim::available_component_list_t
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_BUNDLE_TYPE)
#    define TIMEMORY_BUNDLE_TYPE tim::global_bundle_t
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_OMPT_BUNDLE_TYPE)
#    define TIMEMORY_OMPT_BUNDLE_TYPE tim::ompt_bundle_t
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_MPIP_BUNDLE_TYPE)
#    define TIMEMORY_MPIP_BUNDLE_TYPE tim::mpip_bundle_t
#endif
//
//--------------------------------------------------------------------------------------//
//
