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

#include "timemory/backends/threading.hpp"
#include "timemory/components/ompt/macros.hpp"
#include "timemory/components/ompt/types.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/demangle.hpp"
//
#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/components.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/types.hpp"
//
#include <deque>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace openmp
{
//
//--------------------------------------------------------------------------------------//
//
#define TIMEMORY_OMPT_ENUM_LABEL_PAIR(TYPE)                                              \
    {                                                                                    \
        TYPE, #TYPE                                                                      \
    }
//
//--------------------------------------------------------------------------------------//
//
template <typename EnumT, int64_t Value>
struct ompt_enum_label;

template <typename EnumT>
struct ompt_enum_info;

#define TIMEMORY_OMPT_ENUM_INFO(VALUE, MIN_VALUE, MAX_VALUE, IS_BITSET)                  \
    template <>                                                                          \
    struct ompt_enum_info<VALUE>                                                         \
    {                                                                                    \
        using type                        = VALUE;                                       \
        static constexpr auto min         = MIN_VALUE;                                   \
        static constexpr auto max         = MAX_VALUE;                                   \
        static constexpr auto is_bitset   = IS_BITSET;                                   \
        static constexpr auto unsupported = "unsupported_" #VALUE;                       \
    };                                                                                   \
                                                                                         \
    template <int64_t Value>                                                             \
    struct ompt_enum_label<VALUE, Value>                                                 \
    {                                                                                    \
        static constexpr auto    name  = "unsupported_" #VALUE;                          \
        static constexpr int64_t value = -1;                                             \
    };

#define TIMEMORY_OMPT_ENUM_LABEL(VALUE)                                                  \
    template <>                                                                          \
    struct ompt_enum_label<decltype(VALUE), VALUE>                                       \
    {                                                                                    \
        static constexpr auto name  = #VALUE;                                            \
        static constexpr auto value = VALUE;                                             \
    };

TIMEMORY_OMPT_ENUM_INFO(ompt_callbacks_t, 1, 37, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_record_t, 1, 3, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_record_native_t, 1, 2, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_set_result_t, 0, 5, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_thread_t, 1, 4, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_scope_endpoint_t, 1, 3, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_dispatch_t, 1, 5, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_sync_region_t, 1, 10, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_target_data_op_t, 1, 20, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_work_t, 1, 13, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_mutex_t, 1, 7, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_task_status_t, 1, 8, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_target_t, 1, 12, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_dependence_type_t, 1, 35, false)
TIMEMORY_OMPT_ENUM_INFO(ompt_severity_t, 1, 2, false)
// bitsets
TIMEMORY_OMPT_ENUM_INFO(ompt_native_mon_flag_t, 0x01, 0x08, true)
TIMEMORY_OMPT_ENUM_INFO(ompt_task_flag_t, 0x00000001, 0x80000000, true)
TIMEMORY_OMPT_ENUM_INFO(ompt_parallel_flag_t, 0x00000001, 0x80000000, true)
TIMEMORY_OMPT_ENUM_INFO(ompt_target_map_flag_t, 0x01, 0x20, true)
TIMEMORY_OMPT_ENUM_INFO(ompt_cancel_flag_t, 0x01, 0x40, true)
TIMEMORY_OMPT_ENUM_INFO(ompt_frame_flag_t, 0x00, 0x30, true)
TIMEMORY_OMPT_ENUM_INFO(ompt_state_t, 0x000, 0x102, true)

// Alphabetically sorted (for findability)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_cancel)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_control_tool)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_dependences)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_device_finalize)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_device_initialize)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_device_load)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_device_unload)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_dispatch)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_error)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_flush)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_implicit_task)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_lock_destroy)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_lock_init)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_masked)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_mutex_acquire)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_mutex_acquired)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_mutex_released)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_nest_lock)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_parallel_begin)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_parallel_end)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_reduction)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_sync_region)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_sync_region_wait)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_data_op)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_data_op_emi)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_emi)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_map)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_map_emi)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_submit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_target_submit_emi)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_task_create)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_task_dependence)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_task_schedule)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_thread_begin)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_thread_end)
TIMEMORY_OMPT_ENUM_LABEL(ompt_callback_work)
TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_activated)
TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_detected)
TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_discarded_task)
TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_loop)
TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_parallel)
TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_sections)
TIMEMORY_OMPT_ENUM_LABEL(ompt_cancel_taskgroup)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_in)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_inout)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_inout_all_memory)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_inoutset)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_mutexinoutset)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_out)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_out_all_memory)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_sink)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dependence_type_source)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dispatch_distribute_chunk)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dispatch_iteration)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dispatch_section)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dispatch_taskloop_chunk)
TIMEMORY_OMPT_ENUM_LABEL(ompt_dispatch_ws_loop_chunk)
TIMEMORY_OMPT_ENUM_LABEL(ompt_fatal)
TIMEMORY_OMPT_ENUM_LABEL(ompt_frame_application)
TIMEMORY_OMPT_ENUM_LABEL(ompt_frame_cfa)
TIMEMORY_OMPT_ENUM_LABEL(ompt_frame_framepointer)
TIMEMORY_OMPT_ENUM_LABEL(ompt_frame_runtime)
TIMEMORY_OMPT_ENUM_LABEL(ompt_frame_stackaddress)
TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_atomic)
TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_critical)
TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_lock)
TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_nest_lock)
TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_ordered)
TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_test_lock)
TIMEMORY_OMPT_ENUM_LABEL(ompt_mutex_test_nest_lock)
TIMEMORY_OMPT_ENUM_LABEL(ompt_native_data_motion_explicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_native_data_motion_implicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_native_driver)
TIMEMORY_OMPT_ENUM_LABEL(ompt_native_idleness)
TIMEMORY_OMPT_ENUM_LABEL(ompt_native_kernel_execution)
TIMEMORY_OMPT_ENUM_LABEL(ompt_native_kernel_invocation)
TIMEMORY_OMPT_ENUM_LABEL(ompt_native_overhead)
TIMEMORY_OMPT_ENUM_LABEL(ompt_native_runtime)
TIMEMORY_OMPT_ENUM_LABEL(ompt_parallel_invoker_program)
TIMEMORY_OMPT_ENUM_LABEL(ompt_parallel_invoker_runtime)
TIMEMORY_OMPT_ENUM_LABEL(ompt_parallel_league)
TIMEMORY_OMPT_ENUM_LABEL(ompt_parallel_team)
TIMEMORY_OMPT_ENUM_LABEL(ompt_record_invalid)
TIMEMORY_OMPT_ENUM_LABEL(ompt_record_native)
TIMEMORY_OMPT_ENUM_LABEL(ompt_record_native_event)
TIMEMORY_OMPT_ENUM_LABEL(ompt_record_native_info)
TIMEMORY_OMPT_ENUM_LABEL(ompt_record_ompt)
TIMEMORY_OMPT_ENUM_LABEL(ompt_scope_begin)
TIMEMORY_OMPT_ENUM_LABEL(ompt_scope_beginend)
TIMEMORY_OMPT_ENUM_LABEL(ompt_scope_end)
TIMEMORY_OMPT_ENUM_LABEL(ompt_set_always)
TIMEMORY_OMPT_ENUM_LABEL(ompt_set_error)
TIMEMORY_OMPT_ENUM_LABEL(ompt_set_impossible)
TIMEMORY_OMPT_ENUM_LABEL(ompt_set_never)
TIMEMORY_OMPT_ENUM_LABEL(ompt_set_sometimes)
TIMEMORY_OMPT_ENUM_LABEL(ompt_set_sometimes_paired)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_idle)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_overhead)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_undefined)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_atomic)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_barrier)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_barrier_explicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_barrier_implementation)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_barrier_implicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_barrier_implicit_parallel)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_barrier_implicit_workshare)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_barrier_teams)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_critical)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_lock)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_mutex)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_ordered)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_target)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_target_map)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_target_update)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_taskgroup)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_wait_taskwait)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_work_parallel)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_work_reduction)
TIMEMORY_OMPT_ENUM_LABEL(ompt_state_work_serial)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_barrier)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_barrier_explicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_barrier_implementation)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_barrier_implicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_barrier_implicit_parallel)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_barrier_implicit_workshare)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_barrier_teams)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_reduction)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_taskgroup)
TIMEMORY_OMPT_ENUM_LABEL(ompt_sync_region_taskwait)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_alloc)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_alloc_async)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_associate)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_delete)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_delete_async)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_disassociate)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_transfer_from_device)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_transfer_from_device_async)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_transfer_to_device)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_data_transfer_to_device_async)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_enter_data)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_enter_data_nowait)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_exit_data)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_exit_data_nowait)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_alloc)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_delete)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_from)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_implicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_release)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_map_flag_to)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_nowait)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_update)
TIMEMORY_OMPT_ENUM_LABEL(ompt_target_update_nowait)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_cancel)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_complete)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_detach)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_early_fulfill)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_explicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_final)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_implicit)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_initial)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_late_fulfill)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_mergeable)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_merged)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_switch)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_target)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_taskwait)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_undeferred)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_untied)
TIMEMORY_OMPT_ENUM_LABEL(ompt_taskwait_complete)
TIMEMORY_OMPT_ENUM_LABEL(ompt_task_yield)
TIMEMORY_OMPT_ENUM_LABEL(ompt_thread_initial)
TIMEMORY_OMPT_ENUM_LABEL(ompt_thread_other)
TIMEMORY_OMPT_ENUM_LABEL(ompt_thread_unknown)
TIMEMORY_OMPT_ENUM_LABEL(ompt_thread_worker)
TIMEMORY_OMPT_ENUM_LABEL(ompt_warning)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_distribute)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_loop)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_loop_dynamic)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_loop_guided)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_loop_other)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_loop_static)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_scope)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_sections)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_single_executor)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_single_other)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_taskloop)
TIMEMORY_OMPT_ENUM_LABEL(ompt_work_workshare)

template <typename EnumT, size_t Idx, size_t... IdxTail>
auto
get_enum_label(EnumT val, std::index_sequence<Idx, IdxTail...>)
{
    using info  = ompt_enum_info<EnumT>;
    using label = ompt_enum_label<EnumT, Idx>;

    if(val == label::value)
        return label::name;

    if constexpr(sizeof...(IdxTail) > 0)
        return get_enum_label(val, std::index_sequence<IdxTail...>{});

    return info::unsupported;
}

template <typename EnumT>
auto
get_enum_label(EnumT val)
{
    using info          = ompt_enum_info<EnumT>;
    constexpr auto last = info::max + 1;

    return get_enum_label(val, std::make_index_sequence<last>{});
}
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_mutex_t, const char*> ompt_mutex_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_mutex_lock),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_mutex_test_lock),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_mutex_nest_lock),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_mutex_test_nest_lock),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_mutex_critical),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_mutex_atomic),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_mutex_ordered)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_task_flag_t, const char*> ompt_task_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_initial),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_implicit),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_explicit),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_target),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_taskwait),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_undeferred),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_untied),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_final),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_mergeable),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_task_merged)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_target_map_flag_t, const char*> ompt_target_map_labels = {
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_target_map_flag_to),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_target_map_flag_from),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_target_map_flag_alloc),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_target_map_flag_release),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_target_map_flag_delete),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_target_map_flag_implicit)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_dependence_type_t, const char*> ompt_dependence_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_dependence_type_in),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_dependence_type_out),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_dependence_type_inout),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_dependence_type_mutexinoutset),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_dependence_type_source),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_dependence_type_sink),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_dependence_type_inoutset),
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_cancel_flag_t, const char*> ompt_cancel_type_labels = {
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_cancel_parallel),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_cancel_sections),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_cancel_loop),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_cancel_taskgroup),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_cancel_activated),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_cancel_detected),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_cancel_discarded_task)
};
//
//--------------------------------------------------------------------------------------//
//
static std::map<ompt_callbacks_t, const char*> ompt_callback_labels = {
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_thread_begin),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_thread_end),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_parallel_begin),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_parallel_end),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_task_create),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_task_schedule),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_implicit_task),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_target),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_target_data_op),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_target_submit),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_control_tool),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_device_initialize),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_device_finalize),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_device_load),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_device_unload),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_sync_region_wait),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_mutex_released),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_dependences),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_task_dependence),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_work),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_master),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_target_map),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_sync_region),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_lock_init),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_lock_destroy),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_mutex_acquire),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_mutex_acquired),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_nest_lock),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_flush),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_cancel),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_reduction),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_dispatch),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_target_emi),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_target_data_op_emi),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_target_submit_emi),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_target_map_emi),
    TIMEMORY_OMPT_ENUM_LABEL_PAIR(ompt_callback_error)
};
//
//--------------------------------------------------------------------------------------//
//
template <typename ApiT>
struct context_handler
{
    using api_type    = ApiT;
    using this_type   = context_handler<api_type>;
    using bundle_type = typename trait::ompt_handle<api_type>::type;
    static constexpr size_t max_supported_threads =
        trait::max_threads<ApiT, context_handler<ApiT>>::value;

public:
    template <typename KeyT, typename MappedT, typename HashT = std::hash<KeyT>>
    using uomap_t = std::unordered_map<KeyT, MappedT, HashT>;

    template <typename Tag, size_t N = 1>
    static auto& get_map_data(int64_t _tid = threading::get_id())
    {
        using map_type = std::array<uomap_t<uint64_t, bundle_type*>, N>;
        static std::array<map_type, max_supported_threads> _v = {};
        return _v.at(_tid % max_supported_threads);
    }

    template <typename Tag, size_t N = 1>
    static auto& get_data(int64_t _tid = threading::get_id(), size_t _n = 0)
    {
        return get_map_data<Tag, N>(_tid).at(_n % N);
    }

    // tags for above
    struct task_tag
    {};
    struct mutex_tag
    {};
    struct sync_region_tag
    {};
    struct master_tag
    {};
    struct target_tag
    {};
    struct thread_tag
    {};
    struct parallel_tag
    {};
    struct work_tag
    {};
    struct task_create_tag
    {};

    using cleanup_type_list =
        type_list<task_tag, mutex_tag, sync_region_tag, target_tag, thread_tag,
                  parallel_tag, work_tag, master_tag, task_create_tag>;

    template <typename... Args>
    static auto join(Args&&... _args)
    {
        return mpl::apply<std::string>::join(std::forward<Args>(_args)...);
    }

public:
    template <typename... Tp>
    static void cleanup(size_t _idx, type_list<Tp...>);
    static void cleanup();

    TIMEMORY_DEFAULT_OBJECT(context_handler)

    explicit context_handler(mode _v)
    : m_mode{ _v }
    {}

    //----------------------------------------------------------------------------------//
    // callback thread begin
    //----------------------------------------------------------------------------------//
    void operator()(ompt_thread_t thread_type, ompt_data_t* thread_data);

    //----------------------------------------------------------------------------------//
    // callback thread end
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* thread_data);

    //----------------------------------------------------------------------------------//
    // parallel begin
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                    ompt_data_t* parallel_data, unsigned int requested_parallelism,
                    int flags, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // parallel end
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* parallel_data, ompt_data_t* task_data, int flags,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback master
    //----------------------------------------------------------------------------------//
    void operator()(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback implicit task
    //----------------------------------------------------------------------------------//
    void operator()(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data,
                    ompt_data_t* task_data, unsigned int team_size,
                    unsigned int thread_num);

    //----------------------------------------------------------------------------------//
    // callback sync region
    //----------------------------------------------------------------------------------//
    void operator()(ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback mutex acquire
    //----------------------------------------------------------------------------------//
    void operator()(ompt_mutex_t kind, unsigned int hint, unsigned int impl,
                    ompt_wait_id_t wait_id, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback mutex acquired
    // callback mutex released
    //----------------------------------------------------------------------------------//
    void operator()(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback nest lock
    //----------------------------------------------------------------------------------//
    void operator()(ompt_scope_endpoint_t endpoint, ompt_wait_id_t wait_id,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback task create
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                    ompt_data_t* new_task_data, int flags, int has_dependences,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback task scheduler
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* prior_task_data, ompt_task_status_t prior_task_status,
                    ompt_data_t* next_task_data);

    //----------------------------------------------------------------------------------//
    // callback dispatch
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* parallel_data, ompt_data_t* task_data,
                    ompt_dispatch_t kind, ompt_data_t instance);

    //----------------------------------------------------------------------------------//
    // callback work
    //----------------------------------------------------------------------------------//
    void operator()(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                    ompt_data_t* parallel_data, ompt_data_t* task_data, uint64_t count,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback flush
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* thread_data, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback cancel
    //----------------------------------------------------------------------------------//
    void operator()(ompt_data_t* thread_data, int flags, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback target
    //----------------------------------------------------------------------------------//
    void operator()(ompt_target_t kind, ompt_scope_endpoint_t endpoint, int device_num,
                    ompt_data_t* task_data, ompt_id_t target_id, const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback target data op
    //----------------------------------------------------------------------------------//
    void operator()(ompt_id_t target_id, ompt_id_t host_op_id,
                    ompt_target_data_op_t optype, void* src_addr, int src_device_num,
                    void* dest_addr, int dest_device_num, size_t bytes,
                    const void* codeptr);

    //----------------------------------------------------------------------------------//
    // callback target submit
    //----------------------------------------------------------------------------------//
    void operator()(ompt_id_t target_id, ompt_id_t host_op_id,
                    unsigned int requested_num_teams);

    //----------------------------------------------------------------------------------//
    // callback target mapping
    //----------------------------------------------------------------------------------//
    void operator()(ompt_id_t target_id, unsigned int nitems, void** host_addr,
                    void** device_addr, size_t* bytes, unsigned int* mapping_flags);

    //----------------------------------------------------------------------------------//
    // callback target device initialize
    //----------------------------------------------------------------------------------//
    void operator()(int device_num, const char* type, ompt_device_t* device,
                    ompt_function_lookup_t lookup, const char* documentation);

    //----------------------------------------------------------------------------------//
    // callback target device finalize
    //----------------------------------------------------------------------------------//
    void operator()(int device_num);

    //----------------------------------------------------------------------------------//
    // callback target device load
    //----------------------------------------------------------------------------------//
    void operator()(int device_num, const char* filename, int64_t offset_in_file,
                    void* vma_in_file, size_t bytes, void* host_addr, void* device_addr,
                    uint64_t module_id);

    //----------------------------------------------------------------------------------//
    // callback target device unload
    //----------------------------------------------------------------------------------//
    void operator()(int device_num, uint64_t module_id);

public:
    const std::string& key() const { return m_key; }
    void               set_mode(mode _v) { m_mode = _v; }

    friend std::ostream& operator<<(std::ostream& _os, const context_handler& _v)
    {
        _os << _v.m_key;
        return _os;
    }

protected:
    bool        m_enabled = trait::runtime_enabled<this_type>::get();
    mode        m_mode;
    std::string m_key = {};

    template <typename Ct, typename At>
    friend struct callback_connector;
};
}  // namespace openmp
//
//--------------------------------------------------------------------------------------//
//
namespace ompt
{
template <typename ApiT>
finalize_tool_func_t
configure(ompt_function_lookup_t lookup, int, ompt_data_t*);
}  // namespace ompt
}  // namespace tim
