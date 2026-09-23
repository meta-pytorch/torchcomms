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

#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/context.hpp"
#include "timemory/components/ompt/macros.hpp"
#include "timemory/components/ompt/tool.hpp"
#include "timemory/components/user_bundle.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/utility/join.hpp"
#include "timemory/utility/type_list.hpp"

#include <cstdlib>

namespace tim
{
namespace openmp
{
//----------------------------------------------------------------------------------//

struct ompt_functions
{
    ompt_set_callback_t             set_callback             = nullptr;
    ompt_get_callback_t             get_callback             = nullptr;
    ompt_get_proc_id_t              get_proc_id              = nullptr;
    ompt_get_num_places_t           get_num_places           = nullptr;
    ompt_get_num_devices_t          get_num_devices          = nullptr;
    ompt_get_unique_id_t            get_unique_id            = nullptr;
    ompt_get_place_num_t            get_place_num            = nullptr;
    ompt_get_place_proc_ids_t       get_place_proc_ids       = nullptr;
    ompt_get_target_info_t          get_target_info          = nullptr;
    ompt_get_thread_data_t          get_thread_data          = nullptr;
    ompt_get_parallel_info_t        get_parallel_info        = nullptr;
    ompt_get_partition_place_nums_t get_partition_place_nums = nullptr;
    ompt_get_task_info_t            get_task_info            = nullptr;
    ompt_get_task_memory_t          get_task_memory          = nullptr;
    ompt_enumerate_states_t         enumerate_states         = nullptr;
    ompt_enumerate_mutex_impls_t    enumerate_mutex_impls    = nullptr;
    ompt_finalize_tool_t            finalize_tool            = nullptr;
};

//----------------------------------------------------------------------------------//

struct ompt_device_functions
{
    ompt_device_t*               device                = nullptr;
    ompt_get_device_num_procs_t  get_device_num_procs  = nullptr;
    ompt_get_device_time_t       get_device_time       = nullptr;
    ompt_translate_time_t        translate_time        = nullptr;
    ompt_set_trace_ompt_t        set_trace_ompt        = nullptr;
    ompt_set_trace_native_t      set_trace_native      = nullptr;
    ompt_start_trace_t           start_trace           = nullptr;
    ompt_pause_trace_t           pause_trace           = nullptr;
    ompt_stop_trace_t            stop_trace            = nullptr;
    ompt_advance_buffer_cursor_t advance_buffer_cursor = nullptr;
    ompt_get_record_type_t       get_record_type       = nullptr;
    ompt_get_record_ompt_t       get_record_ompt       = nullptr;
    ompt_get_record_native_t     get_record_native     = nullptr;
    ompt_get_record_abstract_t   get_record_abstract   = nullptr;
};

//----------------------------------------------------------------------------------//

template <typename ApiT>
auto&
get_ompt_functions()
{
    static auto _v = ompt_functions{};
    return _v;
}

//----------------------------------------------------------------------------------//

template <typename ApiT>
auto&
get_ompt_device_functions()
{
    static auto _v = std::unordered_map<int, ompt_device_functions>{};
    return _v;
}

using codeptr_ra_resolver_t = std::function<std::string(context_info&)>;

template <typename ApiT>
codeptr_ra_resolver_t&
get_codeptr_ra_resolver()
{
    static codeptr_ra_resolver_t _v = [](context_info& _ctx_info) {
        return (!_ctx_info.arguments.empty())
                   ? ::timemory::join::join(
                         " ", _ctx_info.label,
                         ::timemory::join::join(", ", _ctx_info.arguments))
                   : std::string{ _ctx_info.label };
    };
    return _v;
}

//----------------------------------------------------------------------------------//

struct function_lookup_params
{
    int            device_num    = -1;
    const char*    type          = nullptr;
    ompt_device_t* device        = nullptr;
    const char*    documentation = nullptr;
};

using function_lookup_callback_t = std::function<void(
    ompt_function_lookup_t lookup, const std::optional<function_lookup_params>&)>;

template <typename ApiT>
function_lookup_callback_t&
get_function_lookup_callback()
{
    static function_lookup_callback_t _v =
        [](ompt_function_lookup_t, const std::optional<function_lookup_params>&) {};
    return _v;
}

//----------------------------------------------------------------------------------//

template <typename ApiT>
template <typename... Tp>
void
context_handler<ApiT>::cleanup(size_t _idx, type_list<Tp...>)
{
    auto _cleanup = [_idx](auto& _v) {
        for(auto& iitr : _v)
        {
            for(auto& itr : iitr)
            {
                if(itr.second)
                {
                    TIMEMORY_PRINTF(stderr, "[ompt] stopping %s on thread %li\n",
                                    itr.second->key().c_str(), _idx);
                    itr.second->stop();
                    delete itr.second;
                    itr.second = nullptr;
                }
            }
        }
    };

    TIMEMORY_FOLD_EXPRESSION(_cleanup(get_map_data<Tp>(_idx)));
}

//----------------------------------------------------------------------------------//

template <typename ApiT>
void
context_handler<ApiT>::cleanup()
{
    auto_lock_t _lk{ type_mutex<context_handler<ApiT>>() };
    for(size_t i = 0; i < max_supported_threads; ++i)
    {
        cleanup(max_supported_threads - i - 1, cleanup_type_list{});
    }
}

//----------------------------------------------------------------------------------//
// callback thread begin
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_thread_t thread_type, ompt_data_t* thread_data)
{
    if(!m_enabled)
        return;

    auto _ctx_info =
        context_info{ get_enum_label(thread_type), nullptr, argument_array_t{} };
    m_key = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_start(m_key, get_data<thread_tag>(), thread_data, _ctx_info, thread_type,
                  thread_data);
}

//----------------------------------------------------------------------------------//
// callback thread end
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* thread_data)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_thread_end", nullptr, argument_array_t{} };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_stop(m_key, get_data<thread_tag>(), thread_data, _ctx_info);
}

//----------------------------------------------------------------------------------//
// parallel begin
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                                  ompt_data_t* parallel_data,
                                  unsigned int requested_parallelism, int flags,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info =
        context_info{ "ompt_parallel", codeptr,
                      argument_array_t{ { "parallelism", requested_parallelism },
                                        { "flags", flags } } };
    m_key = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_start(m_key, get_data<parallel_tag>(), task_data, _ctx_info, task_data,
                  task_frame, parallel_data, requested_parallelism, flags, codeptr);
}

//----------------------------------------------------------------------------------//
// parallel end
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  int flags, const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_parallel", codeptr, argument_array_t{} };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_stop(m_key, get_data<parallel_tag>(), task_data, _ctx_info, parallel_data,
                 task_data, flags, codeptr);
}

//----------------------------------------------------------------------------------//
// callback master
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_scope_endpoint_t endpoint,
                                  ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_master", codeptr, argument_array_t{} };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_endpoint(m_key, get_data<master_tag>(), endpoint, task_data, _ctx_info,
                     endpoint, parallel_data, task_data, codeptr);
}

//----------------------------------------------------------------------------------//
// callback implicit task
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_scope_endpoint_t endpoint,
                                  ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  unsigned int team_size, unsigned int thread_num)
{
    if(!m_enabled)
        return;

    // implicit tasks do not have a valid end on non-main threads
    // static auto              _first_global   = std::atomic<size_t>{ 0 };
    // static thread_local auto _first_thread   = std::atomic<size_t>{ 0 };
    // auto                     _first_global_v = _first_global++;
    // auto                     _first_thread_v = _first_thread++;
    // if(_first_global_v > 0 && _first_thread_v == 0)
    //     return;

    auto _ctx_info = context_info{ "ompt_implicit_task", nullptr,
                                   argument_array_t{ { "team_size", team_size },
                                                     { "thread_num", thread_num } } };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_endpoint(m_key, get_data<task_tag>(), endpoint, task_data, _ctx_info,
                     endpoint, parallel_data, task_data, team_size, thread_num);
}

//----------------------------------------------------------------------------------//
// callback sync region
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
                                  ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;

    constexpr size_t N = ompt_sync_region_reduction + 1;
    auto _ctx_info = context_info{ get_enum_label(kind), codeptr, argument_array_t{} };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_endpoint(m_key, get_data<sync_region_tag, N>(threading::get_id(), kind),
                     endpoint, task_data, _ctx_info, kind, endpoint, parallel_data,
                     task_data, codeptr);
}

//----------------------------------------------------------------------------------//
// callback mutex acquire
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_mutex_t kind, unsigned int hint, unsigned int impl,
                                  ompt_wait_id_t wait_id, const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{
        ompt_mutex_type_labels[kind], codeptr,
        argument_array_t{ { "hint", hint }, { "impl", impl }, { "wait_id", wait_id } }
    };
    m_key = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, kind, hint, impl, wait_id, codeptr);
}

//----------------------------------------------------------------------------------//
// callback mutex acquired
// callback mutex released
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_mutex_t kind, ompt_wait_id_t wait_id,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;

    ompt_scope_endpoint_t endpoint;
    switch(m_mode)
    {
        case mode::begin_callback: endpoint = ompt_scope_begin; break;
        case mode::end_callback: endpoint = ompt_scope_end; break;
        default:
        {
            TIMEMORY_PRINTF(stderr,
                            "[ompt] ignoring mutex callback with unknown endpoint\n");
            return;
        }
    };

    auto _task_data  = ompt_data_t{};
    _task_data.value = wait_id;
    auto _ctx_info   = context_info{ ompt_mutex_type_labels[kind], codeptr,
                                   argument_array_t{ { "wait_id", wait_id } } };
    m_key            = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_endpoint(m_key, get_data<mutex_tag>(), endpoint, &_task_data, _ctx_info, kind,
                     wait_id, codeptr);
}

//----------------------------------------------------------------------------------//
// callback nest lock
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_scope_endpoint_t endpoint, ompt_wait_id_t wait_id,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _task_data  = ompt_data_t{};
    _task_data.value = wait_id;

    auto _ctx_info = context_info{ "ompt_nested_lock", codeptr,
                                   argument_array_t{ { "wait_id", wait_id } } };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_endpoint(m_key, get_data<mutex_tag>(), endpoint, &_task_data, _ctx_info,
                     endpoint, wait_id, codeptr);
}

//----------------------------------------------------------------------------------//
// callback task create
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* task_data, const ompt_frame_t* task_frame,
                                  ompt_data_t* new_task_data, int flags,
                                  int has_dependences, const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_task", codeptr, argument_array_t{} };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_construct(m_key, get_data<task_tag>(), new_task_data, _ctx_info, task_data,
                      task_frame, new_task_data, flags, has_dependences, codeptr);
}

//----------------------------------------------------------------------------------//
// callback task scheduler
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t*       prior_task_data,
                                  ompt_task_status_t prior_task_status,
                                  ompt_data_t*       next_task_data)
{
    if(!m_enabled)
        return;

    if(prior_task_data)
    {
        auto _ctx_info =
            context_info{ "ompt_task_schedule", nullptr, argument_array_t{} };
        m_key = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
        switch(prior_task_status)
        {
            case ompt_task_complete:
            case ompt_task_cancel:
            case ompt_task_detach:
                context_stop(m_key, get_data<task_tag>(), prior_task_data, _ctx_info,
                             prior_task_data, prior_task_status, next_task_data);
                break;
            case ompt_task_early_fulfill:
            case ompt_task_late_fulfill:
                context_relaxed_stop(m_key, get_data<task_tag>(), prior_task_data,
                                     _ctx_info, prior_task_data, prior_task_status,
                                     next_task_data);
            case ompt_task_yield:
            case ompt_task_switch:
            default: break;
        }
    }
    if(next_task_data)
    {
        auto _ctx_info =
            context_info{ "ompt_task_schedule", nullptr, argument_array_t{} };
        m_key = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
        context_start_constructed(m_key, get_data<task_tag>(), next_task_data, _ctx_info,
                                  prior_task_data, prior_task_status, next_task_data);
    }
}

//----------------------------------------------------------------------------------//
// callback dispatch
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  ompt_dispatch_t kind, ompt_data_t instance)
{
    if(!m_enabled)
        return;

    const void* codeptr_ra = nullptr;
    auto        _args      = argument_array_t{};

    if(kind == ompt_dispatch_iteration)
    {
        _args.emplace_back("iteration", instance.value);
    }
    else if(kind == ompt_dispatch_section)
    {
        codeptr_ra = instance.ptr;
    }
    else if(kind == ompt_dispatch_ws_loop_chunk || kind == ompt_dispatch_taskloop_chunk ||
            kind == ompt_dispatch_distribute_chunk)
    {
        auto* _chunk = static_cast<ompt_dispatch_chunk_t*>(instance.ptr);
        if(_chunk)
        {
            _args.emplace_back("iteration_start", _chunk->start);
            _args.emplace_back("iteration_count", _chunk->iterations);
        }
    }

    auto _ctx_info = context_info{ get_enum_label(kind), codeptr_ra, std::move(_args) };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, parallel_data, task_data, kind,
                               instance);
}

//----------------------------------------------------------------------------------//
// callback work
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_work_t work_type, ompt_scope_endpoint_t endpoint,
                                  ompt_data_t* parallel_data, ompt_data_t* task_data,
                                  uint64_t count, const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ get_enum_label(work_type), codeptr,
                                   argument_array_t{ { "count", count } } };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_endpoint(m_key, get_data<work_tag>(), endpoint, task_data, _ctx_info,
                     work_type, endpoint, parallel_data, task_data, count, codeptr);
}

//----------------------------------------------------------------------------------//
// callback flush
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* thread_data, const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_flush", codeptr, argument_array_t{} };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, thread_data, codeptr);
}

//----------------------------------------------------------------------------------//
// callback cancel
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_data_t* thread_data, int flags,
                                  const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_cancel", codeptr, argument_array_t{} };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, thread_data, flags, codeptr);
}

//----------------------------------------------------------------------------------//
// callback target
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_target_t kind, ompt_scope_endpoint_t endpoint,
                                  int device_num, ompt_data_t* task_data,
                                  ompt_id_t target_id, const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ get_enum_label(kind), codeptr,
                                   argument_array_t{ { "device", device_num },
                                                     { "target_id", target_id } },
                                   target_args{ target_id, 0 } };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_endpoint(m_key, get_data<target_tag>(), endpoint, task_data, _ctx_info, kind,
                     endpoint, device_num, task_data, target_id, codeptr);
}

//----------------------------------------------------------------------------------//
// callback target data op
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_id_t target_id, ompt_id_t host_op_id,
                                  ompt_target_data_op_t optype, void* src_addr,
                                  int src_device_num, void* dest_addr,
                                  int dest_device_num, size_t bytes, const void* codeptr)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{};
    if(optype == ompt_target_data_delete)
    {
        _ctx_info = context_info{ get_enum_label(optype), codeptr,
                                  argument_array_t{ { "target_id", target_id },
                                                    { "host_op_id", host_op_id },
                                                    { "device", src_device_num },
                                                    { "addr", src_addr } },
                                  target_args{ target_id, host_op_id } };
        m_key     = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    }
    else
    {
        _ctx_info = context_info{ get_enum_label(optype), codeptr,
                                  argument_array_t{ { "target_id", target_id },
                                                    { "host_op_id", host_op_id },
                                                    { "src", src_device_num },
                                                    { "src_addr", src_addr },
                                                    { "dst", dest_device_num },
                                                    { "dst_addr", dest_addr },
                                                    { "bytes", bytes } },
                                  target_args{ target_id, host_op_id } };
        m_key     = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    }

    context_store<bundle_type>(m_key, _ctx_info, target_id, host_op_id, optype, src_addr,
                               src_device_num, dest_addr, dest_device_num, bytes,
                               codeptr);
}

//----------------------------------------------------------------------------------//
// callback target submit
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_id_t target_id, ompt_id_t host_op_id,
                                  unsigned int requested_num_teams)
{
    if(!m_enabled)
        return;

    auto _ctx_info =
        context_info{ "ompt_target_submit", nullptr,
                      argument_array_t{ { "target_id", target_id },
                                        { "host_op_id", host_op_id },
                                        { "requested_num_teams", requested_num_teams } },
                      target_args{ target_id, host_op_id } };
    m_key = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, target_id, host_op_id,
                               requested_num_teams);
}

//----------------------------------------------------------------------------------//
// callback target mapping
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(ompt_id_t target_id, unsigned int nitems,
                                  void** host_addr, void** device_addr, size_t* bytes,
                                  unsigned int* mapping_flags)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_target_mapping", nullptr,
                                   argument_array_t{ { "target_id", target_id } },
                                   target_args{ target_id, 0 } };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, target_id, nitems, host_addr,
                               device_addr, bytes, mapping_flags);
}

//----------------------------------------------------------------------------------//
// callback target device initialize
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(int device_num, const char* type, ompt_device_t* device,
                                  ompt_function_lookup_t lookup,
                                  const char*            documentation)
{
    if(!m_enabled)
        return;
    std::string_view _type = (type) ? type : "??";
    std::string_view _docs = (documentation) ? documentation : "??";

#define TIMEMORY_OMPT_LOOKUP(TYPE, NAME)                                                 \
    {                                                                                    \
        if(settings::verbose() >= 2 || settings::debug())                                \
            TIMEMORY_PRINTF_INFO(stderr, "[ompt][device=%i][type=%s] finding %s...\n",   \
                                 device_num, type, #NAME);                               \
        get_ompt_device_functions<ApiT>().at(device_num).NAME =                          \
            reinterpret_cast<TYPE>(lookup("ompt_" #NAME));                               \
        if(get_ompt_device_functions<ApiT>().at(device_num).NAME == nullptr &&           \
           (settings::verbose() >= 1 || settings::debug()))                              \
        {                                                                                \
            TIMEMORY_PRINTF_WARNING(                                                     \
                stderr, "[ompt][device=%i][type=%s] '%s' function lookup failed\n",      \
                device_num, type, "ompt_" #NAME);                                        \
        }                                                                                \
    }

    get_ompt_device_functions<ApiT>().emplace(device_num, ompt_device_functions{});

    get_ompt_device_functions<ApiT>().at(device_num).device = device;
    TIMEMORY_OMPT_LOOKUP(ompt_get_device_num_procs_t, get_device_num_procs);
    TIMEMORY_OMPT_LOOKUP(ompt_get_device_time_t, get_device_time);
    TIMEMORY_OMPT_LOOKUP(ompt_translate_time_t, translate_time);
    TIMEMORY_OMPT_LOOKUP(ompt_set_trace_ompt_t, set_trace_ompt);
    TIMEMORY_OMPT_LOOKUP(ompt_set_trace_native_t, set_trace_native);
    TIMEMORY_OMPT_LOOKUP(ompt_start_trace_t, start_trace);
    TIMEMORY_OMPT_LOOKUP(ompt_pause_trace_t, pause_trace);
    TIMEMORY_OMPT_LOOKUP(ompt_stop_trace_t, stop_trace);
    TIMEMORY_OMPT_LOOKUP(ompt_advance_buffer_cursor_t, advance_buffer_cursor);
    TIMEMORY_OMPT_LOOKUP(ompt_get_record_type_t, get_record_type);
    TIMEMORY_OMPT_LOOKUP(ompt_get_record_ompt_t, get_record_ompt);
    TIMEMORY_OMPT_LOOKUP(ompt_get_record_native_t, get_record_native);
    TIMEMORY_OMPT_LOOKUP(ompt_get_record_abstract_t, get_record_abstract);

#undef TIMEMORY_OMPT_LOOKUP

    get_function_lookup_callback<ApiT>()(
        lookup, function_lookup_params{ device_num, type, device, documentation });

    auto _ctx_info = context_info{ "ompt_device", nullptr,
                                   argument_array_t{ { "device_num", device_num },
                                                     { "type", _type },
                                                     { "device", device },
                                                     { "docs", _docs } } };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, device_num, type, device, lookup,
                               documentation);
}

//----------------------------------------------------------------------------------//
// callback target device finalize
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(int device_num)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_device_finalize", nullptr, argument_array_t{} };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, device_num);
}

//----------------------------------------------------------------------------------//
// callback target device load
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(int device_num, const char* filename,
                                  int64_t offset_in_file, void* vma_in_file, size_t bytes,
                                  void* host_addr, void* device_addr, uint64_t module_id)
{
    if(!m_enabled)
        return;
    std::string_view _fname = (filename) ? filename : "??";

    auto _ctx_info = context_info{ "ompt_target_load", nullptr,
                                   argument_array_t{ { "device", device_num },
                                                     { "filename", _fname },
                                                     { "offset", offset_in_file },
                                                     { "vma", vma_in_file },
                                                     { "bytes", bytes },
                                                     { "host_addr", host_addr },
                                                     { "device_addr", device_addr },
                                                     { "module_id", module_id } } };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, device_num, filename, offset_in_file,
                               vma_in_file, bytes, host_addr, device_addr, module_id);
}

//----------------------------------------------------------------------------------//
// callback target device unload
//----------------------------------------------------------------------------------//
template <typename ApiT>
void
context_handler<ApiT>::operator()(int device_num, uint64_t module_id)
{
    if(!m_enabled)
        return;

    auto _ctx_info = context_info{ "ompt_target_unload", nullptr,
                                   argument_array_t{ { "device", device_num },
                                                     { "module_id", module_id } } };
    m_key          = get_codeptr_ra_resolver<ApiT>()(_ctx_info);
    context_store<bundle_type>(m_key, _ctx_info, device_num, module_id);
}
}  // namespace openmp
}  // namespace tim
