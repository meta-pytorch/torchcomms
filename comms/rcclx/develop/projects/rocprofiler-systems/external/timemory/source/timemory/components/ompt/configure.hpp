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
#include "timemory/components/ompt/callback_connector.hpp"
#include "timemory/components/ompt/context_handler.hpp"
#include "timemory/components/ompt/macros.hpp"
#include "timemory/components/ompt/ompt_wrapper.hpp"
#include "timemory/components/ompt/tool.hpp"
#include "timemory/log/macros.hpp"

#if !defined(TIMEMORY_OMPT_COMPONENT_HEADER_MODE) ||                                     \
    (defined(TIMEMORY_OMPT_COMPONENT_HEADER_MODE) &&                                     \
     TIMEMORY_OMPT_COMPONENT_HEADER_MODE == 0)
#    include "timemory/components/extern.hpp"
#endif

namespace tim
{
namespace ompt
{
template <typename ApiT>
finalize_tool_func_t
configure(ompt_function_lookup_t lookup, int _v, ompt_data_t* _data)
{
#if defined(TIMEMORY_USE_OMPT)
    //
    //----------------------------------------------------------------------------------//
    //
    using api_type       = ApiT;
    using handle_type    = component::ompt_handle<ApiT>;
    using toolset_type   = typename trait::ompt_handle<api_type>::type;
    using connector_type = openmp::callback_connector<toolset_type, api_type>;
    //
    //----------------------------------------------------------------------------------//
    //
#    define TIMEMORY_OMPT_LOOKUP(TYPE, NAME)                                             \
        {                                                                                \
            if(settings::verbose() >= 2 || settings::debug())                            \
                TIMEMORY_PRINTF_INFO(stderr, "[ompt] finding %s...\n", #NAME);           \
            openmp::get_ompt_functions<ApiT>().NAME =                                    \
                reinterpret_cast<TYPE>(lookup("ompt_" #NAME));                           \
            if(openmp::get_ompt_functions<ApiT>().NAME == nullptr &&                     \
               (settings::verbose() >= 0 || settings::debug()))                          \
            {                                                                            \
                TIMEMORY_PRINTF_WARNING(stderr, "[ompt] '%s' function lookup failed\n",  \
                                        "ompt_" #NAME);                                  \
            }                                                                            \
        }
    //
    //----------------------------------------------------------------------------------//
    //
    openmp::get_function_lookup_callback<ApiT>()(lookup, std::nullopt);
    //
    TIMEMORY_OMPT_LOOKUP(ompt_set_callback_t, set_callback);
    TIMEMORY_OMPT_LOOKUP(ompt_get_callback_t, get_callback);
    TIMEMORY_OMPT_LOOKUP(ompt_get_proc_id_t, get_proc_id);
    TIMEMORY_OMPT_LOOKUP(ompt_get_num_places_t, get_num_places);
    TIMEMORY_OMPT_LOOKUP(ompt_get_num_devices_t, get_num_devices);
    TIMEMORY_OMPT_LOOKUP(ompt_get_unique_id_t, get_unique_id);
    TIMEMORY_OMPT_LOOKUP(ompt_get_place_num_t, get_place_num);
    TIMEMORY_OMPT_LOOKUP(ompt_get_place_proc_ids_t, get_place_proc_ids);
    TIMEMORY_OMPT_LOOKUP(ompt_get_target_info_t, get_target_info);
    TIMEMORY_OMPT_LOOKUP(ompt_get_thread_data_t, get_thread_data);
    TIMEMORY_OMPT_LOOKUP(ompt_get_parallel_info_t, get_parallel_info);
    TIMEMORY_OMPT_LOOKUP(ompt_get_partition_place_nums_t, get_partition_place_nums);
    TIMEMORY_OMPT_LOOKUP(ompt_get_task_info_t, get_task_info);
    TIMEMORY_OMPT_LOOKUP(ompt_get_task_memory_t, get_task_memory);
    TIMEMORY_OMPT_LOOKUP(ompt_enumerate_states_t, enumerate_states);
    TIMEMORY_OMPT_LOOKUP(ompt_enumerate_mutex_impls_t, enumerate_mutex_impls);
    TIMEMORY_OMPT_LOOKUP(ompt_finalize_tool_t, finalize_tool);

#    undef TIMEMORY_OMPT_LOOKUP
    //
    //------------------------------------------------------------------------------//
    //
    if(!trait::is_available<handle_type>::value)
        return nullptr;

    handle_type::configure();
    auto manager = tim::manager::instance();
    if(manager)
    {
        manager->add_cleanup(demangle<handle_type>(),
                             []() { trait::runtime_enabled<toolset_type>::set(false); });
    }

    auto register_callback = [](ompt_callbacks_t cbidx, ompt_callback_t cb) {
        auto        _cb  = reinterpret_cast<ompt_callback_t>(cb);
        int         ret  = openmp::get_ompt_functions<ApiT>().set_callback(cbidx, _cb);
        const auto* name = openmp::get_enum_label(cbidx);
        const auto* retn = openmp::get_enum_label(static_cast<ompt_set_result_t>(ret));
        switch(ret)
        {
            case ompt_set_error:
            case ompt_set_never:
            case ompt_set_impossible:
            case ompt_set_sometimes:
            case ompt_set_sometimes_paired:
            {
                if(settings::verbose() >= 1 || settings::debug())
                {
                    TIMEMORY_PRINTF_WARNING(stderr,
                                            "OMPT Callback for event '%s' registered "
                                            "with return value: '%s'\n",
                                            name, retn);
                }
                break;
            }
            case ompt_set_always:
            {
                if(settings::verbose() >= 2 || settings::debug())
                {
                    TIMEMORY_PRINTF_INFO(stderr,
                                         "OMPT Callback for event '%s' registered with "
                                         "return value: '%s'\n",
                                         name, retn);
                }
                break;
            }
        }
        return ret;
    };
    //
    //----------------------------------------------------------------------------------//
    //
    auto timemory_ompt_register_callback = [&](ompt_callbacks_t name,
                                               ompt_callback_t  cb) {
        (void) register_callback(name, cb);
    };
    //
    //----------------------------------------------------------------------------------//
    //
    //      General thread
    //
    //----------------------------------------------------------------------------------//
    /*
    using thread_begin_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::begin_callback, ompt_thread_t,
                             ompt_data_t*>;

    using thread_end_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::end_callback, ompt_data_t*>;

    timemory_ompt_register_callback(ompt_callback_thread_begin,
                                    TIMEMORY_OMPT_CBCAST(thread_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_thread_end,
                                    TIMEMORY_OMPT_CBCAST(thread_end_cb_t::callback));
    */
    using parallel_begin_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::begin_callback, ompt_data_t*,
                             const ompt_frame_t*, ompt_data_t*, unsigned int, int,
                             const void*>;

    using parallel_end_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::end_callback, ompt_data_t*,
                             ompt_data_t*, int, const void*>;

    timemory_ompt_register_callback(ompt_callback_parallel_begin,
                                    TIMEMORY_OMPT_CBCAST(parallel_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_parallel_end,
                                    TIMEMORY_OMPT_CBCAST(parallel_end_cb_t::callback));

    using master_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::endpoint_callback,
                             ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*,
                             const void*>;

    timemory_ompt_register_callback(ompt_callback_master,
                                    TIMEMORY_OMPT_CBCAST(master_cb_t::callback));

    //----------------------------------------------------------------------------------//
    //
    //      Tasking section
    //
    //----------------------------------------------------------------------------------//

    using task_create_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::store_callback, ompt_data_t*,
                             const ompt_frame_t*, ompt_data_t*, int, int, const void*>;

    using task_schedule_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::store_callback, ompt_data_t*,
                             ompt_task_status_t, ompt_data_t*>;

    using work_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::endpoint_callback, ompt_work_t,
                             ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, uint64_t,
                             const void*>;

    using implicit_task_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::endpoint_callback,
                             ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*,
                             unsigned int, unsigned int>;

    using dispatch_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::end_callback, ompt_data_t*,
                             ompt_data_t*, ompt_dispatch_t, ompt_data_t>;

    timemory_ompt_register_callback(ompt_callback_task_create,
                                    TIMEMORY_OMPT_CBCAST(task_create_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_task_schedule,
                                    TIMEMORY_OMPT_CBCAST(task_schedule_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_work,
                                    TIMEMORY_OMPT_CBCAST(work_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_implicit_task,
                                    TIMEMORY_OMPT_CBCAST(implicit_task_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_dispatch,
                                    TIMEMORY_OMPT_CBCAST(dispatch_cb_t::callback));
    /*
    // using task_dependences_cb_t =
    //    openmp::ompt_wrapper<connector_type, openmp::mode::store_callback,
    //                         ompt_data_t*, const ompt_dependence_t*, int>;

    using task_dependence_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::store_callback, ompt_data_t*,
                             ompt_data_t*>;

    // timemory_ompt_register_callback(
    //    ompt_callback_dependences,
    //    TIMEMORY_OMPT_CBCAST(task_dependences_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_task_dependence,
                                    TIMEMORY_OMPT_CBCAST(task_dependence_cb_t::callback));
    */
    //----------------------------------------------------------------------------------//
    //
    //      Target section
    //
    //----------------------------------------------------------------------------------//

    using target_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::endpoint_callback,
                             ompt_target_t, ompt_scope_endpoint_t, int, ompt_data_t*,
                             ompt_id_t, const void*>;

    timemory_ompt_register_callback(ompt_callback_target,
                                    TIMEMORY_OMPT_CBCAST(target_cb_t::callback));

    using target_init_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::begin_callback, uint64_t,
                             const char*, ompt_device_t*, ompt_function_lookup_t,
                             const char*>;

    using target_finalize_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::end_callback, uint64_t>;

    timemory_ompt_register_callback(ompt_callback_device_initialize,
                                    TIMEMORY_OMPT_CBCAST(target_init_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_device_finalize,
                                    TIMEMORY_OMPT_CBCAST(target_finalize_cb_t::callback));

    using target_load_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::begin_callback, uint64_t,
                             const char*, int64_t, void*, size_t, void*, void*, uint64_t>;

    using target_unload_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::end_callback, uint64_t,
                             uint64_t>;

    timemory_ompt_register_callback(ompt_callback_device_load,
                                    TIMEMORY_OMPT_CBCAST(target_load_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_device_unload,
                                    TIMEMORY_OMPT_CBCAST(target_unload_cb_t::callback));

    using target_data_op_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::store_callback, ompt_id_t,
                             ompt_id_t, ompt_target_data_op_t, void*, int, void*, int,
                             size_t, const void*>;

    using target_submit_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::store_callback, ompt_id_t,
                             ompt_id_t, unsigned int>;

    using target_mapping_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::store_callback, ompt_id_t,
                             unsigned int, void**, void**, size_t*, unsigned int*>;

    timemory_ompt_register_callback(ompt_callback_target_data_op,
                                    TIMEMORY_OMPT_CBCAST(target_data_op_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target_submit,
                                    TIMEMORY_OMPT_CBCAST(target_submit_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_target_map,
                                    TIMEMORY_OMPT_CBCAST(target_mapping_cb_t::callback));

    //----------------------------------------------------------------------------------//
    //
    //      Sync/work section
    //
    //----------------------------------------------------------------------------------//

    using sync_region_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::endpoint_callback,
                             ompt_sync_region_t, ompt_scope_endpoint_t, ompt_data_t*,
                             ompt_data_t*, const void*>;

    timemory_ompt_register_callback(ompt_callback_sync_region,
                                    TIMEMORY_OMPT_CBCAST(sync_region_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_reduction,
                                    TIMEMORY_OMPT_CBCAST(sync_region_cb_t::callback));

    using mutex_nest_lock_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::endpoint_callback,
                             ompt_scope_endpoint_t, ompt_wait_id_t, const void*>;

    timemory_ompt_register_callback(ompt_callback_nest_lock,
                                    TIMEMORY_OMPT_CBCAST(mutex_nest_lock_cb_t::callback));

    using mutex_begin_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::begin_callback, ompt_mutex_t,
                             ompt_wait_id_t, const void*>;
    using mutex_end_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::end_callback, ompt_mutex_t,
                             ompt_wait_id_t, const void*>;

    timemory_ompt_register_callback(ompt_callback_mutex_acquired,
                                    TIMEMORY_OMPT_CBCAST(mutex_begin_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_mutex_released,
                                    TIMEMORY_OMPT_CBCAST(mutex_end_cb_t::callback));
    /*
    using mutex_acquire_cb_t =
        openmp::ompt_wrapper<connector_type, openmp::mode::begin_callback, ompt_mutex_t,
                             unsigned int, unsigned int, ompt_wait_id_t, const void*>;

    timemory_ompt_register_callback(ompt_callback_lock_init,
                                    TIMEMORY_OMPT_CBCAST(mutex_acquire_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_lock_destroy,
                                    TIMEMORY_OMPT_CBCAST(mutex_end_cb_t::callback));
    */
    // timemory_ompt_register_callback(ompt_callback_mutex_acquire,
    //                                TIMEMORY_OMPT_CBCAST(mutex_acquire_cb_t::callback));

    //----------------------------------------------------------------------------------//
    //
    //      Miscellaneous section
    //
    //----------------------------------------------------------------------------------//
    /*
    using flush_cb_t = openmp::ompt_wrapper<connector_type, openmp::mode::store_callback,
                                            ompt_data_t*, const void*>;

    using cancel_cb_t = openmp::ompt_wrapper<connector_type, openmp::mode::store_callback,
                                             ompt_data_t*, int, const void*>;

    timemory_ompt_register_callback(ompt_callback_flush,
                                    TIMEMORY_OMPT_CBCAST(flush_cb_t::callback));
    timemory_ompt_register_callback(ompt_callback_cancel,
                                    TIMEMORY_OMPT_CBCAST(cancel_cb_t::callback));
    */
    if(settings::verbose() > 1 || settings::debug())
        TIMEMORY_PRINTF(stderr, "\n");

    return openmp::get_ompt_functions<ApiT>().finalize_tool;
#endif
    (void) lookup;
    (void) _v;
    (void) _data;

    return nullptr;
}
}  // namespace ompt
}  // namespace tim
