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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/**
 * \file timemory/components/craypat/backends.hpp
 * \brief Implementation of the craypat functions/utilities
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/utility/types.hpp"

#include <regex>
#include <string>

#if defined(TIMEMORY_USE_CRAYPAT) && !defined(CRAYPAT)
// if only TIMEMORY_USE_CRAYPAT is defined
#    define CRAYPAT
//
#elif defined(CRAYPAT) && !defined(TIMEMORY_USE_CRAYPAT)
// if only CRAYPAT is defined
#    define TIMEMORY_USE_CRAYPAT
//
#endif

#if defined(TIMEMORY_USE_CRAYPAT)
#    include <pat_api.h>
#endif

#if !defined(PAT_API_FAIL)
#    define PAT_API_FAIL 0 /* API call returned failure */
#endif
#if !defined(PAT_API_OK)
#    define PAT_API_OK 1 /* API call returned success */
#endif
#if !defined(PAT_STATE_OFF)
#    define PAT_STATE_OFF 0 /* deactivated */
#endif
#if !defined(PAT_STATE_ON)
#    define PAT_STATE_ON 1 /* activated */
#endif
#if !defined(PAT_STATE_QUERY)
#    define PAT_STATE_QUERY 2 /* prompt for state */
#endif
#if !defined(PAT_CTRS_CPU)
#    define PAT_CTRS_CPU 0 /* HWPCs on processor */
#endif
#if !defined(PAT_CTRS_NETWORK)
#    define PAT_CTRS_NETWORK 1 /* NWPCs on network router */
#endif
#if !defined(PAT_CTRS_ACCEL)
#    define PAT_CTRS_ACCEL 2 /* HWPCs on attached GPUs */
#endif
#if !defined(PAT_CTRS_RAPL)
#    define PAT_CTRS_RAPL 4 /* Running Avr Power Level on package */
#endif
#if !defined(PAT_CTRS_PM)
#    define PAT_CTRS_PM 5 /* Cray Power Management on node */
#endif
#if !defined(PAT_CTRS_UNCORE)
#    define PAT_CTRS_UNCORE 6 /* Intel Uncore on socket */
#endif

// don't define these symbols in header-only mode
#if defined(TIMEMORY_COMPONENT_SOURCE) || defined(TIMEMORY_USE_COMPONENT_EXTERN)
//
extern "C"
{
    //
    extern int PAT_record(int);
    extern int PAT_flush_buffer(unsigned long*);
    extern int PAT_region_begin(int, const char*);
    extern int PAT_region_end(int);
    extern int PAT_heap_stats(void);
    extern int PAT_counters(int, const char*[], unsigned long[], int*);
    // openmp
    extern void PAT_omp_barrier_enter(void);
    extern void PAT_omp_barrier_exit(void);
    extern void PAT_omp_loop_enter(void);
    extern void PAT_omp_loop_exit(void);
    extern void PAT_omp_master_enter(void);
    extern void PAT_omp_master_exit(void);
    extern void PAT_omp_parallel_begin(void);
    extern void PAT_omp_parallel_end(void);
    extern void PAT_omp_parallel_enter(void);
    extern void PAT_omp_parallel_exit(void);
    extern void PAT_omp_section_begin(void);
    extern void PAT_omp_section_end(void);
    extern void PAT_omp_sections_enter(void);
    extern void PAT_omp_sections_exit(void);
    extern void PAT_omp_single_enter(void);
    extern void PAT_omp_single_exit(void);
    extern void PAT_omp_task_begin(void);
    extern void PAT_omp_task_end(void);
    extern void PAT_omp_task_enter(void);
    extern void PAT_omp_task_exit(void);
    extern void PAT_omp_workshare_enter(void);
    extern void PAT_omp_workshare_exit(void);
    // open-acc
    extern void     PAT_acc_async_kernel_end(int, long int, int);
    extern long int PAT_acc_async_kernel_enter(int);
    extern void     PAT_acc_async_kernel_exit(int);
    extern void     PAT_acc_async_transfer_end(int, long int, int);
    extern long int PAT_acc_async_transfer_enter(int);
    extern void     PAT_acc_async_transfer_exit(int, long int, long int);
    extern void     PAT_acc_barrier_enter(int);
    extern void     PAT_acc_barrier_exit(int);
    extern void     PAT_acc_data_enter(int);
    extern void     PAT_acc_data_exit(int);
    extern void     PAT_acc_kernel_enter(int);
    extern void     PAT_acc_kernel_exit(int);
    extern void     PAT_acc_loop_enter(int);
    extern void     PAT_acc_loop_exit(int);
    extern void     PAT_acc_region_enter(int);
    extern void     PAT_acc_region_exit(int);
    extern void     PAT_acc_region_loop_enter(int);
    extern void     PAT_acc_region_loop_exit(int);
    extern void     PAT_acc_sync_enter(int);
    extern void     PAT_acc_sync_exit(int);
    extern void     PAT_acc_transfer_enter(int);
    extern void     PAT_acc_transfer_exit(int, long int, long int);
    extern void     PAT_acc_update_enter(int);
    extern void     PAT_acc_update_exit(int);
    //
}  // extern "C"
//
#endif

//======================================================================================//
//
namespace tim
{
namespace backend
{
namespace craypat
{
//
//--------------------------------------------------------------------------------------//
//
inline int
record(int val)
{
#if defined(CRAYPAT)
    return PAT_record(val);
#else
    consume_parameters(val);
    return PAT_API_FAIL;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int
flush_buffer(unsigned long* val)
{
#if defined(CRAYPAT)
    return PAT_flush_buffer(val);
#else
    consume_parameters(val);
    return PAT_API_FAIL;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int
region_begin(int id, const char* label)
{
#if defined(CRAYPAT)
    return PAT_region_begin(id, label);
#else
    consume_parameters(id, label);
    return PAT_API_FAIL;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int
region_end(int id)
{
#if defined(CRAYPAT)
    return PAT_region_end(id);
#else
    consume_parameters(id);
    return PAT_API_FAIL;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int
heap_stats()
{
#if defined(CRAYPAT)
    return PAT_heap_stats();
#else
    return PAT_API_FAIL;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int
counters(int category, const char**& names, unsigned long*& values, int*& nevents)
{
#if defined(CRAYPAT)
    return PAT_counters(category, names, values, nevents);
#else
    consume_parameters(category, names, values, nevents);
    return PAT_API_FAIL;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int
get_category(const std::string& key)
{
    using regex_array_t              = std::vector<std::pair<std::regex, int>>;
    static regex_array_t regex_array = []() {
        regex_array_t tmp;
        auto regex_constants = std::regex_constants::egrep | std::regex_constants::icase;
        auto add_regex       = [&](const std::string& regex_expr, int val) {
            tmp.push_back({ std::regex(regex_expr, regex_constants), val });
        };
        add_regex("accel", PAT_CTRS_ACCEL);
        add_regex("cpu", PAT_CTRS_CPU);
        add_regex("net", PAT_CTRS_NETWORK);
        add_regex("(pm|power)", PAT_CTRS_PM);
        add_regex("rapl", PAT_CTRS_RAPL);
        add_regex("core", PAT_CTRS_UNCORE);
        return tmp;
    }();

    for(const auto& itr : regex_array)
    {
        if(std::regex_search(key, itr.first))
            return itr.second;
    }
    return -1;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace craypat
}  // namespace backend
}  // namespace tim
//
//======================================================================================//
