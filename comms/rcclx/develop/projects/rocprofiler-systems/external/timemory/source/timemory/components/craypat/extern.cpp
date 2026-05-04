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

#include "timemory/components/craypat/extern.hpp"

#include "timemory/components/craypat/backends.hpp"

#if !defined(CRAYPAT)
//
extern "C"
{
    //
    static auto PAT_async_ret = std::numeric_limits<long int>::max();
    //
    int PAT_record(int) { return PAT_API_FAIL; }
    int PAT_flush_buffer(unsigned long*) { return PAT_API_FAIL; }
    int PAT_region_begin(int, const char*) { return PAT_API_FAIL; }
    int PAT_region_end(int) { return PAT_API_FAIL; }
    int PAT_heap_stats(void) { return PAT_API_FAIL; }
    int PAT_counters(int, const char*[], unsigned long[], int*) { return PAT_API_FAIL; }
    // openmp
    void PAT_omp_barrier_enter(void) {}
    void PAT_omp_barrier_exit(void) {}
    void PAT_omp_loop_enter(void) {}
    void PAT_omp_loop_exit(void) {}
    void PAT_omp_master_enter(void) {}
    void PAT_omp_master_exit(void) {}
    void PAT_omp_parallel_begin(void) {}
    void PAT_omp_parallel_end(void) {}
    void PAT_omp_parallel_enter(void) {}
    void PAT_omp_parallel_exit(void) {}
    void PAT_omp_section_begin(void) {}
    void PAT_omp_section_end(void) {}
    void PAT_omp_sections_enter(void) {}
    void PAT_omp_sections_exit(void) {}
    void PAT_omp_single_enter(void) {}
    void PAT_omp_single_exit(void) {}
    void PAT_omp_task_begin(void) {}
    void PAT_omp_task_end(void) {}
    void PAT_omp_task_enter(void) {}
    void PAT_omp_task_exit(void) {}
    void PAT_omp_workshare_enter(void) {}
    void PAT_omp_workshare_exit(void) {}
    // open-acc
    void     PAT_acc_async_kernel_end(int, long int, int) {}
    long int PAT_acc_async_kernel_enter(int) { return PAT_async_ret; }
    void     PAT_acc_async_kernel_exit(int) {}
    void     PAT_acc_async_transfer_end(int, long int, int) {}
    long int PAT_acc_async_transfer_enter(int) { return PAT_async_ret; }
    void     PAT_acc_async_transfer_exit(int, long int, long int) {}
    void     PAT_acc_barrier_enter(int) {}
    void     PAT_acc_barrier_exit(int) {}
    void     PAT_acc_data_enter(int) {}
    void     PAT_acc_data_exit(int) {}
    void     PAT_acc_kernel_enter(int) {}
    void     PAT_acc_kernel_exit(int) {}
    void     PAT_acc_loop_enter(int) {}
    void     PAT_acc_loop_exit(int) {}
    void     PAT_acc_region_enter(int) {}
    void     PAT_acc_region_exit(int) {}
    void     PAT_acc_region_loop_enter(int) {}
    void     PAT_acc_region_loop_exit(int) {}
    void     PAT_acc_sync_enter(int) {}
    void     PAT_acc_sync_exit(int) {}
    void     PAT_acc_transfer_enter(int) {}
    void     PAT_acc_transfer_exit(int, long int, long int) {}
    void     PAT_acc_update_enter(int) {}
    void     PAT_acc_update_exit(int) {}
    //
}  // extern "C"
//
#endif

TIMEMORY_EXTERN_COMPONENT(craypat_record, false, void)
TIMEMORY_EXTERN_COMPONENT(craypat_region, false, void)
TIMEMORY_EXTERN_COMPONENT(craypat_counters, true, std::vector<unsigned long>)
TIMEMORY_EXTERN_COMPONENT(craypat_heap_stats, false, void)
TIMEMORY_EXTERN_COMPONENT(craypat_flush_buffer, false, unsigned long)
