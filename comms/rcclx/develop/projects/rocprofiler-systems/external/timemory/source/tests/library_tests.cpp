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

#include "timemory/compat/timemory_c.h"
#include "timemory/library.h"

#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;

//
//  functions defined in extern/custom-record-functions
//
extern void
custom_create_record(const char* name, uint64_t* id, int n, int* ct);
extern void
custom_delete_record(uint64_t id);
extern uint64_t
get_wc_storage_size();  // gets wall_clock storage size
extern uint64_t
get_cu_storage_size();  // gets cpu_util storage size
extern uint64_t
get_cc_storage_size();  // gets cpu_clock storage size
extern uint64_t
get_pr_storage_size();  // gets peak_rss storage size
extern uint64_t
get_uc_storage_size();  // gets user_clock storage size
extern uint64_t
get_sc_storage_size();  // gets system_clock storage size

//--------------------------------------------------------------------------------------//

static int    test_argc = 0;
static char** test_argv = nullptr;

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
inline void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class library_tests : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        timemory_set_environ("TIMEMORY_DEBUG", "OFF", 1, 0);
        timemory_set_environ("TIMEMORY_BANNER", "OFF", 1, 0);
        timemory_set_environ("TIMEMORY_VERBOSE", "1", 1, 0);
        timemory_set_environ("TIMEMORY_DART_COUNT", "1", 1, 0);
        timemory_set_environ("TIMEMORY_JSON_OUTPUT", "ON", 1, 0);
        timemory_set_environ("TIMEMORY_DART_OUTPUT", "OFF", 1, 0);

        if(!timemory_library_is_initialized())
            timemory_init_library(test_argc, test_argv);
    }

    static void TearDownTestSuite()
    {
        if(timemory_library_is_initialized())
            timemory_finalize_library();
    }

    void SetUp() override
    {
        timemory_set_default("wall_clock, cpu_util, cpu_clock, peak_rss");

        wc_size_orig = get_wc_storage_size();
        cu_size_orig = get_cu_storage_size();
        cc_size_orig = get_cc_storage_size();
        pr_size_orig = get_pr_storage_size();
        uc_size_orig = get_uc_storage_size();
        sc_size_orig = get_sc_storage_size();

        printf("\n");
        printf("wc_size_orig = %lu\n", (long unsigned) wc_size_orig);
        printf("cu_size_orig = %lu\n", (long unsigned) cu_size_orig);
        printf("cc_size_orig = %lu\n", (long unsigned) cc_size_orig);
        printf("pr_size_orig = %lu\n", (long unsigned) pr_size_orig);
        printf("uc_size_orig = %lu\n", (long unsigned) uc_size_orig);
        printf("sc_size_orig = %lu\n", (long unsigned) sc_size_orig);
        printf("\n");

        ret = 0;
    }

protected:
    uint64_t wc_size_orig;
    uint64_t cu_size_orig;
    uint64_t cc_size_orig;
    uint64_t pr_size_orig;
    uint64_t uc_size_orig;
    uint64_t sc_size_orig;
    long     ret;
};

//--------------------------------------------------------------------------------------//

#define TEST_NAME TIMEMORY_JOIN(".", "library_tests", details::get_test_name()).c_str()

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, record)
{
    timemory_push_components("wall_clock, cpu_util");

    {
        uint64_t idx = 0;
        timemory_begin_record(TEST_NAME, &idx);
        ret += details::fibonacci(35);
        timemory_end_record(idx);
    }

    {
        uint64_t idx = 0;
        timemory_begin_record(TEST_NAME, &idx);
        ret += details::fibonacci(35);
        timemory_end_record(idx);
    }

    timemory_pop_components();

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 0;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, get_record)
{
    {
        auto idx = timemory_get_begin_record(TEST_NAME);
        ret += details::fibonacci(35);
        timemory_end_record(idx);
    }

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, scoped_record)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    {
        timemory_scoped_record tmp(TEST_NAME);
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, scoped_record_enum)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    {
        timemory_scoped_record tmp(TEST_NAME, CPU_CLOCK, PEAK_RSS);
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 0;
    auto cu_n = cu_size_orig + 0;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, scope_record_index)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    {
        timemory_scoped_record tmp(TEST_NAME, "wall_clock, cpu_clock");
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 0;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, function_pointers)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    timemory_create_function = &custom_create_record;
    timemory_delete_function = &custom_delete_record;

    auto idx = timemory_get_begin_record_types(TEST_NAME, "wall_clock");
    ret += details::fibonacci(35);
    timemory_end_record(idx);

    timemory_create_function = nullptr;
    timemory_delete_function = nullptr;

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, c_marker_macro)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    {
        void* a =
            TIMEMORY_C_BLANK_MARKER(TEST_NAME, WALL_CLOCK, CPU_CLOCK, CPU_UTIL, PEAK_RSS);
        ret += details::fibonacci(35);
        void* b = TIMEMORY_C_BLANK_MARKER(TEST_NAME, WALL_CLOCK, CPU_CLOCK, CPU_UTIL);
        ret += details::fibonacci(35);
        FREE_TIMEMORY_C_MARKER(b);
        ret += details::fibonacci(35);
        FREE_TIMEMORY_C_MARKER(a);
    }

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 2;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, c_auto_timer_macro)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    {
        void* a = TIMEMORY_C_BLANK_AUTO_TIMER(TEST_NAME);
        ret += details::fibonacci(35);
        void* b = TIMEMORY_C_BLANK_AUTO_TIMER(TEST_NAME);
        ret += details::fibonacci(35);
        FREE_TIMEMORY_C_AUTO_TIMER(a);
        ret += details::fibonacci(35);
        FREE_TIMEMORY_C_AUTO_TIMER(b);
    }

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 2;
    auto pr_n = pr_size_orig + 2;
    auto uc_n = uc_size_orig + 0;
    auto sc_n = sc_size_orig + 0;

    printf("\n");
    printf("wc_size_req = %lu\n", (long unsigned) wc_n);
    printf("cu_size_req = %lu\n", (long unsigned) cu_n);
    printf("cc_size_req = %lu\n", (long unsigned) cc_n);
    printf("pr_size_req = %lu\n", (long unsigned) pr_n);
    printf("uc_size_req = %lu\n", (long unsigned) uc_n);
    printf("sc_size_req = %lu\n", (long unsigned) sc_n);
    printf("\n");
    printf("wc_size_now = %lu\n", (long unsigned) get_wc_storage_size());
    printf("cu_size_now = %lu\n", (long unsigned) get_cu_storage_size());
    printf("cc_size_now = %lu\n", (long unsigned) get_cc_storage_size());
    printf("pr_size_now = %lu\n", (long unsigned) get_pr_storage_size());
    printf("uc_size_now = %lu\n", (long unsigned) get_uc_storage_size());
    printf("sc_size_now = %lu\n", (long unsigned) get_sc_storage_size());
    printf("\n");

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
    ASSERT_EQ(get_uc_storage_size(), uc_n);
    ASSERT_EQ(get_sc_storage_size(), sc_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, region)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    timemory_push_region(TEST_NAME);
    ret += details::fibonacci(35);

    timemory_push_region(TEST_NAME);
    ret += details::fibonacci(35);

    timemory_pop_region(TEST_NAME);
    timemory_pop_region(TEST_NAME);

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 2;
    auto pr_n = pr_size_orig + 2;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, add)
{
    timemory_push_components("wall_clock, cpu_util");

    {
        uint64_t idx = 0;
        timemory_begin_record(TEST_NAME, &idx);
        ret += details::fibonacci(35);
        timemory_end_record(idx);
    }

    timemory_add_components("cpu_clock");

    {
        uint64_t idx = 0;
        timemory_begin_record(TEST_NAME, &idx);
        ret += details::fibonacci(35);
        timemory_end_record(idx);
    }

    timemory_pop_components();

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, remove)
{
    std::array<uint64_t, 2> idx;
    idx.fill(0);

    timemory_push_components("wall_clock, cpu_util, cpu_clock");

    timemory_begin_record(TEST_NAME, &idx[0]);
    ret += details::fibonacci(35);

    timemory_remove_components("cpu_clock");

    timemory_begin_record(TEST_NAME, &idx[1]);
    ret += details::fibonacci(35);

    timemory_end_record(idx[0]);
    timemory_end_record(idx[1]);

    timemory_pop_components();

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, add_remove)
{
    std::array<uint64_t, 2> idx;
    idx.fill(0);

    timemory_push_components("wall_clock, cpu_clock");

    // add and then immediately remove
    timemory_add_components("cpu_util");
    timemory_remove_components("cpu_util");

    timemory_begin_record(TEST_NAME, &idx[0]);
    ret += details::fibonacci(35);

    // add in second region
    timemory_add_components("peak_rss");

    timemory_begin_record(TEST_NAME, &idx[1]);
    ret += details::fibonacci(35);

    // remove
    timemory_remove_components("peak_rss");

    timemory_end_record(idx[0]);
    timemory_end_record(idx[1]);

    timemory_pop_components();

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 0;
    auto cc_n = cc_size_orig + 2;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, record_types)
{
    std::array<uint64_t, 2> idx;
    idx.fill(0);

    timemory_begin_record_types(TEST_NAME, &idx[0], "wall_clock, cpu_util, cpu_clock");
    ret += details::fibonacci(35);

    timemory_begin_record_types(TEST_NAME, &idx[0], "wall_clock, cpu_util");
    ret += details::fibonacci(35);

    timemory_end_record(idx[0]);
    timemory_end_record(idx[1]);

    timemory_pop_components();

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, record_enum)
{
    std::array<uint64_t, 2> idx;
    idx.fill(0);

    timemory_begin_record_enum(TEST_NAME, &idx[0], WALL_CLOCK, CPU_CLOCK, CPU_UTIL,
                               TIMEMORY_COMPONENTS_END);
    ret += details::fibonacci(35);

    timemory_begin_record_enum(TEST_NAME, &idx[0], WALL_CLOCK, CPU_UTIL,
                               TIMEMORY_COMPONENTS_END);
    ret += details::fibonacci(35);

    timemory_end_record(idx[0]);
    timemory_end_record(idx[1]);

    timemory_pop_components();

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, pause_resume)
{
    std::array<uint64_t, 2> idx;
    idx.fill(0);

    timemory_pause();
    timemory_begin_record_enum(TEST_NAME, &idx[0], WALL_CLOCK, CPU_CLOCK, CPU_UTIL,
                               TIMEMORY_COMPONENTS_END);
    ret += details::fibonacci(35);

    timemory_resume();
    timemory_begin_record_enum(TEST_NAME, &idx[0], WALL_CLOCK, CPU_UTIL,
                               TIMEMORY_COMPONENTS_END);
    ret += details::fibonacci(35);

    timemory_end_record(idx[0]);
    timemory_end_record(idx[1]);

    timemory_pop_components();

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 0;
    auto pr_n = pr_size_orig + 0;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(library_tests, override_default)
{
    timemory_set_default("wall_clock, cpu_util");

    {
        uint64_t idx = 0;
        timemory_begin_record(TEST_NAME, &idx);
        ret += details::fibonacci(35);
        timemory_end_record(idx);
    }

    {
        uint64_t idx = 0;
        timemory_begin_record(TEST_NAME, &idx);
        ret += details::fibonacci(35);
        timemory_end_record(idx);
    }

    timemory_pop_components();

    printf("fibonacci(35) = %li\n\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    ASSERT_EQ(get_wc_storage_size(), wc_n);
    ASSERT_EQ(get_cu_storage_size(), cu_n);
    ASSERT_EQ(get_cc_storage_size(), cc_n);
    ASSERT_EQ(get_pr_storage_size(), pr_n);

    timemory_set_environ("TIMEMORY_GLOBAL_COMPONENTS", "", 1, 1);
}

//--------------------------------------------------------------------------------------//

#include "test_macros.hpp"

TIMEMORY_TEST_MAIN

//--------------------------------------------------------------------------------------//
