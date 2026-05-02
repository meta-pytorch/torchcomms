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

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/components/user_bundle/overloads.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/insert.hpp"
#include "timemory/timemory.hpp"
#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/component_hybrid.hpp"

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

using namespace tim::component;
using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;

struct bundle_testing
{};

using auto_bundle_t = tim::auto_tuple<user_global_bundle, user_profiler_bundle>;
using comp_bundle_t = typename auto_bundle_t::component_type;
using bundle0_t     = tim::auto_tuple<wall_clock, cpu_util>;
using bundle1_t     = tim::auto_list<cpu_clock, peak_rss>;

using custom_bundle_t      = user_kokkosp_bundle;
using auto_custom_bundle_t = tim::auto_tuple<custom_bundle_t>;
using comp_custom_bundle_t = typename auto_custom_bundle_t::component_type;

//--------------------------------------------------------------------------------------//

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

class user_bundle_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    void SetUp() override
    {
        TIMEMORY_TEST_CHECK_SKIP;
        wc_size_orig = tim::storage<wall_clock>::instance()->size();
        cu_size_orig = tim::storage<cpu_util>::instance()->size();
        cc_size_orig = tim::storage<cpu_clock>::instance()->size();
        pr_size_orig = tim::storage<peak_rss>::instance()->size();

        printf("\n");
        printf("wc_size_orig = %lu\n", (long unsigned) wc_size_orig);
        printf("cu_size_orig = %lu\n", (long unsigned) cu_size_orig);
        printf("cc_size_orig = %lu\n", (long unsigned) cc_size_orig);
        printf("pr_size_orig = %lu\n", (long unsigned) pr_size_orig);
        printf("\n");

        ret = 0;

        custom_bundle_t::reset();
        user_global_bundle::reset();
        user_profiler_bundle::reset();

        auto bundle1_init = [](bundle1_t& _bundle) {
            TIMEMORY_PRINT_HERE("%s", "bundle1_init");
            if(details::get_test_name() != "bundle_0" &&
               details::get_test_name() != "bundle_3")
            {
                TIMEMORY_PRINT_HERE("%s<%s, %s>", "initialize", "cpu_clock", "peak_rss");

                // check that initialization works
                _bundle.initialize<cpu_clock, peak_rss>();
                ASSERT_TRUE(_bundle.get<cpu_clock>() != nullptr);
                ASSERT_TRUE(_bundle.get<peak_rss>() != nullptr);

                // check that re-initialization is ignored
                cpu_clock* _cptr = _bundle.get<cpu_clock>();
                peak_rss*  _pptr = _bundle.get<peak_rss>();
                _bundle.init<cpu_clock>();
                EXPECT_TRUE(_bundle.get<cpu_clock>() == _cptr);
                EXPECT_TRUE(_bundle.get<peak_rss>() == _pptr);

                // check that disable works
                _bundle.disable<cpu_clock>();
                fflush(stdout);
                fflush(stderr);
                std::cout << std::flush;
                std::cerr << std::flush;
                ASSERT_TRUE(_bundle.get<cpu_clock>() == nullptr);
                ASSERT_TRUE(_bundle.get<peak_rss>() != nullptr);

                // check that re-disabling is ignored
                _bundle.disable<cpu_clock>();
                EXPECT_TRUE(_bundle.get<cpu_clock>() == nullptr);
                EXPECT_TRUE(_bundle.get<peak_rss>() != nullptr);

                // check that initialization worked
                _bundle.init<cpu_clock>();
                ASSERT_TRUE(_bundle.get<cpu_clock>() != nullptr);
                ASSERT_TRUE(_bundle.get<peak_rss>() != nullptr);
            }
        };

        user_global_bundle::configure<bundle0_t, wall_clock, cpu_util>();
        user_profiler_bundle::configure<bundle1_t>(tim::scope::tree{}, false,
                                                   bundle1_init);

        EXPECT_EQ(user_global_bundle::bundle_size(), 1);
        EXPECT_EQ(user_profiler_bundle::bundle_size(), 1);
    }

protected:
    size_t wc_size_orig;
    size_t cu_size_orig;
    size_t cc_size_orig;
    size_t pr_size_orig;
    long   ret;
};

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_0)
{
    {
        TIMEMORY_BLANK_MARKER(auto_bundle_t, details::get_test_name().c_str());
        ret += details::fibonacci(35);
    }

    {
        TIMEMORY_BLANK_MARKER(auto_bundle_t, details::get_test_name().c_str());
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_1)
{
    {
        TIMEMORY_BLANK_MARKER(auto_bundle_t, details::get_test_name().c_str());
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, comp_bundle)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    {
        comp_bundle_t _instance{ details::get_test_name() };
        _instance.get<user_global_bundle>()->clear();

        _instance.start();
        ret += details::fibonacci(35);
        _instance.stop();

        _instance.start();
        ret += details::fibonacci(35);
        _instance.stop();
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 0;
    auto cu_n = cu_size_orig + 0;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, get)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    wall_clock* wc = nullptr;
    cpu_util*   cu = nullptr;
    cpu_clock*  cc = nullptr;
    peak_rss*   pr = nullptr;

    comp_bundle_t _instance{ details::get_test_name() };

    _instance.start();
    ret += details::fibonacci(35);
    details::consume(250);
    _instance.stop();

    wc = _instance.get<wall_clock>();
    cu = _instance.get<cpu_util>();
    cc = _instance.get<cpu_clock>();
    pr = _instance.get<peak_rss>();

    // check everything succeeded
    EXPECT_NE(wc, nullptr) << _instance;
    EXPECT_NE(cu, nullptr) << _instance;
    EXPECT_NE(cc, nullptr) << _instance;
    EXPECT_NE(pr, nullptr) << _instance;

    printf("fibonacci(35) = %li\n", ret);

    // use assert here to ensure we can dereference
    ASSERT_NE(wc, nullptr) << _instance;
    ASSERT_NE(cu, nullptr) << _instance;
    ASSERT_NE(cc, nullptr) << _instance;
    ASSERT_NE(pr, nullptr) << _instance;

    ASSERT_GT(wc->get(), 0.0);
    ASSERT_GT(cu->get(), 0.0);
    ASSERT_GT(cc->get(), 0.0);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, get_bundle)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    wall_clock* wc = nullptr;
    cpu_util*   cu = nullptr;
    cpu_clock*  cc = nullptr;
    peak_rss*   pr = nullptr;

    // test insert + get of a component_tuple within a user_bundle within a
    // component_tuple
    user_global_bundle::reset();
    tim::component_tuple<user_global_bundle> _instance{ details::get_test_name() };
    _instance.get<user_global_bundle>()
        ->insert<tim::component_tuple<wall_clock, cpu_util, cpu_clock, peak_rss>>();

    _instance.start();
    ret += details::fibonacci(35);
    details::consume(250);
    _instance.stop();

    wc = _instance.get<wall_clock>();
    cu = _instance.get<cpu_util>();
    cc = _instance.get<cpu_clock>();
    pr = _instance.get<peak_rss>();

    printf("fibonacci(35) = %li\n", ret);

    ASSERT_NE(wc, nullptr);
    ASSERT_NE(cu, nullptr);
    ASSERT_NE(cc, nullptr);
    ASSERT_NE(pr, nullptr);

    ASSERT_GT(wc->get(), 0.0);
    ASSERT_GT(cu->get(), 0.0);
    ASSERT_GT(cc->get(), 0.0);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_init_func)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    using auto_hybrid_t = tim::auto_hybrid<bundle0_t, bundle1_t>;

    auto init_func = [](auto& al) { al.template initialize<cpu_clock>(); };

    {
        auto_hybrid_t _bundle(details::get_test_name(), tim::scope::tree{}, false,
                              init_func);
        ret += details::fibonacci(35);
    }

    {
        auto_hybrid_t _bundle(details::get_test_name());
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_insert)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    auto init_func = [](auto& al) {
        std::vector<std::string> _init   = { "wall_clock", "cpu_clock" };
        auto&                    _bundle = *al.template get<custom_bundle_t>();
        tim::insert(_bundle, _init);
        tim::insert(_bundle, { CPU_UTIL, PEAK_RSS });
    };

    {
        auto_custom_bundle_t _one(details::get_test_name(), tim::scope::tree{}, false,
                                  init_func);
        ret += details::fibonacci(35);
    }

    {
        comp_custom_bundle_t _two(details::get_test_name(), true, tim::scope::tree{},
                                  init_func);
        _two.start();
        ret += details::fibonacci(35);
        _two.stop();
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);

    auto wc_data = tim::storage<wall_clock>::instance()->get();
    auto cc_data = tim::storage<cpu_clock>::instance()->get();

    ASSERT_GE(wc_data.back().prefix().length(), 4);
    EXPECT_FALSE(wc_data.back().prefix().substr(4).empty());

    ASSERT_GE(cc_data.back().prefix().length(), 4);
    EXPECT_FALSE(cc_data.back().prefix().substr(4).empty());
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_configure)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    std::initializer_list<std::string> _init = { "wall_clock", "cpu_clock" };
    tim::configure<custom_bundle_t>(_init);
    tim::configure<custom_bundle_t>({ CPU_UTIL, PEAK_RSS });

    {
        auto_custom_bundle_t _one(details::get_test_name() + "/one");
        ret += details::fibonacci(35);
        auto_custom_bundle_t _two(details::get_test_name() + "/two");
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 2;
    auto pr_n = pr_size_orig + 2;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_configure_ext)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    custom_bundle_t::reset();
    tim::configure<custom_bundle_t>({ CALIPER,
                                      CPU_CLOCK,
                                      CPU_UTIL,
                                      LIKWID_NVMARKER,
                                      LIKWID_MARKER,
                                      MONOTONIC_CLOCK,
                                      MONOTONIC_RAW_CLOCK,
                                      NUM_IO_IN,
                                      NUM_IO_OUT,
                                      NUM_MAJOR_PAGE_FAULTS,
                                      NUM_MINOR_PAGE_FAULTS,
                                      PAGE_RSS,
                                      PEAK_RSS,
                                      PRIORITY_CONTEXT_SWITCH,
                                      PROCESS_CPU_CLOCK,
                                      PROCESS_CPU_UTIL,
                                      READ_BYTES,
                                      SYS_CLOCK,
                                      TAU_MARKER,
                                      THREAD_CPU_CLOCK,
                                      THREAD_CPU_UTIL,
                                      TRIP_COUNT,
                                      USER_CLOCK,
                                      VIRTUAL_MEMORY,
                                      VOLUNTARY_CONTEXT_SWITCH,
                                      VTUNE_EVENT,
                                      VTUNE_FRAME,
                                      WALL_CLOCK,
                                      WRITTEN_BYTES });

    {
        auto_custom_bundle_t _one(details::get_test_name());
        ret += details::fibonacci(35);
        auto_custom_bundle_t _two(details::get_test_name());
        ret += details::fibonacci(35);
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 2;
    auto cu_n = cu_size_orig + 2;
    auto cc_n = cc_size_orig + 2;
    auto pr_n = pr_size_orig + 2;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);

    auto wc_data = tim::storage<wall_clock>::instance()->get();
    auto cc_data = tim::storage<cpu_clock>::instance()->get();

    ASSERT_GE(wc_data.back().prefix().length(), 4);
    EXPECT_FALSE(wc_data.back().prefix().substr(4).empty());

    ASSERT_GE(cc_data.back().prefix().length(), 4);
    EXPECT_FALSE(cc_data.back().prefix().substr(4).empty());
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, bundle_insert_ext)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    using comp_vec_t = std::set<TIMEMORY_COMPONENT>;

    custom_bundle_t::reset();
    comp_vec_t compenum = { CALIPER,
                            CPU_CLOCK,
                            CPU_UTIL,
                            LIKWID_NVMARKER,
                            LIKWID_MARKER,
                            MONOTONIC_CLOCK,
                            MONOTONIC_RAW_CLOCK,
                            NUM_IO_IN,
                            NUM_IO_OUT,
                            NUM_MAJOR_PAGE_FAULTS,
                            NUM_MINOR_PAGE_FAULTS,
                            PAGE_RSS,
                            PEAK_RSS,
                            PRIORITY_CONTEXT_SWITCH,
                            PROCESS_CPU_CLOCK,
                            PROCESS_CPU_UTIL,
                            READ_BYTES,
                            SYS_CLOCK,
                            TAU_MARKER,
                            THREAD_CPU_CLOCK,
                            THREAD_CPU_UTIL,
                            TRIP_COUNT,
                            USER_CLOCK,
                            VIRTUAL_MEMORY,
                            VOLUNTARY_CONTEXT_SWITCH,
                            VTUNE_EVENT,
                            VTUNE_FRAME,
                            WALL_CLOCK,
                            WRITTEN_BYTES };

    {
        comp_custom_bundle_t _one(details::get_test_name());
        comp_custom_bundle_t _two(details::get_test_name());
        tim::insert(_one.get<custom_bundle_t>(), compenum);

        _one.start();
        ret += details::fibonacci(35);

        _two.start();
        ret += details::fibonacci(35);

        _two.stop();
        _one.stop();
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 1;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);

    auto wc_data = tim::storage<wall_clock>::instance()->get();
    auto cc_data = tim::storage<cpu_clock>::instance()->get();

    ASSERT_GE(wc_data.back().prefix().length(), 4);
    EXPECT_FALSE(wc_data.back().prefix().substr(4).empty());

    ASSERT_GE(cc_data.back().prefix().length(), 4);
    EXPECT_FALSE(cc_data.back().prefix().substr(4).empty());
}

//--------------------------------------------------------------------------------------//

TEST_F(user_bundle_tests, laps)
{
    printf("TEST_NAME: %s\n", details::get_test_name().c_str());

    custom_bundle_t::reset();
    using lw_bundle_t = tim::lightweight_tuple<custom_bundle_t>;

    custom_bundle_t::configure<wall_clock, cpu_clock>();

    size_t      n = 10;
    lw_bundle_t obj{ details::get_test_name(), tim::scope::config(true, true) };

    TIMEMORY_DEBUG_PRINT_HERE("%s", "Real push begin");
    obj.push();
    TIMEMORY_DEBUG_PRINT_HERE("%s", "Real push end");
    for(size_t i = 0; i < n; ++i)
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s", "Real start begin");
        obj.start();
        TIMEMORY_DEBUG_PRINT_HERE("%s", "Real start end");
        ret += details::fibonacci(35);
        TIMEMORY_DEBUG_PRINT_HERE("%s", "Real stop begin");
        obj.stop();
        TIMEMORY_DEBUG_PRINT_HERE("%s", "Real stop end");
    }
    TIMEMORY_DEBUG_PRINT_HERE("%s", "Real pop start");
    obj.pop();
    TIMEMORY_DEBUG_PRINT_HERE("%s", "Real pop end");

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 0;
    auto cc_n = cc_size_orig + 1;
    auto pr_n = pr_size_orig + 0;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);

    auto wc_data = tim::storage<wall_clock>::instance()->get();
    auto cc_data = tim::storage<cpu_clock>::instance()->get();

    EXPECT_EQ(wc_data.back().data().get_laps(), 10) << "data: " << wc_data.back().data();
    EXPECT_EQ(cc_data.back().data().get_laps(), 10) << "data: " << cc_data.back().data();
}

//--------------------------------------------------------------------------------------//
