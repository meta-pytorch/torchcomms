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

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using namespace tim::component;

// make different types to access and change traits individually

template <size_t Idx, bool StartSleep, bool StopSleep>
struct test_clock : public base<test_clock<Idx, StartSleep, StopSleep>>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using this_type  = test_clock<Idx, StartSleep, StopSleep>;
    using base_type  = base<this_type, value_type>;
    using string_t   = std::string;

    // since this is a template class, need these statements
    using base_type::accum;
    using base_type::load;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    static int64_t    unit() { return wall_clock::unit(); }
    static string_t   label() { return string_t("test_clock_") + std::to_string(Idx); }
    static string_t   description() { return "wall time"; }
    static string_t   display_unit() { return wall_clock::display_unit(); }
    static value_type record() { return wall_clock::record(); }

    double get() const { return load() / static_cast<double>(base_type::get_unit()); }

    double get_display() const { return get(); }

    void start()
    {
        if(StartSleep)
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        value = record();
    }

    void stop()
    {
        if(StopSleep)
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto tmp = record();
        accum += (tmp - value);
        value = tmp;
    }
};

using priority_start_wc = test_clock<0, false, true>;
using priority_stop_wc  = test_clock<1, true, false>;

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
template <>
struct is_timing_category<priority_start_wc> : public std::true_type
{};
template <>
struct is_timing_category<priority_stop_wc> : public std::true_type
{};
template <>
struct uses_timing_units<priority_start_wc> : public std::true_type
{};
template <>
struct uses_timing_units<priority_stop_wc> : public std::true_type
{};
template <>
struct start_priority<priority_start_wc> : public std::integral_constant<int, -64>
{};
template <>
struct stop_priority<priority_stop_wc> : public std::integral_constant<int, -64>
{};
}  // namespace trait
}  // namespace tim

//--------------------------------------------------------------------------------------//

using tuple_t =
    tim::component_tuple<wall_clock, cpu_clock, priority_start_wc, priority_stop_wc>;

using plus_t  = typename tuple_t::operation_t<tim::operation::plus>;
using start_t = typename tuple_t::operation_t<tim::operation::start>;
using stop_t  = typename tuple_t::operation_t<tim::operation::stop>;

using apply_v = tim::mpl::apply<void>;

//--------------------------------------------------------------------------------------//
// this function consumes approximately "n" milliseconds of wall time
//
#define do_sleep(N) std::this_thread::sleep_for(std::chrono::milliseconds(N))

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
}  // namespace details

//--------------------------------------------------------------------------------------//

class priority_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    void SetUp() override
    {
        TIMEMORY_TEST_CHECK_SKIP;
        priority_start_wc::get_label() = "priority-start-clock";
        priority_stop_wc::get_label()  = "priority-stop-clock";
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(priority_tests, simple_check)
{
    // std::cout << "plus  : " << tim::demangle<plus_t>() << "\n";
    // std::cout << "start : " << tim::demangle<start_t>() << "\n";
    // std::cout << "stop  : " << tim::demangle<stop_t>() << "\n";

    tuple_t t(details::get_test_name(), true);

    // start/stop all to check laps
    t.start();
    t.stop();

    t.start();

    do_sleep(250);  // TOTAL TIME: 0.25 seconds

    // priority_start(t);

    do_sleep(250);  // TOTAL TIME: 0.50 seconds

    // standard_start(t);

    do_sleep(500);  // TOTAL TIME: 1.00 seconds

    // priority_stop(t);

    do_sleep(125);  // TOTAL TIME: 1.125 seconds

    // standard_stop(t);

    do_sleep(125);  // TOTAL TIME: 1.25 seconds

    t.stop();

    // t.start();
    // details::consume(500);
    // t.stop();

    auto& native_wc = *t.get<wall_clock>();
    auto& pstart_wc = *t.get<priority_start_wc>();
    auto& pstop_wc  = *t.get<priority_stop_wc>();

    // printf("\n");
    // std::cout << native_wc << std::endl;
    // std::cout << pstart_wc << std::endl;
    // std::cout << pstop_wc << std::endl;
    // printf("\n");
    // std::cout << t << std::endl;
    // printf("\n");

    // each start()/stop() on tuple_t resets internal lap count to zero
    ASSERT_EQ(native_wc.get_laps(), 1);
    ASSERT_EQ(pstart_wc.get_laps(), 1);
    ASSERT_EQ(pstop_wc.get_laps(), 1);
    // tuple_t will have total number of start()/stop()
    ASSERT_EQ(t.laps(), 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(priority_tests, start_stop)
{
    // std::cout << "plus  : " << tim::demangle<plus_t>() << "\n";
    // std::cout << "start : " << tim::demangle<start_t>() << "\n";
    // std::cout << "stop  : " << tim::demangle<stop_t>() << "\n";

    tuple_t t(details::get_test_name(), true);

    t.start();

    do_sleep(500);  // TOTAL TIME: 0.50 seconds

    t.stop();

    auto& native_wc = *t.get<wall_clock>();
    auto& pstart_wc = *t.get<priority_start_wc>();
    auto& pstop_wc  = *t.get<priority_stop_wc>();

    // printf("\n");
    // std::cout << native_wc << std::endl;
    // std::cout << pstart_wc << std::endl;
    // std::cout << pstop_wc << std::endl;
    // printf("\n");
    // std::cout << t << std::endl;
    // printf("\n");

    double native_exp = 1.0;
    double pstart_exp = 1.5;
    double pstop_exp  = 0.5;

#if defined(TIMEMORY_MACOS)
    // GitHub CI for macOS is very noisy
    EXPECT_NEAR(native_exp, native_wc.get(), 0.275);
    EXPECT_NEAR(pstart_exp, pstart_wc.get(), 0.275);
    EXPECT_NEAR(pstop_exp, pstop_wc.get(), 0.275);
#else
    EXPECT_NEAR(native_exp, native_wc.get(), 0.125);
    EXPECT_NEAR(pstart_exp, pstart_wc.get(), 0.125);
    EXPECT_NEAR(pstop_exp, pstop_wc.get(), 0.125);
#endif
}

//--------------------------------------------------------------------------------------//
