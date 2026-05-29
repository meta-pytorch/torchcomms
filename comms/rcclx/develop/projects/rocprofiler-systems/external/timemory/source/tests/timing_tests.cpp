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
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace tim::component;
using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

#if defined(TIMEMORY_MACOS)
// GitHub CI for macOS is very noisy
static const double util_tolerance  = 40.0;
static const double timer_tolerance = 0.25;
#else
static const double util_tolerance  = 15.0;
static const double timer_tolerance = 0.025;
#endif

#define CHECK_AVAILABLE(type)                                                            \
    if(!tim::trait::is_available<type>::value)                                           \
    {                                                                                    \
        printf("skipping %s because %s is not available\n",                              \
               details::get_test_name().c_str(), tim::demangle<type>().c_str());         \
        return;                                                                          \
    }

//--------------------------------------------------------------------------------------//
namespace details
{
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
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

class timing_tests : public ::testing::Test
{
protected:
    static void configure()
    {
        tim::settings::timing_units() = "sec";
        tim::settings::precision()    = 9;
        tim::threading::affinity::set();
    }

    TIMEMORY_TEST_SUITE_SETUP(configure())
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
std::string
datastr(Tp* obj, const std::string& prefix = "")
{
    using namespace tim::stl;
    using namespace tim::stl::ostream;
    std::stringstream ss;
    ss << std::boolalpha;
    if(prefix.empty())
        ss << tim::demangle<Tp>();
    else
        ss << prefix;
    ss << ":\n";
    ss << "    value     :: " << obj->get_value() << '\n';
    ss << "    accum     :: " << obj->get_accum() << '\n';
    ss << "    units     :: " << obj->get_unit() << '\n';
    ss << "    disp unit :: " << obj->get_display_unit() << '\n';
    ss << "    laps      :: " << obj->get_laps() << '\n';
    ss << "    transient :: " << obj->get_is_transient() << '\n';
    ss << "    running   :: " << obj->get_is_running() << '\n';
    ss << "    depth chg :: " << obj->get_depth_change() << '\n';
    ss << "    on stack  :: " << obj->get_is_on_stack() << '\n';
    ss << "    is flat   :: " << obj->get_is_flat() << '\n';
    return ss.str();
}

template <typename Tp>
std::string
datastr(Tp& obj, const std::string& prefix = "")
{
    return datastr(&obj, prefix);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, wall_timer)
{
    CHECK_AVAILABLE(wall_clock);

    wall_clock obj{};
    obj.start();
    details::do_sleep(1000);
    obj.stop();

    tim::lightweight_tuple<wall_clock> ct(details::get_test_name());
    ct.start();
    details::do_sleep(1000);
    ct.stop();
    auto wc = ct.get<wall_clock>();

    std::stringstream ss{};
    ss << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
       << ct << "\n"
       << datastr(obj, "manual wall_clock") << "\n"
       << datastr(wc, "component wall_clock");

    EXPECT_NEAR(1.0, obj.get(), timer_tolerance) << ss.str();
    EXPECT_NEAR(obj.get(), ct.get<wall_clock>()->get(), timer_tolerance) << ss.str();
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, monotonic_timer)
{
    CHECK_AVAILABLE(monotonic_clock);
    monotonic_clock obj{};
    obj.start();
    details::do_sleep(1000);
    obj.stop();
    EXPECT_NEAR(1.0, obj.get(), timer_tolerance)
        << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
        << datastr(obj);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, monotonic_raw_timer)
{
    CHECK_AVAILABLE(monotonic_raw_clock);
    monotonic_raw_clock obj{};
    obj.start();
    details::do_sleep(1000);
    obj.stop();
    EXPECT_NEAR(1.0, obj.get(), timer_tolerance)
        << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
        << datastr(obj);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, system_timer)
{
    CHECK_AVAILABLE(system_clock);
    system_clock obj{};
    obj.start();
    std::thread t(details::do_sleep, 1000);
    t.join();
    obj.stop();
    EXPECT_NEAR(0.0, obj.get(), timer_tolerance)
        << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
        << datastr(obj);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, user_timer)
{
    CHECK_AVAILABLE(user_clock);
    user_clock obj{};
    obj.start();
    std::thread t(details::do_sleep, 1000);
    t.join();
    obj.stop();
    EXPECT_NEAR(0.0, obj.get(), timer_tolerance)
        << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
        << datastr(obj);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, cpu_timer)
{
    CHECK_AVAILABLE(cpu_clock);
    cpu_clock obj{};
    obj.start();
    for(int i = 0; i < 10; ++i)
        details::consume(100);
    obj.stop();
    EXPECT_NEAR(1.0, obj.get(), timer_tolerance)
        << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
        << datastr(obj);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, cpu_utilization)
{
    CHECK_AVAILABLE(cpu_util);

    cpu_util obj{};
    obj.start();
    for(int i = 0; i < 10; ++i)
    {
        details::consume(75);
        details::do_sleep(25);
    }
    obj.stop();
    EXPECT_NEAR(75.0, obj.get(), util_tolerance)
        << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
        << datastr(obj);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, thread_cpu_timer)
{
    CHECK_AVAILABLE(thread_cpu_clock);
    auto _func = []() {
        tim::threading::affinity::set();
        thread_cpu_clock obj{};
        obj.start();
        std::thread t(details::fibonacci, 43);
        t.join();
        obj.stop();
        EXPECT_NEAR(0.0, obj.get(), timer_tolerance)
            << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
            << datastr(obj);
    };
    std::thread{ _func }.join();
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, thread_cpu_utilization)
{
    CHECK_AVAILABLE(thread_cpu_util);

    auto _func = []() {
        auto _consume = []() {
            tim::threading::affinity::set();
            for(int i = 0; i < 10; ++i)
                details::consume(200);
        };
        tim::threading::affinity::set();
        thread_cpu_util obj{};
        obj.start();
        std::thread t{ _consume };
        for(int i = 0; i < 10; ++i)
        {
            // over-estimate the cpu utilization bc
            // this tends to fail during CI
            details::consume(250);
            details::do_sleep(150);
        }
        t.join();
        obj.stop();

        std::cout << obj << "\n";
        EXPECT_NEAR(50.0, obj.get(), util_tolerance)
            << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
            << datastr(obj);
    };
    auto _t = std::thread{ _func };
    std::this_thread::yield();
    _t.join();
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, process_cpu_timer)
{
    CHECK_AVAILABLE(process_cpu_clock);
    process_cpu_clock obj{};
    obj.start();
    for(int i = 0; i < 10; ++i)
        details::consume(100);
    obj.stop();
    // this test seems to fail sporadically
    EXPECT_NEAR(1.0, obj.get(), 2.5 * timer_tolerance)
        << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
        << datastr(obj);
}

//--------------------------------------------------------------------------------------//

TEST_F(timing_tests, process_cpu_utilization)
{
    CHECK_AVAILABLE(process_cpu_util);
    auto _consume = []() {
        tim::threading::affinity::set();
        for(int i = 0; i < 10; ++i)
            details::consume(200);
    };

    process_cpu_util obj{};
    obj.start();
    std::thread t{ _consume };
    for(int i = 0; i < 10; ++i)
    {
        details::consume(100);
        details::do_sleep(100);
    }
    t.join();
    obj.stop();
    EXPECT_NEAR(150.0, obj.get(), util_tolerance)
        << "\n[" << details::get_test_name() << "]> result: " << obj << "\n"
        << datastr(obj);
}

//--------------------------------------------------------------------------------------//
