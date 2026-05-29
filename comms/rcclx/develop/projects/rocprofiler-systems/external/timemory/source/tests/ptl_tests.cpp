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

#include "PTL/TaskGroup.hh"
#include "PTL/TaskRunManager.hh"
#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using tuple_t =
    tim::auto_tuple_t<wall_clock, cpu_util, thread_cpu_clock, thread_cpu_util>;

static const long   nfib = 39;
static const long   noff = 4;
static const double pi_epsilon =
    static_cast<double>(std::numeric_limits<float>::epsilon());

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

double
compute_pi(uint64_t nstart, uint64_t nstop, double step, uint64_t nblock)
{
    if(nstop - nstart < nblock)
    {
        TIMEMORY_BASIC_MARKER(tuple_t, "compute");
        double sum = 0.0;
        for(uint64_t i = nstart; i < nstop; ++i)
        {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        return sum;
    }
    else
    {
        TIMEMORY_BASIC_MARKER(tuple_t, "split");
        uint64_t             iblk = nstop - nstart;
        double               sum1 = 0.0;
        double               sum2 = 0.0;
        PTL::TaskGroup<void> tg;
        tg.exec([&]() { sum1 = compute_pi(nstart, nstop - iblk / 2, step, nblock); });
        tg.exec([&]() { sum2 = compute_pi(nstop - iblk / 2, nstop, step, nblock); });
        tg.wait();
        return sum1 + sum2;
    }
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class ptl_tests : public ::testing::Test
{
protected:
    static auto& get_manager()
    {
        static auto _instance = std::make_unique<PTL::TaskRunManager>();
        return _instance;
    }

    static void SetUpTestSuite()
    {
        PTL::TaskRunManager& manager = *get_manager();

        tim::settings::verbose()           = 0;
        tim::settings::debug()             = false;
        tim::settings::json_output()       = false;
        tim::settings::flamegraph_output() = false;
        tim::timemory_init(&test_argc, &test_argv);
        tim::settings::dart_output() = false;
        tim::settings::dart_count()  = 1;

        tim::signals::enable_signal_detection(
            { tim::signals::sys_signal::SegFault, tim::signals::sys_signal::Illegal });

        std::cout << "Initializing thread-pool... " << std::flush;
        manager.Initialize(tim::get_env<uint64_t>("NUM_THREADS", 4));
        std::cout << "Done" << std::endl;
        tim::threading::affinity::set();
    }

    static void TearDownTestSuite()
    {
        tim::signals::enable_signal_detection(
            tim::signals::signal_settings::get_default());
        auto& manager = get_manager();

        std::cout << "Tweaking settings..." << std::endl;
        auto tmp                 = tim::settings::push<TIMEMORY_API>();
        tim::settings::debug()   = true;
        tim::settings::verbose() = 4;

        std::cout << "Terminating thread-pool... " << std::flush;
        manager->Terminate();
        std::cout << "Deleting thread-pool... " << std::flush;
        manager.reset();
        //
        std::cout << "Restoring settings..." << std::endl;
        tim::settings::pop<TIMEMORY_API>();
        tim::settings::debug()   = tmp->get_debug();
        tim::settings::verbose() = tmp->get_verbose();

        std::cout << "Finalizing..." << std::endl;
        tim::timemory_finalize();
        std::cout << "Done" << std::endl;
    }
};

//--------------------------------------------------------------------------------------//

static std::atomic<int> thread_count;
int&
get_tid()
{
    static thread_local int _id = thread_count++;
    return _id;
}

//--------------------------------------------------------------------------------------//

TEST_F(ptl_tests, regions)
{
    tim::trait::runtime_enabled<ompt_native_handle>::set(false);
    user_ompt_bundle::configure<wall_clock, cpu_clock, cpu_util, thread_cpu_clock>();

    TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name());

    std::atomic<uint64_t> sum(0);
    {
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/master/0");
        auto func = [&]() {
            std::string region = "AAAAA";
            TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/worker/", region,
                                  "/", PTL::ThreadPool::get_this_thread_id());
            sum += details::fibonacci(nfib);
            details::do_sleep(500);
        };
        PTL::TaskGroup<void> tg;
        for(int i = 0; i < 4; i++)
            tg.exec(func);
        tg.join();
    }
    // synchronize the sea-level threads
    tim::manager::master_instance()->synchronize();
    {
        std::string region = "BBBBB";
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/master/1");
        auto func = [&]() {
            TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/worker/", region,
                                  "/", PTL::ThreadPool::get_this_thread_id());
            sum += details::fibonacci(nfib + noff);
            details::do_sleep(500);
        };
        PTL::TaskGroup<void> tg;
        for(int i = 0; i < 4; i++)
            tg.exec(func);
        tg.join();
    }

    printf("[%s]> sum: %lu\n", details::get_test_name().c_str(),
           static_cast<unsigned long>(sum));

    auto wc = tim::storage<wall_clock>::instance()->get();
    auto tc = tim::storage<thread_cpu_clock>::instance()->get();

    int64_t wc_nlaps = 0;
    int64_t tc_nlaps = 0;
    for(auto& itr : wc)
        wc_nlaps += itr.data().get_laps();
    for(auto& itr : tc)
        tc_nlaps += itr.data().get_laps();

    ASSERT_EQ(wc_nlaps, tc_nlaps);
}

//--------------------------------------------------------------------------------------//

TEST_F(ptl_tests, pi_task)
{
    tim::trait::runtime_enabled<ompt_native_handle>::set(true);
    user_ompt_bundle::configure<wall_clock, cpu_clock, cpu_util, thread_cpu_clock>();

    TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name());

    uint64_t num_threads = tim::get_env<uint64_t>("NUM_THREADS", 4);
    uint64_t num_steps   = tim::get_env<uint64_t>("NUM_STEPS", 500000000);

    double step   = 1.0 / static_cast<double>(num_steps);
    auto   nblock = num_steps / (2 * num_threads * num_threads);

    PTL::TaskGroup<double> tg([](double& _sum, double _thr) { return _sum += _thr; });
    tg.exec(&details::compute_pi, 0, num_steps, step, nblock);

    double pi = step * tg.join();
    printf("[%s]> pi: %f\n", details::get_test_name().c_str(), pi);

#if defined(TIMEMORY_WINDOWS) && !defined(M_PI)
#    define M_PI 3.14159265358979323846
#endif

    ASSERT_NEAR(pi, M_PI, pi_epsilon);
}

//--------------------------------------------------------------------------------------//

TEST_F(ptl_tests, async)
{
    TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name());

    auto run_fib = [&](uint64_t n) { return details::fibonacci(n); };

    std::atomic<uint64_t> sum(0);

    auto run_a = [&]() {
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/worker/AAAAA/",
                              get_tid());
        for(int i = 0; i < 2; ++i)
        {
            sum += run_fib(nfib);
            details::do_sleep(500);
        }
        if(get_tid() > 0)
            thread_count--;
    };

    auto run_b = [&]() {
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/worker/BBBBB/",
                              get_tid());
        for(int i = 0; i < 2; ++i)
        {
            sum += run_fib(44);
            details::do_sleep(nfib + noff);
        }
        if(get_tid() > 0)
            thread_count--;
    };

    PTL::TaskGroup<void> tg;

    {
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/master/", get_tid());
        tg.exec(run_a);
        details::do_sleep(10);
    }

    {
        TIMEMORY_BLANK_MARKER(tuple_t, details::get_test_name(), "/master/1");
        tg.exec(run_b);
        details::do_sleep(10);
    };

    tg.wait();

    printf("[%s]> sum: %lu\n", details::get_test_name().c_str(),
           static_cast<unsigned long>(sum));

    auto wc = tim::storage<wall_clock>::instance()->get();
    auto tc = tim::storage<thread_cpu_clock>::instance()->get();

    int64_t wc_nlaps = 0;
    int64_t tc_nlaps = 0;
    for(auto& itr : wc)
        wc_nlaps += itr.data().get_laps();
    for(auto& itr : tc)
        tc_nlaps += itr.data().get_laps();

    ASSERT_EQ(wc_nlaps, tc_nlaps);
}

//--------------------------------------------------------------------------------------//
