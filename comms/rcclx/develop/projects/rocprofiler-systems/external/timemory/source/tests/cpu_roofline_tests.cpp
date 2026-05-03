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

#include "timemory/components/roofline/cpu_roofline.hpp"
#include "timemory/ert.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using namespace tim::component;

using float_type   = double;
using roofline_t   = cpu_roofline<float_type>;
using fib_list_t   = std::vector<int64_t>;
using auto_tuple_t = tim::auto_tuple_t<wall_clock, cpu_clock, cpu_util, roofline_t>;
using auto_list_t  = tim::auto_list_t<wall_clock, cpu_clock, cpu_util, roofline_t>;
using device_t     = tim::device::cpu;
using roofline_ert_config_t = typename roofline_t::ert_config_type<float_type>;

// unless specified number of threads, use the number of available cores
#if !defined(NUM_THREADS)
#    define NUM_THREADS std::thread::hardware_concurrency()
#endif

void
check(auto_list_t& l);
void
check_const(const auto_list_t& l);
template <typename Tp>
void customize_roofline(int64_t, int64_t, int64_t);

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
inline double
fibonacci(double n)
{
    return (n < 2.0) ? n : (fibonacci(n - 1.0) + fibonacci(n - 2.0));
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

// get a random entry from vector
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class cpu_roofline_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(cpu_roofline_tests, run)
{
    tim::print_env();

    int64_t    num_threads   = 2;           // default number of threads
    int64_t    working_size  = 64;          // default working set size
    int64_t    memory_factor = 2;           // default multiple of max cache size
    fib_list_t fib_values    = { 30, 35 };  // default values for fibonacci calcs

    //
    // override method for determining how many threads to run
    //
    roofline_ert_config_t::get_num_threads() = [=]() { return num_threads; };

    //
    // allow for customizing the roofline
    //
    if(tim::get_env("CUSTOMIZE_ROOFLINE", true))
        customize_roofline<float_type>(num_threads, working_size, memory_factor);

    //
    // execute fibonacci in a thread
    //
    auto exec_fibonacci = [&](int64_t n) {
        TIMEMORY_BLANK_CALIPER(0, auto_tuple_t, "fibonacci(", n, ")");
        auto ret = details::fibonacci(n);
        TIMEMORY_CALIPER_APPLY0(0, stop);
        printf("fibonacci(%li) = %.1f\n", static_cast<long>(n), ret);
    };

    //
    // overall timing
    //
    auto _main = TIMEMORY_BLANK_HANDLE(auto_tuple_t, "overall_timer");
    _main.report_at_exit(true);
    wall_clock total;
    total.start();

    //
    // run fibonacci calculations
    //
    for(const auto& n : fib_values)
    {
        std::vector<std::thread> threads;
        for(int64_t i = 0; i < num_threads; ++i)
            threads.push_back(std::thread(exec_fibonacci, n));
        for(auto& itr : threads)
            itr.join();
    }

    //
    // stop the overall timing
    //
    _main.stop();

    //
    // overall timing
    //
    std::this_thread::sleep_for(std::chrono::seconds(1));
    total.stop();

    std::cout << "Total time: " << total << std::endl;

    auto _roofl = _main.get<roofline_t>();
    if(_roofl)
        std::cout << *_roofl << std::endl;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
customize_roofline(int64_t num_threads, int64_t working_size, int64_t memory_factor)
{
    using ert_params_t   = tim::ert::exec_params;
    using ert_data_ptr_t = typename roofline_t::ert_data_ptr_t;
    using ert_counter_t  = typename roofline_t::ert_counter_type<Tp>;

    // sets up the configuration
    roofline_ert_config_t::get_executor() = [=](ert_data_ptr_t data) {
        // test getting the cache info
        auto l1_size = tim::ert::cache_size::get<1>();
        auto l2_size = tim::ert::cache_size::get<2>();
        auto l3_size = tim::ert::cache_size::get<3>();
        auto lm_size = tim::ert::cache_size::get_max();

        auto     dtype      = tim::demangle(typeid(Tp).name());
        uint64_t align_size = 64;
#if defined(TIMEMORY_RELAXED_TESTING)
        uint64_t max_size = memory_factor * l2_size;
#else
        uint64_t max_size = memory_factor * lm_size;
#endif

        // log the cache info
        std::cout << "[INFO]> L1 cache size: " << (l1_size / tim::units::kilobyte)
                  << " KB, L2 cache size: " << (l2_size / tim::units::kilobyte)
                  << " KB, L3 cache size: " << (l3_size / tim::units::kilobyte)
                  << " KB, max cache size: " << (lm_size / tim::units::kilobyte)
                  << " KB\n\n"
                  << "[INFO]> num-threads      : " << num_threads << "\n"
                  << "[INFO]> min-working-set  : " << working_size << " B\n"
                  << "[INFO]> max-data-size    : " << max_size << " B\n"
                  << "[INFO]> alignment        : " << align_size << "\n"
                  << "[INFO]> data type        : " << dtype << "\n"
                  << std::endl;

        ert_params_t  params(working_size, max_size, num_threads);
        ert_counter_t _counter(params, data, align_size);

        return _counter;
    };

    // does the execution of ERT
    auto callback = [=](ert_counter_t& _counter) {
        // these are the kernel functions we want to calculate the peaks with
        auto store_func = [](Tp& a, const Tp& b) { a = b; };
        auto add_func   = [](Tp& a, const Tp& b, const Tp& c) { a = b + c; };
        auto fma_func   = [](Tp& a, const Tp& b, const Tp& c) { a = a * b + c; };

        // set bytes per element
        _counter.bytes_per_element = sizeof(Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = "scalar_add";
        // run the operation _counter kernels
        tim::ert::ops_main<1>(_counter, add_func, store_func);

        // set the label
        _counter.label = "vector_fma";
        // run the kernels (<4> is ideal for avx, <8> is ideal for KNL)
        tim::ert::ops_main<4>(_counter, fma_func, store_func);
    };

    // set the callback
    roofline_t::set_executor_callback<Tp>(callback);
}

//--------------------------------------------------------------------------------------//
