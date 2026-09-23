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
//

#include "timemory/ert.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/testing.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>

using namespace tim::component;

#if !defined(ROOFLINE_FP_BYTES)
#    define ROOFLINE_FP_BYTES 8
#endif

#if ROOFLINE_FP_BYTES == 8
using float_type = double;
#elif ROOFLINE_FP_BYTES == 4
using float_type = float;
#else
#    error "ROOFLINE_FP_BYTES must be either 4 or 8"
#endif

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

//--------------------------------------------------------------------------------------//

float_type
fibonacci(float_type n);
float_type
random_fibonacci(float_type n);
void
check(auto_list_t& l);
void
check_const(const auto_list_t& l);
template <typename Tp>
void customize_roofline(int64_t, int64_t, int64_t);

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::settings::verbose()     = 1;
    tim::settings::json_output() = true;
    tim::dmp::initialize(argc, argv);
    tim::timemory_init(argc, argv);
    tim::print_env();

    int64_t    num_threads   = 2;                   // default number of threads
    int64_t    working_size  = 64;                  // default working set size
    int64_t    memory_factor = 2;                   // default multiple of max cache size
    fib_list_t fib_values    = { { 35, 38, 43 } };  // default values for fibonacci calcs

    if(argc > 1)
        num_threads = atol(argv[1]);
    if(argc > 2)
        working_size = atol(argv[2]);
    if(argc > 3)
        memory_factor = atol(argv[3]);
    if(argc > 4)
    {
        fib_values.clear();
        for(int i = 4; i < argc; ++i)
            fib_values.push_back(atol(argv[i]));
    }

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
        auto ret = fibonacci(n);
        TIMEMORY_CALIPER_APPLY(0, stop);
        printf("fibonacci(%li) = %.1f\n", static_cast<long>(n), ret);
    };

    //
    // execute random_fibonacci in a thread
    //
    auto exec_random_fibonacci = [&](int64_t n) {
        TIMEMORY_BLANK_CALIPER(1, auto_tuple_t, "random_fibonacci(", n, ")");
        auto ret = random_fibonacci(n);
        TIMEMORY_CALIPER_APPLY(1, stop);
        printf("random_fibonacci(%li) = %.1f\n", static_cast<long>(n), ret);
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
    // run the random fibonacci calculations
    //
    for(const auto& n : fib_values)
    {
        std::vector<std::thread> threads;
        for(int64_t i = 0; i < num_threads; ++i)
            threads.push_back(std::thread(exec_random_fibonacci, n));
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

    auto_list_t l(__FUNCTION__);
    check(l);
    check_const(l);
    std::cout << std::endl;

    tim::timemory_finalize();

    std::cout << std::flush;
    return 0;
}

//--------------------------------------------------------------------------------------//

#define ftwo static_cast<float_type>(2.0)
#define fone static_cast<float_type>(1.0)

//--------------------------------------------------------------------------------------//

float_type
fibonacci(float_type n)
{
    return (n < ftwo) ? n : fone * (fibonacci(n - 1) + fibonacci(n - 2));
}

//--------------------------------------------------------------------------------------//

float_type
random_fibonacci(float_type n)
{
    // this is intentionally different between runs so that we scatter
    // the arithmetic intensity vs. the compute rate between runs to simulate
    // different algorithms
    static std::atomic<int>                tid;
    static thread_local std::random_device rd;
    static thread_local std::mt19937       gen(rd() + tid++);
    auto                                   get_random = [&]() {
        return 1.75 * std::generate_canonical<float_type, 16>(gen) + 0.9;
    };
    auto m1 = get_random();
    auto m2 = get_random();
    return (n < ftwo) ? n : (random_fibonacci(n - m1) + random_fibonacci(n - m2));
}

//--------------------------------------------------------------------------------------//

void
check_const(const auto_list_t& l)
{
    const wall_clock* rc = l.get<wall_clock>();
    std::cout << "[demangle-test]> type: " << tim::demangle(typeid(rc).name())
              << std::endl;
}

//--------------------------------------------------------------------------------------//

void
check(auto_list_t& l)
{
    wall_clock* rc = l.get<wall_clock>();
    std::cout << "[demangle-test]> type: " << tim::demangle(typeid(rc).name())
              << std::endl;
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
        uint64_t max_size   = memory_factor * lm_size;

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
        tim::ert::ops_main<4, 8>(_counter, fma_func, store_func);
    };

    // set the callback
    roofline_t::set_executor_callback<Tp>(callback);
}

//--------------------------------------------------------------------------------------//
