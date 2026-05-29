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

#include "timemory/components/papi/types.hpp"
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

using tuple_t = tim::component_tuple<wall_clock, cpu_clock, cpu_util, peak_rss>;
using list_t  = tim::component_list<wall_clock, cpu_clock, cpu_util, peak_rss, page_rss,
                                   papi_array_t, vtune_frame, vtune_event>;
using auto_hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
using hybrid_t      = typename auto_hybrid_t::component_type;

static const int64_t niter       = 20;
static const int64_t nelements   = 0.95 * (tim::units::get_page_size() * 500);
static const auto    memory_unit = std::pair<int64_t, string_t>(tim::units::KiB, "KiB");

// acceptable absolute error
#if defined(TIMEMORY_MACOS)
// GitHub CI for macOS is very noisy
static const double util_tolerance  = 10.0;
static const double timer_tolerance = 0.2;
#else
static const double util_tolerance  = 5.0;
static const double timer_tolerance = 0.1;
#endif

// acceptable relative error
// static const double util_epsilon  = 0.5;
// static const double timer_epsilon = 0.02;

// acceptable compose error
static const double compose_tolerance = 1.0e-9;

//--------------------------------------------------------------------------------------//

#include "timemory/operations/types/compose.hpp"

inline tim::component::cpu_clock
operator+(const tim::component::user_clock&   cuser,
          const tim::component::system_clock& csys)
{
    return tim::operation::compose<tim::component::cpu_clock, tim::component::user_clock,
                                   tim::component::system_clock>::generate(cuser, csys);
}

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
TIMEMORY_FLATTEN void
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
TIMEMORY_FLATTEN void
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

// this function ensures an allocation cannot be optimized
template <typename Tp>
inline size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

inline void
allocate()
{
    std::vector<int64_t> v(nelements, 15);
    auto                 ret  = fibonacci(0);
    long                 nfib = details::random_entry(v);
    for(int64_t i = 0; i < niter; ++i)
    {
        nfib = details::random_entry(v);
        ret += details::fibonacci(nfib);
    }
    printf("fibonacci(%li) * %li = %li\n", (long) nfib, (long) niter, ret);
}

// template <typename Tp, typename Up, typename Vp = typename Tp::value_type,
//           typename FuncT = std::function<Vp(Vp)>>
// inline void
// print_info(
//     const Tp& obj, const Up& expected, const string_t& unit,
//     FuncT _func = [](const Vp& _obj) { return _obj; })
// {
//     std::cout << std::endl;
//     std::cout << "[" << get_test_name() << "]>  measured : " << obj << std::endl;
//     std::cout << "[" << get_test_name() << "]>  expected : " << expected << " " << unit
//               << std::endl;
//     std::cout << "[" << get_test_name() << "]>     value : " << _func(obj.get_value())
//               << std::endl;
//     std::cout << "[" << get_test_name() << "]>     accum : " << _func(obj.get_accum())
//               << std::endl;
// }

}  // namespace details

//--------------------------------------------------------------------------------------//

class hybrid_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    void SetUp() override
    {
        TIMEMORY_TEST_CHECK_SKIP;
#if defined(TIMEMORY_USE_PAPI)
        papi_array_t::get_initializer() = []() {
            return std::vector<std::string>{ "PAPI_TOT_CYC" };
        };
#else
        static_assert(list_t::can_heap_init<papi_array_t>() == false,
                      "Error! should not be able to heap initialize!");
#endif

        static_assert(tuple_t::can_stack_init<wall_clock>() == true,
                      "Error! should be able to stack initialize!");

        static_assert(list_t::can_stack_init<wall_clock>() == false,
                      "Error! should not be able to stack initialize!");

        static_assert(list_t::can_heap_init<wall_clock>() == true,
                      "Error! should be able to heap initialize!");

        hybrid_t::get_initializer() = [](auto& l) {
            l.template initialize<wall_clock, cpu_clock, cpu_util, peak_rss, page_rss,
                                  papi_array_t>();
        };
    }
};

//--------------------------------------------------------------------------------------//

template <typename T>
using identity_type_t = typename T::type;

TEST_F(hybrid_tests, type_check)
{
    std::cout << tim::demangle<auto_hybrid_t>() << std::endl;
    std::cout << tim::demangle<hybrid_t>() << std::endl;
    std::cout << tim::demangle<typename auto_hybrid_t::base_type>() << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, hybrid)
{
    // warm up
    {
        namespace quirk = tim::quirk;
        hybrid_t{ details::get_test_name(),
                  quirk::config<quirk::explicit_push, quirk::explicit_pop>{} }
            .start()
            .stop();
    }

    hybrid_t obj(details::get_test_name());

    std::promise<void> _prom{};
    auto               _consume = [&_prom](long _n) {
        _prom.set_value();
        details::consume(_n);
    };
    auto        _fut = _prom.get_future();
    std::thread t(_consume, 1000);
    _fut.wait();
    obj.start();
    details::do_sleep(500);
    details::consume(1500);
    t.join();
    obj.stop();
    // std::cout << "\n" << obj << std::endl;

    // auto clock_convert    = [](const int64_t& _obj) { return _obj; };
    // auto cpu_util_convert = [](const std::pair<int64_t, int64_t>& val) {
    //     return static_cast<double>(val.first) / val.second * 100.0;
    // };

    auto* t_rc   = obj.get<wall_clock>();
    auto* t_cpu  = obj.get<cpu_clock>();
    auto* t_util = obj.get<cpu_util>();

    // details::print_info(*t_rc, 2.0, "sec", clock_convert);
    // details::print_info(*t_cpu, 2.5, "sec", clock_convert);
    // details::print_info(*t_util, 125.0, "%", cpu_util_convert);

    ASSERT_TRUE(t_rc != nullptr) << obj;
    ASSERT_TRUE(t_cpu != nullptr) << obj;
    ASSERT_TRUE(t_util != nullptr) << obj;

    EXPECT_NEAR(2.0, t_rc->get(), timer_tolerance) << obj;
    EXPECT_NEAR(2.5, t_cpu->get(), 2.0 * timer_tolerance) << obj;
    EXPECT_NEAR(125.0, t_util->get(), 2.0 * util_tolerance) << obj;

    auto* l_rc   = obj.get<wall_clock*>();
    auto* l_cpu  = obj.get<cpu_clock*>();
    auto* l_util = obj.get<cpu_util*>();
    auto* l_page = obj.get<page_rss>();

    // std::cout << tim::demangle<hybrid_t>() << std::endl;
    // std::cout << obj << std::endl;

    EXPECT_EQ(l_rc, nullptr) << obj;
    EXPECT_EQ(l_cpu, nullptr) << obj;
    EXPECT_EQ(l_util, nullptr) << obj;
    ASSERT_NE(l_page, nullptr) << obj;

    auto page_size = tim::units::get_page_size();

    // details::print_info(*l_page, page_size, tim::settings::memory_units(),
    // clock_convert);

    EXPECT_NEAR(l_page->get(), page_size, 2.0 * page_size) << obj;
}

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, auto_timer)
{
    tim::auto_timer obj(details::get_test_name());
    std::thread     t(details::consume, 500);
    details::do_sleep(250);
    details::consume(750);
    t.join();
    obj.stop();
    // std::cout << "\n" << obj << std::endl;

    // auto clock_convert    = [](const int64_t& _obj) { return _obj; };
    // auto cpu_util_convert = [](const std::pair<int64_t, int64_t>& val) {
    //     return static_cast<double>(val.first) / val.second * 100.0;
    // };

    auto  _cpu  = *obj.get<cpu_clock>();
    auto& _rc   = *obj.get<wall_clock>();
    auto& _util = *obj.get<cpu_util>();

    // details::print_info(_rc, 1.0, "sec", clock_convert);
    // details::print_info(_cpu, 1.25, "sec", clock_convert);
    // details::print_info(_util, 125.0, "%", cpu_util_convert);

    EXPECT_NEAR(1.0, _rc.get(), timer_tolerance);
    EXPECT_NEAR(1.25, _cpu.get(), 2.0 * timer_tolerance);
    EXPECT_NEAR(125.0, _util.get(), 2.0 * util_tolerance);

    cpu_clock _cpu_obj = *obj.get<cpu_clock>();
    double    _cpu_val = obj.get<cpu_clock>()->get();
    EXPECT_NEAR(_cpu_obj.get(), _cpu_val, compose_tolerance);
    // details::print_info(_cpu_obj, _cpu_val, "sec");

    auto _obj  = tim::get(obj);
    auto _cpu2 = std::get<1>(_obj);

    EXPECT_NEAR(1.0e-9, _cpu.get(), _cpu2);
}

//--------------------------------------------------------------------------------------//

TEST_F(hybrid_tests, compose)
{
    using bundle_t = tim::component_tuple<user_clock, system_clock>;
    using result_t = std::tuple<double, double>;

    bundle_t obj(details::get_test_name());
    obj.start();
    details::do_sleep(250);  // in millseconds
    details::consume(750);   // in millseconds
    obj.stop();
    // std::cout << "\n" << obj << std::endl;

    result_t  _cpu_ret = obj.get();
    cpu_clock _cpu_obj = *obj.get<user_clock>() + *obj.get<system_clock>();
    double    _cpu_val = obj.get<user_clock>()->get() + obj.get<system_clock>()->get();

    // details::print_info(_cpu_obj, 0.75, "sec");

    EXPECT_NEAR(0.75, _cpu_val, 2.0 * timer_tolerance);
    EXPECT_NEAR(_cpu_obj.get(), _cpu_val, compose_tolerance);
    EXPECT_NEAR(_cpu_val, std::get<0>(_cpu_ret) + std::get<1>(_cpu_ret),
                compose_tolerance);

    printf("\n");
}

//--------------------------------------------------------------------------------------//
