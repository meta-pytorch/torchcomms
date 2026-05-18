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

#if defined(__GNUC__)
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

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

class derived_tests : public ::testing::Test
{
protected:
    static void config() { tim::settings::mpi_thread() = false; }
    TIMEMORY_TEST_SUITE_SETUP(config())
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_tuple_wc_cc)
{
    // std::cout << '\n';
    using toolset_t = tim::component_tuple<wall_clock, cpu_clock, cpu_util>;

    toolset_t obj(details::get_test_name());
    obj.start();
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_TRUE(obj.get<cpu_util>()->is_derived());
    auto manual_calc = 100. * obj.get<cpu_clock>()->get() / obj.get<wall_clock>()->get();
    ASSERT_NEAR(obj.get<cpu_util>()->get(), manual_calc, 1.0e-6);
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_tuple_wc_uc_sc)
{
    // std::cout << '\n';
    using toolset_t = tim::auto_tuple<wall_clock, user_clock, system_clock, cpu_util>;

    toolset_t obj(details::get_test_name());
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_TRUE(obj.get<cpu_util>()->is_derived());
    auto manual_calc = 100. *
                       (obj.get<user_clock>()->get() + obj.get<system_clock>()->get()) /
                       obj.get<wall_clock>()->get();
    ASSERT_NEAR(obj.get<cpu_util>()->get(), manual_calc, 1.0e-6);
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_tuple_wc)
{
    // std::cout << '\n';
    using toolset_t = tim::auto_tuple<wall_clock, cpu_util>;

    toolset_t obj(details::get_test_name());
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_FALSE(obj.get<cpu_util>()->is_derived());
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_list_wc_cc)
{
    // std::cout << '\n';
    using toolset_t = tim::component_list<wall_clock, cpu_clock, cpu_util>;

    toolset_t::get_initializer() = [](toolset_t& cl) {
        cl.initialize<wall_clock, cpu_clock, cpu_util>();
    };

    toolset_t obj(details::get_test_name());
    obj.start();
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_TRUE(obj.get<cpu_util>()->is_derived());
    auto manual_calc = 100. * obj.get<cpu_clock>()->get() / obj.get<wall_clock>()->get();
    ASSERT_NEAR(obj.get<cpu_util>()->get(), manual_calc, 1.0e-6);
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_list_wc_uc_sc)
{
    // std::cout << '\n';
    using toolset_t = tim::auto_list<wall_clock, user_clock, system_clock, cpu_util>;

    toolset_t::get_initializer() = [](toolset_t& cl) {
        cl.initialize<wall_clock, user_clock, system_clock, cpu_util>();
    };

    toolset_t obj(details::get_test_name());
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_TRUE(obj.get<cpu_util>()->is_derived());
    auto manual_calc = 100. *
                       (obj.get<user_clock>()->get() + obj.get<system_clock>()->get()) /
                       obj.get<wall_clock>()->get();
    ASSERT_NEAR(obj.get<cpu_util>()->get(), manual_calc, 1.0e-6);
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_list_wc)
{
    // std::cout << '\n';
    using toolset_t = tim::auto_list<wall_clock, cpu_util>;

    toolset_t::get_initializer() = [](toolset_t& cl) {
        cl.initialize<wall_clock, cpu_util>();
    };

    toolset_t obj(details::get_test_name());
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_FALSE(obj.get<cpu_util>()->is_derived());
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_bundle_wc_cc)
{
    // std::cout << '\n';
    using toolset_t =
        tim::component_bundle<TIMEMORY_API, wall_clock, cpu_clock*, cpu_util*>;

    toolset_t::get_initializer() = [](auto& cl) {
        cl.template initialize<cpu_clock, cpu_util>();
    };

    toolset_t obj(details::get_test_name());
    obj.start();
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_TRUE(obj.get_component<cpu_clock>() != nullptr);
    ASSERT_TRUE(obj.get_component<cpu_util>() != nullptr);
    EXPECT_TRUE(obj.get_component<cpu_util>()->is_derived());
    auto manual_calc = 100. * obj.get_component<cpu_clock>()->get() /
                       obj.get_component<wall_clock>()->get();
    EXPECT_NEAR(obj.get_component<cpu_util>()->get(), manual_calc, 1.0e-6);
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_bundle_wc_uc_sc)
{
    // std::cout << '\n';
    using toolset_t =
        tim::auto_bundle<TIMEMORY_API, wall_clock, user_clock, system_clock*, cpu_util*>;

    toolset_t::get_initializer() = [](toolset_t& cl) {
        cl.initialize<system_clock, cpu_util>();
    };

    toolset_t obj(details::get_test_name());
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_TRUE(obj.get_component<cpu_util>()->is_derived());
    auto manual_calc = 100. *
                       (obj.get_component<user_clock>()->get() +
                        obj.get_component<system_clock>()->get()) /
                       obj.get_component<wall_clock>()->get();
    ASSERT_NEAR(obj.get_component<cpu_util>()->get(), manual_calc, 1.0e-6);
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(derived_tests, cpu_util_bundle_wc)
{
    // std::cout << '\n';
    using toolset_t = tim::auto_bundle<tim::project::timemory, wall_clock, cpu_util*>;

    toolset_t::get_initializer() = [](toolset_t& cl) { cl.initialize<cpu_util>(); };

    toolset_t obj(details::get_test_name());
    details::consume(1000);
    obj.stop();

    // std::cout << obj << "\n";

    ASSERT_FALSE(obj.get_component<cpu_util>()->is_derived());
    // std::cout << '\n';
}

//--------------------------------------------------------------------------------------//
