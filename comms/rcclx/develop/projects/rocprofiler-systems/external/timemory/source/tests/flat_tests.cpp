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

using namespace tim::component;

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

#if defined(TIMEMORY_WINDOWS)
using toolset_t = tim::auto_tuple<wall_clock, tim::quirk::flat_scope>;
#else
using toolset_t = tim::auto_tuple<wall_clock>;
#endif

TIMEMORY_DEFINE_CONCRETE_TRAIT(flat_storage, monotonic_clock, true_type)
TIMEMORY_DECLARE_EXTERN_COMPONENT(wall_clock, true, int64_t)
TIMEMORY_DECLARE_EXTERN_COMPONENT(monotonic_clock, true, int64_t)
TIMEMORY_DECLARE_EXTERN_COMPONENT(monotonic_raw_clock, true, int64_t)

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

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n, bool instr)
{
    if(instr)
    {
        TIMEMORY_BASIC_MARKER(toolset_t, "");
        return (n < 2) ? n : (fibonacci(n - 1, true) + fibonacci(n - 2, false));
    }
    else
    {
        return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
    }
}

// this function consumes an unknown number of cpu resources
inline long
static_string_fibonacci(long n, bool instr)
{
    if(instr)
    {
        toolset_t _marker{ tim::static_string{ "static_string_fibonacci" } };
        return (n < 2) ? n
                       : (static_string_fibonacci(n - 1, true) +
                          static_string_fibonacci(n - 2, false));
    }
    else
    {
        return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
    }
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

class flat_tests : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        tim::set_env(TIMEMORY_SETTINGS_PREFIX "FLAT_PROFILE", "ON", 1);
        tim::settings::verbose()     = 0;
        tim::settings::debug()       = false;
        tim::settings::json_output() = true;
        tim::settings::mpi_thread()  = false;
        tim::dmp::initialize(test_argc, test_argv);
        tim::timemory_init(test_argc, test_argv);
        tim::settings::dart_output() = true;
        tim::settings::dart_count()  = 1;
        tim::settings::banner()      = false;
        tim::settings::parse();
    }

    static void TearDownTestSuite()
    {
        tim::timemory_finalize();
        tim::dmp::finalize();
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, parse)
{
    tim::settings::flat_profile() = false;
    tim::set_env(TIMEMORY_SETTINGS_PREFIX "FLAT_PROFILE", "ON", 1);
    tim::settings::parse();
    std::cout << "\nflat_profile() = " << std::boolalpha << tim::settings::flat_profile()
              << std::endl;
    auto ret = tim::get_env<bool>(TIMEMORY_SETTINGS_PREFIX "FLAT_PROFILE", false);
    std::cout << "environment = " << std::boolalpha << ret << '\n' << std::endl;
    EXPECT_TRUE(ret);
    EXPECT_TRUE(tim::settings::flat_profile());
}

//--------------------------------------------------------------------------------------//
#if !defined(TIMEMORY_WINDOWS)

TEST_F(flat_tests, get_default)
{
    tim::settings::flat_profile() = false;
    tim::set_env(TIMEMORY_SETTINGS_PREFIX "FLAT_PROFILE", "ON", 1);
    tim::settings::parse();
    auto _scope = tim::scope::config{};
    std::cout << "\nscope: " << _scope << '\n' << std::endl;
    EXPECT_TRUE(_scope.is_flat());
    EXPECT_FALSE(_scope.is_tree());
    EXPECT_FALSE(_scope.is_timeline());
    EXPECT_FALSE(_scope.is_flat_timeline());
    EXPECT_FALSE(_scope.is_tree_timeline());
}

#endif
//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, flat)
{
    tim::settings::flat_profile() = false;
    auto _scope                   = tim::scope::config(tim::scope::flat{});
    std::cout << "\nscope: " << _scope << '\n' << std::endl;
    EXPECT_TRUE(_scope.is_flat());
    EXPECT_FALSE(_scope.is_tree());
    EXPECT_FALSE(_scope.is_timeline());
    EXPECT_FALSE(_scope.is_flat_timeline());
    EXPECT_FALSE(_scope.is_tree_timeline());
}

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, quirk)
{
    tim::settings::flat_profile() = false;
    using bundle_t                = tim::component_bundle<TIMEMORY_API, wall_clock>;
    bundle_t _bundle(details::get_test_name(),
                     tim::quirk::config<tim::quirk::flat_scope>{});
    auto     _scope = _bundle.get_scope();
    std::cout << "\nscope: " << _scope << '\n' << std::endl;
    EXPECT_TRUE(_scope.is_flat());
    EXPECT_FALSE(_scope.is_tree());
    EXPECT_FALSE(_scope.is_timeline());
    EXPECT_FALSE(_scope.is_flat_timeline());
    EXPECT_FALSE(_scope.is_tree_timeline());
}

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, type_trait)
{
    tim::settings::flat_profile()     = false;
    tim::settings::timeline_profile() = false;
    tim::trait::report<monotonic_clock>::depth(true);
    using bundle_t = tim::component_bundle<TIMEMORY_API, monotonic_clock>;
    bundle_t _outer(details::get_test_name());
    bundle_t _inner(details::get_test_name());
    _outer.start();
    _inner.start();
    details::consume(1000);
    _inner.stop();
    _outer.stop();

    monotonic_clock* _outer_mc = _outer.get<monotonic_clock>();
    monotonic_clock* _inner_mc = _inner.get<monotonic_clock>();

    EXPECT_TRUE(_outer_mc->get_is_flat());
    EXPECT_FALSE(_outer_mc->get_depth_change());
    EXPECT_TRUE(_inner_mc->get_is_flat());
    EXPECT_FALSE(_inner_mc->get_depth_change());

    EXPECT_EQ(_outer_mc->get_iterator()->data().get_laps(), 2);
    EXPECT_EQ(_inner_mc->get_iterator()->depth(), 1);
    EXPECT_TRUE(_outer_mc->get_iterator() == _inner_mc->get_iterator());

    auto _scope = _outer.get_scope();
    std::cout << "\nscope: " << _scope << '\n' << std::endl;
    EXPECT_TRUE(_scope.is_tree());
    EXPECT_FALSE(_scope.is_flat());
    EXPECT_FALSE(_scope.is_timeline());
    EXPECT_FALSE(_scope.is_flat_timeline());
    EXPECT_FALSE(_scope.is_tree_timeline());
    EXPECT_TRUE(_scope == _inner.get_scope());
}

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, general_quirk)
{
    tim::settings::flat_profile() = false;
    using bundle_t                = tim::component_bundle<TIMEMORY_API, wall_clock>;
    bundle_t _bundle(details::get_test_name(),
                     tim::quirk::config<tim::quirk::flat_scope, tim::quirk::no_store>{});
    auto     _scope1 = _bundle.get_scope();
    auto     _store1 = _bundle.get_store();
    auto     _scope2 = bundle_t::get_scope_config<tim::quirk::flat_scope>();
    auto _store2 = bundle_t::get_store_config<tim::quirk::config<tim::quirk::no_store>>();

    std::cout << "\n[1] scope config : " << std::boolalpha << _scope1 << '\n';
    std::cout << "[1] store config : " << std::boolalpha << _store1 << '\n';

    EXPECT_TRUE(_scope1.is_flat());
    EXPECT_FALSE(_scope1.is_tree());
    EXPECT_FALSE(_scope1.is_timeline());
    EXPECT_FALSE(_scope1.is_flat_timeline());
    EXPECT_FALSE(_scope1.is_tree_timeline());
    EXPECT_FALSE(_store1);

    std::cout << "\n[2] scope config : " << std::boolalpha << _scope2 << '\n';
    std::cout << "[2] store config : " << std::boolalpha << _store2 << '\n';

    EXPECT_TRUE(_scope2.is_flat());
    EXPECT_FALSE(_scope2.is_tree());
    EXPECT_FALSE(_scope2.is_timeline());
    EXPECT_FALSE(_scope2.is_flat_timeline());
    EXPECT_FALSE(_scope2.is_tree_timeline());
    EXPECT_FALSE(_store2);
}

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, general)
{
    auto bsize = tim::storage<wall_clock>::instance()->size();

    {
        long n = 25;
        TIMEMORY_BLANK_MARKER(toolset_t, details::get_test_name());
        auto ret = details::fibonacci(n, true);
        printf("\nfibonacci(%li) = %li\n", n, ret);
    }

    auto esize = tim::storage<wall_clock>::instance()->size();
    printf("\nbsize = %lu\n", (unsigned long) bsize);
    printf("esize = %lu\n\n", (unsigned long) esize);
    auto data = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(esize - bsize, 2);
    EXPECT_EQ(data.at(bsize + 1).depth(), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(flat_tests, general_static_string)
{
    auto bsize = tim::storage<wall_clock>::instance()->size();

    {
        long      n = 25;
        toolset_t _marker{ tim::static_string{ "flat_tests.general_static_string" } };
        auto      ret = details::static_string_fibonacci(n, true);
        printf("\nfibonacci(%li) = %li\n", n, ret);
    }

    auto esize = tim::storage<wall_clock>::instance()->size();
    printf("\nbsize = %lu\n", (unsigned long) bsize);
    printf("esize = %lu\n\n", (unsigned long) esize);
    auto data = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(esize - bsize, 2);
    EXPECT_EQ(data.at(bsize + 1).depth(), 0);
}

//--------------------------------------------------------------------------------------//
