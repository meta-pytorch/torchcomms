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
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

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

class type_trait_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

using namespace tim;
using namespace tim::component;

#define TEST_API TIMEMORY_API
#define TEST_NAME(A, ...)                                                                \
    std::string(#A).substr(0, std::string(#A).find_first_of('<')).c_str()

#define TEST_TRUE_TRAIT(CONCEPT, ...)                                                    \
    {                                                                                    \
        constexpr auto expr_v = CONCEPT<__VA_ARGS__>::value;                             \
        printf("%s for %s is %s\n", TEST_NAME(CONCEPT, ""), TEST_NAME(__VA_ARGS__, ""),  \
               (expr_v) ? "true" : "false");                                             \
        EXPECT_TRUE(expr_v);                                                             \
    }

#define TEST_FALSE_TRAIT(CONCEPT, ...)                                                   \
    {                                                                                    \
        constexpr auto expr_v = CONCEPT<__VA_ARGS__>::value;                             \
        printf("%s for %s is %s\n", TEST_NAME(CONCEPT, ""), TEST_NAME(__VA_ARGS__, ""),  \
               (expr_v) ? "true" : "false");                                             \
        EXPECT_FALSE(expr_v);                                                            \
    }

//--------------------------------------------------------------------------------------//

TEST_F(type_trait_tests, is_null_type)
{
    puts("");
    puts("[expect: false]");
    TEST_FALSE_TRAIT(concepts::is_null_type, double)
    TEST_FALSE_TRAIT(concepts::is_null_type, true_type)
    TEST_FALSE_TRAIT(concepts::is_null_type, std::tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_null_type, type_list<wall_clock, cpu_clock>)
    TEST_FALSE_TRAIT(concepts::is_null_type, component_bundle<TEST_API, wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_null_type, component_tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_null_type, component_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_null_type, auto_bundle<TEST_API, wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_null_type, auto_tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_null_type, auto_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_null_type, lightweight_tuple<wall_clock>)
    puts("[expect: true]");
    TEST_TRUE_TRAIT(concepts::is_null_type, void)
    TEST_TRUE_TRAIT(concepts::is_null_type, false_type)
    TEST_TRUE_TRAIT(concepts::is_null_type, null_type)
    TEST_TRUE_TRAIT(concepts::is_null_type, std::tuple<>)
    TEST_TRUE_TRAIT(concepts::is_null_type, type_list<>)
    TEST_TRUE_TRAIT(concepts::is_null_type, component_tuple<>)
    TEST_TRUE_TRAIT(concepts::is_null_type, component_list<>)
    TEST_TRUE_TRAIT(concepts::is_null_type, auto_tuple<>)
    TEST_TRUE_TRAIT(concepts::is_null_type, auto_list<>)
    TEST_TRUE_TRAIT(concepts::is_null_type, lightweight_tuple<>)
    puts("");
}

TEST_F(type_trait_tests, is_empty)
{
    puts("");
    puts("[expect: false]");
    TEST_FALSE_TRAIT(concepts::is_empty, std::tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_empty, type_list<wall_clock, cpu_clock>)
    TEST_FALSE_TRAIT(concepts::is_empty, component_bundle<TEST_API, wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_empty, component_tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_empty, component_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_empty, auto_bundle<TEST_API, wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_empty, auto_tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_empty, auto_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_empty, lightweight_tuple<wall_clock>)
    puts("[expect: true]");
    TEST_TRUE_TRAIT(concepts::is_empty, std::tuple<>)
    TEST_TRUE_TRAIT(concepts::is_empty, type_list<>)
    TEST_TRUE_TRAIT(concepts::is_empty, component_bundle<TEST_API>)
    TEST_TRUE_TRAIT(concepts::is_empty, component_tuple<>)
    TEST_TRUE_TRAIT(concepts::is_empty, component_list<>)
    TEST_TRUE_TRAIT(concepts::is_empty, auto_bundle<TEST_API>)
    TEST_TRUE_TRAIT(concepts::is_empty, auto_tuple<>)
    TEST_TRUE_TRAIT(concepts::is_empty, auto_list<>)
    TEST_TRUE_TRAIT(concepts::is_empty, lightweight_tuple<>)
    puts("");
}

TEST_F(type_trait_tests, is_variadic)
{
    puts("");
    TEST_TRUE_TRAIT(concepts::is_variadic, std::tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_variadic, type_list<wall_clock, cpu_clock>)
    TEST_TRUE_TRAIT(concepts::is_variadic, component_bundle<TEST_API, wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_variadic, component_tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_variadic, component_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_variadic, auto_bundle<TEST_API, wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_variadic, auto_tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_variadic, auto_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_variadic, lightweight_tuple<wall_clock>)
    puts("");
}

TEST_F(type_trait_tests, is_wrapper)
{
    puts("");
    TEST_FALSE_TRAIT(concepts::is_wrapper, std::tuple<wall_clock, cpu_clock>)
    TEST_FALSE_TRAIT(concepts::is_wrapper, type_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_wrapper, component_bundle<TEST_API, wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_wrapper, component_tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_wrapper, component_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_wrapper, auto_bundle<TEST_API, wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_wrapper, auto_tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_wrapper, auto_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_wrapper, lightweight_tuple<wall_clock>)
    puts("");
}

TEST_F(type_trait_tests, is_stack_wrapper)
{
    puts("");
    TEST_FALSE_TRAIT(concepts::is_stack_wrapper, std::tuple<wall_clock, cpu_clock>)
    TEST_FALSE_TRAIT(concepts::is_stack_wrapper, type_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_stack_wrapper, component_bundle<TEST_API, wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_stack_wrapper, component_tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_stack_wrapper, component_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_stack_wrapper, auto_bundle<TEST_API, wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_stack_wrapper, auto_tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_stack_wrapper, auto_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_stack_wrapper, lightweight_tuple<wall_clock>)
    puts("");
}

TEST_F(type_trait_tests, is_heap_wrapper)
{
    puts("");
    TEST_FALSE_TRAIT(concepts::is_heap_wrapper, std::tuple<wall_clock, cpu_clock>)
    TEST_FALSE_TRAIT(concepts::is_heap_wrapper, type_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_heap_wrapper, component_bundle<TEST_API, wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_heap_wrapper, component_tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_heap_wrapper, component_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_heap_wrapper, auto_bundle<TEST_API, wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_heap_wrapper, auto_tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_heap_wrapper, auto_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_heap_wrapper, lightweight_tuple<wall_clock>)
    puts("");
}

TEST_F(type_trait_tests, is_auto_wrapper)
{
    puts("");
    TEST_FALSE_TRAIT(concepts::is_auto_wrapper, std::tuple<wall_clock, cpu_clock>)
    TEST_FALSE_TRAIT(concepts::is_auto_wrapper, type_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_auto_wrapper, component_bundle<TEST_API, wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_auto_wrapper, component_tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_auto_wrapper, component_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_auto_wrapper, auto_bundle<TEST_API, wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_auto_wrapper, auto_tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_auto_wrapper, auto_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_auto_wrapper, lightweight_tuple<wall_clock>)
    puts("");
}

TEST_F(type_trait_tests, is_comp_wrapper)
{
    puts("");
    TEST_FALSE_TRAIT(concepts::is_comp_wrapper, std::tuple<wall_clock, cpu_clock>)
    TEST_FALSE_TRAIT(concepts::is_comp_wrapper, type_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_comp_wrapper, component_bundle<TEST_API, wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_comp_wrapper, component_tuple<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_comp_wrapper, component_list<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_comp_wrapper, auto_bundle<TEST_API, wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_comp_wrapper, auto_tuple<wall_clock>)
    TEST_FALSE_TRAIT(concepts::is_comp_wrapper, auto_list<wall_clock>)
    TEST_TRUE_TRAIT(concepts::is_comp_wrapper, lightweight_tuple<wall_clock>)
    puts("");
}

//--------------------------------------------------------------------------------------//
