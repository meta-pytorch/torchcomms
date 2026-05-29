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

#include "timemory/mpl/apply.hpp"
#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

static const double epsilon = 1.1 * std::numeric_limits<double>::epsilon();

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

class apply_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(apply_tests, set_value)
{
    const double           value = 10.0;
    std::array<double, 12> arr;
    tim::mpl::apply<void>::set_value(arr, value);
    for(auto& itr : arr)
    {
        auto diff = itr - value;
        printf("itr value = %16.12f (expected = %16.12f); diff = %16.12e\n", itr, value,
               diff);
        ASSERT_NEAR(itr, value, epsilon);
    }
}

//--------------------------------------------------------------------------------------//
// declare a component and set it to always off
TIMEMORY_DECLARE_COMPONENT(always_off)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::always_off, false_type)
//
TEST_F(apply_tests, traits)
{
    using namespace tim::component;
    tim::trait::apply<tim::trait::runtime_enabled>::set<
        wall_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss>(false);
    tim::trait::apply<tim::trait::runtime_enabled>::set<user_clock, peak_rss, always_off>(
        true);

    enum
    {
        WallClockIdx = 0,
        UserClockIdx,
        SystemClockIdx,
        CpuClockIdx,
        CpuUtilIdx,
        PeakRssIdx,
        AlwaysOffIdx
    };

    auto check = tim::trait::apply<tim::trait::runtime_enabled>::get<
        wall_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss,
        always_off>();

    EXPECT_FALSE(std::get<WallClockIdx>(check));
    EXPECT_TRUE(std::get<UserClockIdx>(check));
    EXPECT_FALSE(std::get<SystemClockIdx>(check));
    EXPECT_FALSE(std::get<CpuClockIdx>(check));
    EXPECT_FALSE(std::get<CpuUtilIdx>(check));
    EXPECT_TRUE(std::get<PeakRssIdx>(check));
    EXPECT_FALSE(std::get<AlwaysOffIdx>(check));
}

//--------------------------------------------------------------------------------------//
