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

class chained_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

using namespace tim::component;
static long count_val = 0;

namespace details
{
//
template <typename BundleT = void>
long
fibonacci(long);
//
template <typename BundleT>
long
fibonacci(long n)
{
    return BundleT{ "fibonacci" }
        .start()
        .execute([n]() {
            return (n < 2) ? n : (fibonacci<BundleT>(n - 1) + fibonacci<BundleT>(n - 2));
        })
        .stop()
        .return_result();
}
//
template <>
long
fibonacci<void>(long n)
{
    return (n < 2) ? n : (fibonacci<void>(n - 1) + fibonacci<void>(n - 2));
}
//
template <>
long
fibonacci<long>(long n)
{
    ++count_val;
    return (n < 2) ? n : (fibonacci<long>(n - 1) + fibonacci<long>(n - 2));
}
//
}  // namespace details
//--------------------------------------------------------------------------------------//

TEST_F(chained_tests, fibonacci)
{
    using bundle_t     = tim::component_bundle<TIMEMORY_API, trip_count>;
    using lw_timer_t   = tim::lightweight_tuple<wall_clock, tim::quirk::auto_start>;
    using auto_timer_t = tim::auto_bundle<TIMEMORY_API, wall_clock>;

    tim::settings::precision() = 6;
    long nfib                  = 30;

    std::pair<lw_timer_t, long> count_tmp =
        lw_timer_t{ "count" }.start().execute(details::fibonacci<long>, nfib).stop();

    auto real_ret = lw_timer_t{ "real" }
                        .start()
                        .execute(details::fibonacci<void>, nfib)
                        .stop()
                        .get_bundle_and_result();

    auto test_ret = auto_timer_t{ "test" }
                        .execute(details::fibonacci<bundle_t>, nfib)
                        .stop()
                        .get_bundle_and_result();

    auto real_timer = real_ret.first;
    auto test_timer = test_ret.first;
    auto real_val   = real_ret.second;
    auto test_val   = test_ret.second;

    EXPECT_EQ(real_val, count_tmp.second);
    EXPECT_EQ(real_val, test_val);

    auto tc_data  = tim::storage<trip_count>::instance()->get();
    long meas_val = 0;
    long laps_val = 0;
    for(auto& itr : tc_data)
    {
        meas_val += itr.data().get();
        laps_val += itr.data().get_laps();
    }

    EXPECT_EQ(count_val, meas_val);
    EXPECT_EQ(count_val, laps_val);

    std::cout << real_timer << std::endl;
    std::cout << test_timer << std::endl;

    auto tc_storage = tim::storage<trip_count>::instance()->get();

    EXPECT_EQ(tc_storage.size(), nfib);
}

//--------------------------------------------------------------------------------------//
