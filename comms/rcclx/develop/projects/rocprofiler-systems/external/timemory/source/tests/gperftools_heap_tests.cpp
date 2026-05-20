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

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;
using floating_t     = double;

static const int64_t niter     = 10;
static const int64_t page_size = tim::units::get_page_size();

using tuple_t =
    tim::component_tuple_t<wall_clock, gperftools_cpu_profiler, gperftools_heap_profiler>;
using list_t =
    tim::component_list_t<wall_clock, gperftools_cpu_profiler, gperftools_heap_profiler>;
using auto_tuple_t = typename tuple_t::auto_type;
using auto_list_t  = typename list_t::auto_type;

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

// this function ensures an allocation cannot be optimized
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

// this function consumes an unknown number of cpu resources
long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

void
allocate(int64_t nfactor)
{
    std::vector<int64_t> v(nfactor * page_size, 35);
    auto                 ret  = fibonacci(0);
    long                 nfib = details::random_entry(v);
    for(int64_t i = 0; i < niter; ++i)
    {
        nfib = random_entry(v);
        ret += fibonacci(nfib);
    }
    printf("fibonacci(%li) * %li = %li\n", (long) nfib, (long) niter, ret);
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class gperf_heap_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    void SetUp() override
    {
        TIMEMORY_TEST_CHECK_SKIP;
        setenv("MALLOCSTATS", "1", 1);
        setenv("CPUPROFILE_REALTIME", "1", 1);
        setenv("CPUPROFILE_FREQUENCY", "500", 1);
        list_t::get_initializer() = [](auto& obj) {
            obj.template initialize<wall_clock, gperftools_cpu_profiler,
                                    gperftools_heap_profiler>();
        };
    }
};

//--------------------------------------------------------------------------------------//

TEST_F(gperf_heap_tests, heap_profile)
{
    details::allocate(100);
    {
        TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name());
        details::allocate(10);
    }
    details::allocate(200);
    {
        TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name());
        details::allocate(50);
    }
    details::allocate(500);
}

//--------------------------------------------------------------------------------------//
