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
#include "timemory/utility/signals.hpp"

#include "gtest/gtest.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace tim::stl;
using namespace tim::component;

using papi_tuple_t = papi_tuple<PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_LST_INS>;

using auto_tuple_t =
    tim::auto_tuple_t<wall_clock, thread_cpu_clock, thread_cpu_util, process_cpu_clock,
                      process_cpu_util, peak_rss, page_rss>;

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
//--------------------------------------------------------------------------------------//
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
//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n)
{
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, n);
        return (n < 2) ? n : fibonacci(n - 1, cutoff) + fibonacci(n - 2, cutoff);
    }
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
//--------------------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
int64_t
time_fibonacci(int32_t n)
{
    TIMEMORY_MARKER(auto_tuple_t, "");
    return fibonacci(n);
}
//--------------------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
int64_t
time_fibonacci(int32_t n, int32_t cutoff)
{
    TIMEMORY_MARKER(auto_tuple_t, "");
    return fibonacci(n, cutoff);
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class upcxx_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(upcxx_tests, general)
{
    using auto_types_t                = tim::convert_t<auto_tuple_t, tim::type_list<>>;
    tim::settings::collapse_threads() = true;
    auto manager                      = tim::manager::instance();
    tim::manager::get_storage<auto_types_t>::clear(manager);
    auto starting_storage_size = tim::manager::get_storage<auto_types_t>::size(manager);
    auto data_size             = auto_tuple_t::size();
    std::atomic<long> ret;

    // accumulate metrics on full run
    TIMEMORY_BLANK_CALIPER(tot, auto_tuple_t, details::get_test_name(), "/[total]");

    // run a fibonacci calculation and accumulate metric
    auto run_fibonacci = [&](long n) {
        TIMEMORY_BLANK_MARKER(auto_tuple_t, "run_fibonacci");
        ret += details::time_fibonacci(n, n - 2);
    };

    // run longer fibonacci calculations on two threads
    TIMEMORY_BLANK_CALIPER(master_thread_a, auto_tuple_t, details::get_test_name(),
                           "/[master_thread]/0");
    run_fibonacci(40);
    run_fibonacci(41);
    TIMEMORY_CALIPER_APPLY(master_thread_a, stop);

    {
        // run longer fibonacci calculations on two threads
        TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name(),
                              "/[master_thread]/1");
        run_fibonacci(40);
        run_fibonacci(41);
    }

    // stop the total
    TIMEMORY_CALIPER_APPLY(tot, stop);

    std::cout << "\nfibonacci total: " << ret.load() << "\n" << std::endl;

    auto rc_storage = tim::storage<wall_clock>::instance()->upc_get();
    auto upc_rank   = tim::upc::rank();
    auto upc_size   = tim::upc::size();

    // auto rc_print = [=](const decltype(rc_storage)& _storage) {
    //     printf("\n");
    //     size_t w = 0;
    //     for(const auto& ritr : _storage)
    //         for(const auto& itr : ritr)
    //             w = std::max<size_t>(w, std::get<2>(itr).length());
    //     int64_t idx = 0;
    //     for(const auto& ritr : _storage)
    //     {
    //         printf("[%i]> idx: %i, size: %i\n", (int) upc_rank, (int) idx,
    //                (int) ritr.size());
    //         for(const auto& itr : ritr)
    //         {
    //             std::cout << std::setw(w) << std::left << std::get<2>(itr) << " : "
    //                       << std::get<1>(itr);
    //             auto _hierarchy = std::get<5>(itr);
    //             for(size_t i = 0; i < _hierarchy.size(); ++i)
    //             {
    //                 if(i == 0)
    //                     std::cout << " :: ";
    //                 std::cout << _hierarchy[i];
    //                 if(i + 1 < _hierarchy.size())
    //                     std::cout << "/";
    //             }
    //             std::cout << std::endl;
    //         }
    //         ++idx;
    //     }
    //     printf("\n");
    // };

    if(upc_rank == 0)
    {
        std::this_thread::sleep_for(std::chrono::seconds(upc_size - upc_rank - 1));
        EXPECT_EQ(rc_storage.size(), upc_size);
        // rc_print(rc_storage);
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::seconds(upc_size - upc_rank - 1));
        EXPECT_EQ(rc_storage.size(), 1);
        // rc_print(rc_storage);
    }

    auto final_storage_size = tim::manager::get_storage<auto_types_t>::size(manager);
    auto expected           = (final_storage_size - starting_storage_size);

    EXPECT_EQ(expected, 15 * data_size);

    const size_t store_size = 15;

    if(tim::trait::is_available<wall_clock>::value)
    {
        EXPECT_EQ(tim::storage<wall_clock>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<thread_cpu_clock>::value)
    {
        EXPECT_EQ(tim::storage<thread_cpu_clock>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<thread_cpu_util>::value)
    {
        EXPECT_EQ(tim::storage<thread_cpu_util>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<process_cpu_clock>::value)
    {
        EXPECT_EQ(tim::storage<process_cpu_clock>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<process_cpu_util>::value)
    {
        EXPECT_EQ(tim::storage<process_cpu_util>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<peak_rss>::value)
    {
        EXPECT_EQ(tim::storage<peak_rss>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<page_rss>::value)
    {
        EXPECT_EQ(tim::storage<page_rss>::instance()->get().size(), store_size);
    }
}

//--------------------------------------------------------------------------------------//
