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
#include "timemory/environment/types.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/backends/memory.hpp"
#include "timemory/backends/mpi.hpp"
#include "timemory/components/rusage/components.hpp"
#include "timemory/timemory.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/tpls/cereal/cereal/archives/json.hpp"
#include "timemory/tpls/cereal/cereal/details/helpers.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/signals.hpp"

#include "gtest/gtest.h"
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
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

class mpi_tests : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        tim::settings::verbose()     = 0;
        tim::settings::debug()       = false;
        tim::settings::json_output() = true;
        tim::settings::mpi_thread()  = false;
        tim::mpi::initialize(test_argc, test_argv);
        tim::timemory_init(test_argc, test_argv);
        tim::settings::dart_output()        = true;
        tim::settings::dart_count()         = 1;
        tim::settings::banner()             = false;
        tim::settings::collapse_processes() = false;
    }

    static void TearDownTestSuite()
    {
        tim::timemory_finalize();
        tim::dmp::finalize();
    }

    void SetUp() override
    {
        TIMEMORY_TEST_CHECK_SKIP;
        tim::mpi::barrier();
    }
    void TearDown() override { tim::mpi::barrier(); }
};

//--------------------------------------------------------------------------------------//

TEST_F(mpi_tests, general)
{
    using auto_types_t                = tim::convert_t<auto_tuple_t, tim::type_list<>>;
    tim::settings::collapse_threads() = true;
    auto manager                      = tim::manager::instance();
    // tim::manager::get_storage<auto_tuple_t>::clear(manager);
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
    TIMEMORY_CALIPER_APPLY0(master_thread_a, stop);

    {
        // run longer fibonacci calculations on two threads
        TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name(),
                              "/[master_thread]/1");
        run_fibonacci(40);
        run_fibonacci(41);
    }

    TIMEMORY_CALIPER_APPLY0(tot, stop);

    std::cout << "\nfibonacci total: " << ret.load() << "\n" << std::endl;

    auto rc_storage = tim::storage<wall_clock>::instance()->mpi_get();
    auto mpi_rank   = tim::mpi::rank();
    auto mpi_size   = tim::mpi::size();
    if(mpi_rank == 0)
    {
        EXPECT_EQ(rc_storage.size(), mpi_size);
        printf("\n");
        size_t w = 0;
        for(const auto& ritr : rc_storage)
            for(const auto& itr : ritr)
                w = std::max<size_t>(w, std::get<5>(itr).length());
        for(const auto& ritr : rc_storage)
            for(const auto& itr : ritr)
            {
                std::cout << std::setw(w) << std::left << std::get<5>(itr) << " : "
                          << std::get<7>(itr);
                auto _hierarchy = std::get<6>(itr);
                for(size_t i = 0; i < _hierarchy.size(); ++i)
                {
                    if(i == 0)
                        std::cout << " :: ";
                    std::cout << _hierarchy[i];
                    if(i + 1 < _hierarchy.size())
                        std::cout << "/";
                }
                std::cout << std::endl;
            }
        printf("\n");
    }
    else
    {
        EXPECT_EQ(rc_storage.size(), 1);
    }

    auto final_storage_size = tim::manager::get_storage<auto_types_t>::size(manager);
    auto expected           = (final_storage_size - starting_storage_size);

    EXPECT_EQ(expected, 15 * data_size);

    const size_t store_size =
        15 + (starting_storage_size / std::tuple_size<auto_tuple_t>::value);

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

TEST_F(mpi_tests, per_thread)
{
    using auto_types_t                = tim::convert_t<auto_tuple_t, tim::type_list<>>;
    tim::settings::collapse_threads() = false;
    auto manager                      = tim::manager::instance();
    // tim::manager::get_storage<auto_tuple_t>::clear(manager);
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
    {
        std::thread t_40(run_fibonacci, 40);
        std::thread t_41(run_fibonacci, 41);
        t_40.join();
        t_41.join();
    }
    TIMEMORY_CALIPER_APPLY0(master_thread_a, stop);

    {
        // run longer fibonacci calculations on two threads
        TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name(),
                              "/[master_thread]/1");
        std::thread t_40(run_fibonacci, 40);
        std::thread t_41(run_fibonacci, 41);
        t_40.join();
        t_41.join();
    }

    TIMEMORY_CALIPER_APPLY0(tot, stop);

    std::cout << "\nfibonacci total: " << ret.load() << "\n" << std::endl;

    auto rc_storage = tim::storage<wall_clock>::instance()->mpi_get();
    auto mpi_rank   = tim::mpi::rank();
    auto mpi_size   = tim::mpi::size();
    if(mpi_rank == 0)
    {
        EXPECT_EQ(rc_storage.size(), mpi_size);
        printf("\n");
        size_t w = 0;
        for(const auto& ritr : rc_storage)
            for(const auto& itr : ritr)
                w = std::max<size_t>(w, std::get<5>(itr).length());
        for(const auto& ritr : rc_storage)
            for(const auto& itr : ritr)
            {
                std::cout << std::setw(w) << std::left << std::get<5>(itr) << " : "
                          << std::get<7>(itr);
                auto _hierarchy = std::get<6>(itr);
                for(size_t i = 0; i < _hierarchy.size(); ++i)
                {
                    if(i == 0)
                        std::cout << " :: ";
                    std::cout << _hierarchy[i];
                    if(i + 1 < _hierarchy.size())
                        std::cout << "/";
                }
                std::cout << std::endl;
            }
        printf("\n");
    }
    else
    {
        EXPECT_EQ(rc_storage.size(), 1);
    }

    auto final_storage_size = tim::manager::get_storage<auto_types_t>::size(manager);
    auto expected           = (final_storage_size - starting_storage_size);

    EXPECT_EQ(expected, 19 * data_size);

    const size_t store_size =
        19 + (starting_storage_size / std::tuple_size<auto_tuple_t>::value);

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

TEST_F(mpi_tests, vector_get)
{
    auto_tuple_t tot{ details::get_test_name() };

    tot.start();

    long ret = 0;

    // run a fibonacci calculation and accumulate metric
    auto run_fibonacci = [&](long n) { ret += details::time_fibonacci(n, n - 2); };

    run_fibonacci(42);

    tot.stop();

    std::vector<wall_clock> wc_vec;
    std::vector<peak_rss>   pr_vec;

    tim::settings::verbose() = 2;

    tim::settings::collapse_processes() = true;
    tim::operation::finalize::mpi_get<wall_clock, true>{ wc_vec, *tot.get<wall_clock>() };

    tim::settings::collapse_processes() = false;
    tim::settings::node_count()         = 2;
    tim::operation::finalize::mpi_get<peak_rss, true>{ pr_vec, *tot.get<peak_rss>() };

    tim::settings::verbose() = 0;

    std::cout << "\nfibonacci total: " << ret << "\n" << std::endl;

    if(tim::mpi::rank() == 0)
    {
        std::cout << "WALL-CLOCK: " << std::endl;
        for(const auto& itr : wc_vec)
            std::cout << "    " << itr << std::endl;
        EXPECT_EQ(wc_vec.size(), 1);

        std::cout << "PEAK-RSS: " << std::endl;
        for(const auto& itr : pr_vec)
            std::cout << "    " << itr << std::endl;
        auto nc = tim::settings::node_count();
        if(nc > tim::mpi::size())
            nc = 1;
        EXPECT_EQ(pr_vec.size(), nc);
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(mpi_tests, send_recv_overflow)
{
    namespace comp   = tim::component;
    namespace mpi    = tim::mpi;
    using char_vec_t = std::vector<char>;
    using mpi_get_t  = tim::operation::finalize::mpi_get<char_vec_t, true>;

    const int64_t required_gb = 13 + (6 * (mpi::size() - 1));
    if(tim::memory::free_memory() < required_gb * tim::units::gigabyte)
    {
        std::cerr << "Skipping test " << details::get_test_name()
                  << " because the amount of free memory is less than " << required_gb
                  << " GB: " << (tim::memory::free_memory() / tim::units::megabyte)
                  << " MB (total memory: "
                  << (tim::memory::total_memory() / tim::units::megabyte) << " MB)\n";
        return;
    }
    else
    {
        std::cerr << "Executing test " << details::get_test_name()
                  << " because the amount of free memory exceeds " << required_gb
                  << " GB: " << (tim::memory::free_memory() / tim::units::megabyte)
                  << " MB (total memory: "
                  << (tim::memory::total_memory() / tim::units::megabyte) << " MB)\n";
    }

    comp::peak_rss _prss{};
    _prss.start();

    printf("[%s][%i] %-20s : %22s\n", details::get_test_name().c_str(), mpi::rank(),
           "Initial peak memory", TIMEMORY_JOIN("", _prss).c_str());

    tim::mpi::barrier();

    auto _size     = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
    auto _generate = [_size](int32_t _init) {
        char_vec_t _data{};
        _data.reserve(_size);
        uint8_t c = _init;
        for(size_t i = 0; i < _size; ++i, ++c)
        {
            _data.emplace_back(static_cast<char>(c));
        }
        return _data;
    };

    std::vector<char_vec_t> _rank_data = {};
    mpi_get_t{ false }(_rank_data, _generate(tim::mpi::rank()));

    _prss.stop();

    printf("[%s][%i] %-20s : %22s\n", details::get_test_name().c_str(), mpi::rank(),
           "Peak memory", TIMEMORY_JOIN("", _prss).c_str());

    if(mpi::rank() == 0)
    {
        auto _perc = [](size_t _num, size_t _denom) {
            return static_cast<double>(_num) / _denom * 100.;
        };

        for(size_t i = 0; i < _rank_data.size(); ++i)
        {
            auto _expected = _generate(i);
            ASSERT_EQ(_expected.size(), _rank_data.at(i).size());
            for(size_t j = 0; j < _size; ++j)
            {
                char _exp = _expected[j];
                char _dat = _rank_data[i][j];
                ASSERT_EQ(_exp, _dat)
                    << "rank: " << i << ", position: " << j
                    << ", expected value: " << (_expected[j])
                    << ", actual value: " << (_rank_data[i][j])
                    << ", total size: " << _size << ", % correct: " << _perc(j, _size)
                    << ", remaining: " << (_size - j);
            }
        }
    }
}

//--------------------------------------------------------------------------------------//
