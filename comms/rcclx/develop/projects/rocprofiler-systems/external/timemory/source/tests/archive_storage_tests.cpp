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

using namespace tim::component;

using comp_bundle_t =
    tim::component_bundle_t<TIMEMORY_API, wall_clock, cpu_clock, current_peak_rss>;
using auto_bundle_t = tim::convert_t<comp_bundle_t, tim::auto_bundle<TIMEMORY_API>>;

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
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

void
generate_history(int n, int m)
{
    {
        std::stringstream ss;
        ss << "Generating history on thread: " << std::this_thread::get_id() << std::endl;
        std::cout << ss.str() << std::flush;
    }
    for(int j = 0; j < n; ++j)
    {
        TIMEMORY_MARKER(auto_bundle_t, "outer-loop-", j % 2);
        consume(250);
        for(int i = 0; i < m; ++i)
        {
            TIMEMORY_MARKER(auto_bundle_t, "inner-loop-", i);
            fibonacci(38 + (i % 2));
            {
                TIMEMORY_BASIC_MARKER(auto_bundle_t, "do-sleep");
                do_sleep(50 * (i + 1));
            }
            {
                TIMEMORY_BASIC_MARKER(auto_bundle_t, "consume");
                consume(50 * (i + 1));
            }
        }
    }
    {
        std::stringstream ss;
        ss << "History generated on thread: " << std::this_thread::get_id() << std::endl;
        std::cout << ss.str() << std::flush;
    }
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class archive_storage_tests : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        tim::settings::verbose()           = 0;
        tim::settings::debug()             = false;
        tim::settings::mpi_thread()        = false;
        tim::settings::cout_output()       = false;
        tim::settings::text_output()       = true;
        tim::settings::flamegraph_output() = false;
        tim::dmp::initialize(test_argc, test_argv);
        tim::timemory_init(test_argc, test_argv);
        tim::settings::dart_output() = true;
        tim::settings::dart_count()  = 1;
        tim::settings::banner()      = false;

        std::vector<std::thread> threads;
        for(uint64_t i = 0; i < 2; ++i)
            threads.emplace_back(std::thread(details::generate_history, 5, 2));
        for(auto& itr : threads)
            itr.join();
        // std::cout << "Configured" << std::endl;
    }

    static void TearDownTestSuite()
    {
        tim::timemory_finalize();
        tim::dmp::finalize();
        // std::cout << "Finalized" << std::endl;
    }

    static std::map<std::string, std::string> f_results;
};

//--------------------------------------------------------------------------------------//

std::map<std::string, std::string> archive_storage_tests::f_results = {};

//--------------------------------------------------------------------------------------//

TEST_F(archive_storage_tests, vector_hierarchy)
{
    auto wc_storage = tim::storage<wall_clock>::instance();
    auto cc_storage = tim::storage<cpu_clock>::instance();

    using wc_get_t        = tim::operation::finalize::get<wall_clock, true>;
    using cc_get_t        = tim::operation::finalize::get<cpu_clock, true>;
    using wc_basic_tree_t = typename wc_get_t::basic_tree_type;
    using cc_basic_tree_t = typename cc_get_t::basic_tree_type;

    std::vector<wc_basic_tree_t> wc_vec;
    std::vector<cc_basic_tree_t> cc_vec;

    wc_storage->get(wc_vec);
    cc_storage->get(cc_vec);

    auto print = [&](const auto& arr) {
        std::stringstream ss;
        ss << "  [size: " << arr.size() << "]\n";
        for(const auto& itr : arr)
        {
            ss << "  [" << tim::get_hash_identifier(itr->get_value().hash()) << "]["
               << itr->get_value().inclusive().stats() << "]\n";
        }
        return ss.str();
    };
    EXPECT_EQ(wc_vec.size(), 1);
    EXPECT_EQ(cc_vec.size(), 1);
    EXPECT_EQ(wc_vec.front().get_children().size(), 2)
        << print(wc_vec.front().get_children());
    EXPECT_EQ(cc_vec.front().get_children().size(), 2)
        << print(cc_vec.front().get_children());
    EXPECT_EQ(wc_vec.front().get_children().front()->get_children().size(), 2)
        << print(wc_vec.front().get_children().front()->get_children());
    EXPECT_EQ(cc_vec.front().get_children().front()->get_children().size(), 2)
        << print(cc_vec.front().get_children().front()->get_children());
}

//--------------------------------------------------------------------------------------//

TEST_F(archive_storage_tests, archive_hierarchy)
{
    if(tim::dmp::rank() == 0)
    {
        using archive_t              = tim::cereal::JSONOutputArchive;
        using api_t                  = TIMEMORY_API;
        auto              wc_storage = tim::storage<wall_clock>::instance();
        auto              cc_storage = tim::storage<cpu_clock>::instance();
        std::stringstream ss;
        {
            auto ar = tim::policy::output_archive<archive_t, api_t>::get(ss);
            wc_storage->get(*ar);
            cc_storage->get(*ar);
        }
        f_results[details::get_test_name()] = ss.str();
        // std::cout << ss.str() << std::endl;
        auto fname =
            tim::settings::compose_output_filename(details::get_test_name(), "json");
        std::ofstream ofs{};
        if(tim::filepath::open(ofs, fname))
            ofs << ss.str() << std::endl;
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(archive_storage_tests, mpi_archive_hierarchy)
{
    using archive_t              = tim::cereal::JSONOutputArchive;
    using api_t                  = TIMEMORY_API;
    auto              wc_storage = tim::storage<wall_clock>::instance();
    auto              cc_storage = tim::storage<cpu_clock>::instance();
    std::stringstream ss;
    {
        auto ar = tim::policy::output_archive<archive_t, api_t>::get(ss);
        wc_storage->mpi_get(*ar);
        cc_storage->mpi_get(*ar);
    }
    f_results[details::get_test_name()] = ss.str();
    if(tim::dmp::rank() > 0)
        return;
    auto fname = tim::settings::compose_output_filename(details::get_test_name(), "json");
    std::ofstream ofs{};
    if(tim::filepath::open(ofs, fname))
    {
        ofs << ss.str() << std::endl;
    }
    else
    {
        std::cout << ss.str() << std::endl;
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(archive_storage_tests, upc_archive_hierarchy)
{
    using archive_t              = tim::cereal::JSONOutputArchive;
    using api_t                  = TIMEMORY_API;
    auto              wc_storage = tim::storage<wall_clock>::instance();
    auto              cc_storage = tim::storage<cpu_clock>::instance();
    std::stringstream ss;
    {
        auto ar = tim::policy::output_archive<archive_t, api_t>::get(ss);
        wc_storage->upc_get(*ar);
        cc_storage->upc_get(*ar);
    }
    f_results[details::get_test_name()] = ss.str();
    if(tim::dmp::rank() > 0)
        return;
    auto fname = tim::settings::compose_output_filename(details::get_test_name(), "json");
    std::ofstream ofs{};
    if(tim::filepath::open(ofs, fname))
    {
        ofs << ss.str() << std::endl;
    }
    else
    {
        std::cout << ss.str() << std::endl;
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(archive_storage_tests, dmp_archive_hierarchy)
{
    using archive_t              = tim::cereal::JSONOutputArchive;
    using api_t                  = TIMEMORY_API;
    auto              wc_storage = tim::storage<wall_clock>::instance();
    auto              cc_storage = tim::storage<cpu_clock>::instance();
    std::stringstream ss;
    {
        auto ar = tim::policy::output_archive<archive_t, api_t>::get(ss);
        wc_storage->dmp_get(*ar);
        cc_storage->dmp_get(*ar);
    }
    f_results[details::get_test_name()] = ss.str();
    if(tim::dmp::rank() > 0)
        return;
    auto fname = tim::settings::compose_output_filename(details::get_test_name(), "json");
    std::ofstream ofs{};
    if(tim::filepath::open(ofs, fname))
    {
        ofs << ss.str() << std::endl;
    }
    else
    {
        std::cout << ss.str() << std::endl;
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(archive_storage_tests, check_archive)
{
    if(tim::dmp::rank() > 0)
        return;

    if(!tim::mpi::is_initialized() && !tim::upc::is_initialized())
    {
        EXPECT_EQ(f_results["archive_hierarchy"], f_results["mpi_archive_hierarchy"]);
        EXPECT_EQ(f_results["archive_hierarchy"], f_results["upc_archive_hierarchy"]);
        EXPECT_EQ(f_results["archive_hierarchy"], f_results["dmp_archive_hierarchy"]);
    }
    else if(tim::mpi::is_initialized() && !tim::upc::is_initialized())
    {
        EXPECT_NE(f_results["archive_hierarchy"], f_results["mpi_archive_hierarchy"]);
        EXPECT_EQ(f_results["archive_hierarchy"], f_results["upc_archive_hierarchy"]);
        EXPECT_NE(f_results["archive_hierarchy"], f_results["dmp_archive_hierarchy"]);
        EXPECT_EQ(f_results["mpi_archive_hierarchy"], f_results["dmp_archive_hierarchy"]);
    }
    else if(!tim::mpi::is_initialized() && tim::upc::is_initialized())
    {
        EXPECT_EQ(f_results["archive_hierarchy"], f_results["mpi_archive_hierarchy"]);
        EXPECT_NE(f_results["archive_hierarchy"], f_results["upc_archive_hierarchy"]);
        EXPECT_NE(f_results["archive_hierarchy"], f_results["dmp_archive_hierarchy"]);
        EXPECT_EQ(f_results["upc_archive_hierarchy"], f_results["dmp_archive_hierarchy"]);
    }
    else if(tim::mpi::is_initialized() && tim::upc::is_initialized())
    {
        EXPECT_NE(f_results["archive_hierarchy"], f_results["mpi_archive_hierarchy"]);
        EXPECT_NE(f_results["archive_hierarchy"], f_results["upc_archive_hierarchy"]);
        EXPECT_NE(f_results["archive_hierarchy"], f_results["dmp_archive_hierarchy"]);
        EXPECT_NE(f_results["mpi_archive_hierarchy"], f_results["dmp_archive_hierarchy"]);
        EXPECT_NE(f_results["upc_archive_hierarchy"], f_results["dmp_archive_hierarchy"]);
    }
}

//--------------------------------------------------------------------------------------//

// ensure the storage is initialized on the master thread
TIMEMORY_INITIALIZE_STORAGE(wall_clock, cpu_clock, current_peak_rss)
