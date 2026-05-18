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
#include "timemory/api/kokkosp.hpp"
#include "timemory/timemory.hpp"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

TIMEMORY_TEST_DEFAULT_MAIN

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

class kokkosp_tests : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        tim::settings::memory_units() = "kb";
        kokkosp_init_library(0, 0, 0, nullptr);
        kokkosp_parse_args(test_argc, test_argv);
        kokkosp_declare_metadata(
            "test_suite",
            ::testing::UnitTest::GetInstance()->current_test_suite()->name());
        tim::settings::dart_output() = true;
        tim::settings::dart_count()  = 1;
    }

    static void TearDownTestSuite() { kokkosp_finalize_library(); }
};

//--------------------------------------------------------------------------------------//

TEST_F(kokkosp_tests, profiling_routines)
{
    auto _beg_sz = tim::storage<tim::component::wall_clock>::instance()->size();

    std::array<uint64_t, 4> idx;
    uint32_t                sec_idx = 0;
    idx.fill(0);

    kokkosp_profile_event(details::get_test_name().c_str());

    kokkosp_begin_parallel_for("parallel_for", 0, &idx.at(0));

    details::consume(100);

    kokkosp_begin_parallel_reduce("parallel_reduce", 0, &idx.at(1));

    details::consume(100);

    kokkosp_begin_parallel_scan("parallel_scan", 0, &idx.at(2));

    details::consume(100);

    kokkosp_begin_fence("fence", 0, &idx.at(3));

    details::consume(100);

    kokkosp_push_profile_region("profile_region");

    details::consume(100);

    kokkosp_create_profile_section("profile_section", &sec_idx);
    kokkosp_start_profile_section(sec_idx);

    details::consume(100);

    kokkosp_stop_profile_section(sec_idx);
    kokkosp_destroy_profile_section(sec_idx);

    details::consume(100);

    kokkosp_pop_profile_region();

    details::consume(100);

    kokkosp_end_fence(idx.at(3));

    details::consume(100);

    kokkosp_end_parallel_scan(idx.at(2));

    details::consume(100);

    kokkosp_end_parallel_reduce(idx.at(1));

    details::consume(100);

    kokkosp_end_parallel_for(idx.at(0));

    auto _end_sz = tim::storage<tim::component::wall_clock>::instance()->size();

    EXPECT_EQ(_end_sz - _beg_sz, 6)
        << " begin size: " << _beg_sz << ", end size: " << _end_sz;
}

//--------------------------------------------------------------------------------------//

TEST_F(kokkosp_tests, data_routines)
{
    size_t _sz = 250000;

    {
        SpaceHandle _handle = { "HOST\0" };
        auto*       _data   = new double[_sz];
        kokkosp_allocate_data(_handle, "allocate_data", (void*) _data,
                              _sz * sizeof(double));
        details::do_sleep(100);
        kokkosp_deallocate_data(_handle, "allocate_data", (void*) _data,
                                _sz * sizeof(double));
        delete[] _data;
    }

    _sz *= 2;

    {
        SpaceHandle _src_handle = { "HOST\0" };
        SpaceHandle _dst_handle = { "HOST\0" };

        auto* _src = new double[_sz];
        auto* _dst = new double[_sz];
        for(size_t i = 0; i < _sz; ++i)
            _src[i] = static_cast<double>(i) + 1;

        kokkosp_begin_deep_copy(_dst_handle, "target_data", (void*) _dst, _src_handle,
                                "source_data", (void*) _src, _sz * sizeof(double));

        memcpy(_dst, _src, _sz * sizeof(double));

        kokkosp_end_deep_copy();

        delete[] _dst;
        delete[] _src;
    }
}

//--------------------------------------------------------------------------------------//
