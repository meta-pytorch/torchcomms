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

#include "nccl_test/all_gather.h"
#include "nccl_test/all_reduce.h"
#include "nccl_test/alltoall.h"
#include "nccl_test/broadcast.h"
#include "nccl_test/reduce.h"
#include "nccl_test/reduce_scatter.h"
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

extern "C"
{
    extern void     timemory_register_ncclp();
    extern void     timemory_deregister_ncclp();
    extern uint64_t timemory_start_ncclp();
    extern uint64_t timemory_stop_ncclp(uint64_t);
}

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

template <typename Tp = tim::component::wall_clock>
auto
get_size()
{
    return tim::storage<Tp>::instance()->size();
}
}  // namespace details

//--------------------------------------------------------------------------------------//

namespace
{
int    _test_argc = 0;
char** _test_argv = nullptr;
}  // namespace

class nccl_tests : public ::testing::Test
{
protected:
    static void setup_args()
    {
        _test_argc = test_argc + 2;
        _test_argv = new char*[_test_argc];
        for(int i = 0; i < test_argc; ++i)
            _test_argv[i] = test_argv[i];
        _test_argv[test_argc + 0] = strdup("-c");
        _test_argv[test_argc + 1] = strdup("0");
    }

    static void cleanup_args()
    {
        free(_test_argv[test_argc + 0]);
        free(_test_argv[test_argc + 1]);
    }

    TIMEMORY_TEST_SUITE_SETUP(setup_args())
    TIMEMORY_TEST_SUITE_TEARDOWN(cleanup_args())

    virtual void SetUp() override
    {
        TIMEMORY_TEST_CHECK_SKIP;
        EXPECT_EQ(test_argc + 2, _test_argc);
        ASSERT_TRUE(_test_argv != nullptr);

        tim::set_env("TIMEMORY_NCCLP_REJECT_LIST",
                     "ncclGroupStart, ncclGroupEnd, ncclCommCuDevice, ncclCommUserRank",
                     0);
        m_idx = timemory_start_ncclp();
    }
    virtual void TearDown() override { timemory_stop_ncclp(m_idx); }

    static void ignore_warnings()
    {
        char fake[1];
        wordSize(ncclInt8);
        getHostHash("");
        getHostName(fake, 0);
        ncclstringtotype(fake);
        ncclstringtoop(fake);
    }

    uint64_t m_idx = 0;
};

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, all_gather)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(allGatherEngine, _test_argc, _test_argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, all_reduce)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(allReduceEngine, _test_argc, _test_argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, all_to_all)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(alltoAllEngine, _test_argc, _test_argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, broadcast)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(broadcastEngine, _test_argc, _test_argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, reduce)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(reduceEngine, _test_argc, _test_argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//

TEST_F(nccl_tests, reduce_scatter)
{
    auto beg_sz = details::get_size();
    int  ret    = test_main(reduceScatterEngine, _test_argc, _test_argv);
    auto end_sz = details::get_size();
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(end_sz - beg_sz, 1);
}

//--------------------------------------------------------------------------------------//
