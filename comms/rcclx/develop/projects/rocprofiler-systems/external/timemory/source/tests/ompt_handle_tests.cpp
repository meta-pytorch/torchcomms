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

#include "timemory/library.h"

#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
#include <thread>
#include <vector>

extern "C"
{
    extern uint64_t timemory_start_ompt();
    extern uint64_t timemory_stop_ompt(uint64_t id);
    extern void     timemory_register_ompt();
    extern void     timemory_deregister_ompt();
}

static const double pi_epsilon =
    static_cast<double>(std::numeric_limits<float>::epsilon());
static const long nfib = 39;
static const long noff = 4;

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

double
compute_pi(uint64_t nstart, uint64_t nstop, double step, uint64_t nblock)
{
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    if(nstop - nstart < nblock)
    {
        for(uint64_t i = nstart; i < nstop; ++i)
        {
            double x = (i + 0.5) * step;
            sum      = sum + 4.0 / (1.0 + x * x);
        }
    }
    else
    {
        uint64_t iblk = nstop - nstart;
#pragma omp task shared(sum1)
        sum1 = compute_pi(nstart, nstop - iblk / 2, step, nblock);
#pragma omp task shared(sum2)
        sum2 = compute_pi(nstop - iblk / 2, nstop, step, nblock);
#pragma omp taskwait
        sum = sum1 + sum2;
    }
    return sum;
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class ompt_handle_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

static std::atomic<int> thread_count;
int&
get_tid()
{
    static thread_local int _id = thread_count++;
    return _id;
}

//--------------------------------------------------------------------------------------//

TEST_F(ompt_handle_tests, off)
{
    omp_set_num_threads(4);

    std::atomic<uint64_t> sum(0);
    {
#pragma omp parallel for
        for(int i = 0; i < 4; i++)
        {
            sum += details::fibonacci(nfib);
            details::do_sleep(500);
        }
    }

    {
#pragma omp parallel for
        for(int i = 0; i < 4; i++)
        {
            sum += details::fibonacci(nfib + noff);
            details::do_sleep(500);
        }
    }

    printf("[%s]> sum: %lu\n", details::get_test_name().c_str(),
           static_cast<unsigned long>(sum));
}

//--------------------------------------------------------------------------------------//

TEST_F(ompt_handle_tests, registration)
{
    timemory_register_ompt();

    omp_set_num_threads(4);

    std::atomic<uint64_t> sum(0);
    {
#pragma omp parallel for
        for(int i = 0; i < 4; i++)
        {
            sum += details::fibonacci(nfib);
        }
    }

    {
#pragma omp parallel for
        for(int i = 0; i < 4; i++)
        {
            sum += details::fibonacci(nfib + noff);
        }
    }

    printf("[%s]> sum: %lu\n", details::get_test_name().c_str(),
           static_cast<unsigned long>(sum));

    timemory_deregister_ompt();
}

//--------------------------------------------------------------------------------------//

TEST_F(ompt_handle_tests, init)
{
    auto idx = timemory_start_ompt();

    uint64_t num_threads = 4;
    uint64_t num_steps   = 500000000;

    omp_set_num_threads(num_threads);

    double sum    = 0.0;
    double step   = 1.0 / static_cast<double>(num_steps);
    auto   nblock = num_steps / num_threads;

#pragma omp parallel
    {
#pragma omp single
        {
            sum = details::compute_pi(0, num_steps, step, nblock);
        }
    }

    double pi = step * sum;
    printf("[%s]> pi: %f\n", details::get_test_name().c_str(), pi);

    ASSERT_NEAR(pi, M_PI, pi_epsilon);

    EXPECT_EQ(idx, 1);
    idx = timemory_stop_ompt(idx);
    EXPECT_EQ(idx, 0);
}

//--------------------------------------------------------------------------------------//
