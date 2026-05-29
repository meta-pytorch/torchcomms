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

#include "timemory/macros/attributes.hpp"

#if !defined(DISABLE_TIMEMORY)
#    include "timemory/timemory.hpp"
#endif

#if defined(TIMEMORY_USE_MPI)
#    include <mpi.h>
#endif

#include <algorithm>
#include <atomic>
#include <functional>
#include <string>
#include <thread>

#if !defined(DISABLE_TIMEMORY)
using bundle_t = tim::component_tuple<tim::component::wall_clock>;
#endif

#if defined(ENABLE_MT)
#    define INSTRUMENTATION_TESTS_NAME instrumentation_mt_tests
#else
#    define INSTRUMENTATION_TESTS_NAME instrumentation_tests
#endif

//--------------------------------------------------------------------------------------//

auto&
get_rng()
{
    static std::mt19937 rng;
    return rng;
}
//
namespace details
{
//
void
setup_rng()
{
    get_rng().seed(std::random_device()());
}
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
inline auto
do_sleep(long n)
{
    TIMEMORY_BASIC_MARKER(bundle_t, "");
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
    return n;
}

// this function consumes approximately "t" milliseconds of cpu time
TIMEMORY_NOINLINE auto
consume(long n)
{
    TIMEMORY_BASIC_MARKER(bundle_t, "");
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
    return n;
}

// get a random entry from vector
template <typename Tp>
auto
random_entry(const std::vector<Tp>& v)
{
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(get_rng()));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class INSTRUMENTATION_TESTS_NAME : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(INSTRUMENTATION_TESTS_NAME, random_entry)
{
    std::vector<float> _v(100, 0.0);
    float              _i = 1.243f;
    std::generate(_v.begin(), _v.end(), [&_i]() { return _i * (_i + 1.43f); });
    auto _ret = details::random_entry(_v);

    EXPECT_TRUE(_ret > 0) << "ret = " << _ret;
}

//--------------------------------------------------------------------------------------//

TEST_F(INSTRUMENTATION_TESTS_NAME, consume)
{
    std::uniform_int_distribution<std::mt19937::result_type> dist(100, 1000);
    auto _ret = details::consume(dist(get_rng()));
    EXPECT_TRUE(_ret >= 100);
    EXPECT_TRUE(_ret <= 1000);

#if defined(TIMEMORY_USE_MPI)
    long _recv = 0;
    MPI_Allreduce(&_ret, &_recv, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_TRUE(_recv > 0) << "recv = " << _recv;
#endif
}

//--------------------------------------------------------------------------------------//

TEST_F(INSTRUMENTATION_TESTS_NAME, sleep)
{
    std::uniform_int_distribution<std::mt19937::result_type> dist(100, 1000);
    auto _ret = details::do_sleep(dist(get_rng()));
    EXPECT_TRUE(_ret >= 100);
    EXPECT_TRUE(_ret <= 1000);
}

//--------------------------------------------------------------------------------------//
#if defined(ENABLE_MT)
TEST_F(INSTRUMENTATION_TESTS_NAME, mt_consume)
{
    auto _consume = []() {
        std::uniform_int_distribution<std::mt19937::result_type> dist(100, 1000);
#    if !defined(DISABLE_TIMEMORY)
        auto _tid = tim::threading::get_id() + 1;
#    else
        std::atomic<int64_t> _tidc{};
        auto                 _tid = ++_tidc;
#    endif
        auto _rng = get_rng();
        _rng.seed(std::random_device()() * _tid * _tid);
        auto _ret = details::consume(dist(_rng));
        EXPECT_TRUE(_ret >= 100);
        EXPECT_TRUE(_ret <= 1000);
    };

    std::vector<std::thread> _threads{};
    _threads.reserve(4);
    for(int i = 0; i < 4; ++i)
        _threads.emplace_back(_consume);
    for(auto& itr : _threads)
        itr.join();
    _threads.clear();
}
#endif
//--------------------------------------------------------------------------------------//

static bool _setup = (details::setup_rng(), true);

//--------------------------------------------------------------------------------------//
