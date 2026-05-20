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
#include "timemory/timemory.hpp"

using namespace tim::component;

//--------------------------------------------------------------------------------------//

namespace details
{
//  Get the current tests name
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

#if defined(TIMEMORY_MPI_GOTCHA)
extern "C" void
timemory_trace_set_mpi(bool use, bool attached);
#endif

class trace_tests : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
#if defined(TIMEMORY_MPI_GOTCHA)
        timemory_trace_set_mpi(true, false);
#endif
        timemory_trace_set_env("TIMEMORY_PRECISION", "6");
        timemory_trace_set_env("TIMEMORY_DART_OUTPUT", "ON");
        timemory_trace_set_env("TIMEMORY_DART_COUNT", "1");
        timemory_trace_init("peak_rss", true, test_argv[0]);
        tim::dmp::initialize(test_argc, test_argv);
        tim::timemory_init(test_argc, test_argv);
        timemory_push_trace("setup");
    }

    static void TearDownTestSuite()
    {
        timemory_pop_trace("setup");
        tim::dmp::finalize();
        timemory_trace_finalize();
    }

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

static std::vector<std::string> cxx_hash_ids = { "foo", "bar", "baz", "spam" };
static std::vector<char*>       _hash_ids    = {};
static std::vector<uint64_t>    _hash_nos    = {};

//--------------------------------------------------------------------------------------//

extern "C"
{
    extern void timemory_mpip_library_ctor();
    extern void timemory_register_mpip();
    extern void timemory_deregister_mpip();
    extern void timemory_ompt_library_ctor();
    extern void timemory_register_ompt();
    extern void timemory_deregister_ompt();
}

//--------------------------------------------------------------------------------------//

TEST_F(trace_tests, register)
{
    timemory_mpip_library_ctor();
    timemory_ompt_library_ctor();

    timemory_register_mpip();
    timemory_register_ompt();

    timemory_deregister_mpip();
    timemory_deregister_ompt();
}

//--------------------------------------------------------------------------------------//

TEST_F(trace_tests, add_hash_ids)
{
    auto wc_beg = tim::storage<wall_clock>::instance()->get();
    tim::component::user_trace_bundle::reset();
    tim::component::user_trace_bundle::configure<wall_clock>();

    EXPECT_EQ(tim::component::user_trace_bundle::bundle_size(), 1);

    for(auto& itr : cxx_hash_ids)
    {
        _hash_ids.push_back((char*) itr.c_str());
        _hash_nos.push_back(std::hash<std::string>{}(itr));
    }

    char** _ids = _hash_ids.data();
    timemory_add_hash_ids(cxx_hash_ids.size(), _hash_nos.data(), (const char**) _ids);

    auto d                 = tim::settings::debug();
    tim::settings::debug() = true;
    for(auto& itr : _hash_nos)
    {
        timemory_push_trace_hash(itr);
        details::consume(250);
        timemory_pop_trace_hash(itr);
    }
    tim::settings::debug() = d;

    auto wc_end = tim::storage<wall_clock>::instance()->get();
    auto npos   = std::string::npos;

    EXPECT_GE(wc_end.size(), 4);

    for(size_t i = 0; i < wc_end.size(); ++i)
    {
        EXPECT_FALSE(wc_end.at(i).prefix().find(cxx_hash_ids.at(i)) == npos)
            << "prefix: " << wc_end.at(i).prefix() << ", key: " << cxx_hash_ids.at(i)
            << std::endl;
    }

    EXPECT_NE(wc_beg.size(), wc_end.size());
    EXPECT_EQ(wc_end.size() - wc_beg.size(), 4);
}

//--------------------------------------------------------------------------------------//

TEST_F(trace_tests, modify_components)
{
    auto wc_beg = tim::storage<wall_clock>::instance()->get();
    auto cc_beg = tim::storage<cpu_clock>::instance()->get();

    tim::component::user_trace_bundle::reset();
    tim::component::user_trace_bundle::configure<cpu_clock>();

    auto   ncpu = tim::threading::affinity::hw_physicalcpu();
    auto   ndmp = tim::dmp::size();
    size_t nfac = 1;

    if(ndmp > ncpu)
        nfac = ceil(static_cast<double>(ndmp) / ncpu);

    EXPECT_GT(nfac, 0);

    auto t                          = tim::settings::throttle_value();
    auto v                          = tim::settings::verbose();
    tim::settings::throttle_value() = 5 * nfac * t;
    tim::settings::verbose()        = 2;
    for(size_t i = 0; i < 2 * tim::settings::throttle_count(); ++i)
    {
        for(auto& itr : _hash_nos)
        {
            timemory_push_trace_hash(itr);
            timemory_pop_trace_hash(itr);
        }
    }
    tim::settings::verbose()        = v;
    tim::settings::throttle_value() = t;

    for(auto& itr : cxx_hash_ids)
        EXPECT_TRUE(timemory_is_throttled(itr.c_str()));

    auto wc_end = tim::storage<wall_clock>::instance()->get();
    auto cc_end = tim::storage<cpu_clock>::instance()->get();
    auto npos   = std::string::npos;

    EXPECT_GE(wc_end.size(), 4);
    EXPECT_GE(cc_end.size(), 4);
    EXPECT_EQ(wc_end.size(), cc_end.size());

    for(size_t i = 0; i < wc_end.size(); ++i)
    {
        EXPECT_FALSE(wc_end.at(i).prefix().find(cxx_hash_ids.at(i)) == npos)
            << "prefix: " << wc_end.at(i).prefix() << ", key: " << cxx_hash_ids.at(i)
            << std::endl;
    }
    for(size_t i = 0; i < cc_end.size(); ++i)
    {
        EXPECT_FALSE(cc_end.at(i).prefix().find(cxx_hash_ids.at(i)) == npos)
            << "prefix: " << cc_end.at(i).prefix() << ", key: " << cxx_hash_ids.at(i)
            << std::endl;
    }

    EXPECT_EQ(wc_beg.size(), wc_end.size());
    EXPECT_NE(cc_beg.size(), cc_end.size());
    EXPECT_EQ(cc_end.size() - cc_beg.size(), 4);

    for(auto& itr : cxx_hash_ids)
        timemory_reset_throttle(itr.c_str());
}

//--------------------------------------------------------------------------------------//
