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

#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

static int    test_argc = 0;
static char** test_argv = nullptr;

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

class socket_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(!configured)
        {
            configured                   = true;
            tim::settings::verbose()     = 0;
            tim::settings::debug()       = false;
            tim::settings::json_output() = true;
            tim::settings::mpi_thread()  = false;
            tim::dmp::initialize(test_argc, test_argv);
            tim::timemory_init(test_argc, test_argv);
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;
        }
    }

public:
    static bool configured;
};

bool socket_tests::configured = false;

//--------------------------------------------------------------------------------------//

TEST_F(socket_tests, server_client)
{
    auto nitr = tim::get_env<size_t>("SOCKET_TESTS_NUM_ITER", 100);

    tim::lightweight_tuple<tim::component::network_stats> _net{
        details::get_test_name()
    };

    _net.push();
    tim::socket::manager socket_manager{};
    for(size_t i = 0; i < nitr; ++i)
    {
        _net.start();
        std::cout << std::endl;
        // use promises and futures to emulate the server-client relationship
        std::promise<void> serv_beg;
        std::promise<void> serv_end;

        std::vector<std::string> results;
        auto                     server_func = [&]() {
            serv_end.set_value_at_thread_exit();
            auto handle_data = [&](std::string str) {
                std::cout << "[server]> received: " << str << std::endl;
                results.emplace_back(std::move(str));
            };
            std::cout << "[server]> started listening..." << std::endl;
            serv_beg.set_value();
            tim::socket::manager{}.listen("test", 8080 + i, handle_data);
            std::cout << "[server]> stopped listening..." << std::endl;
        };

        std::thread _server(server_func);
        _server.detach();

        serv_beg.get_future().wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::cout << "[client]> connect..." << std::endl;
        socket_manager.connect("test", "127.0.0.1", 8080 + i);
        for(auto&& itr : { "hello", "world" })
        {
            auto orig = results.size();
            socket_manager.send("test", std::string(itr));
            while(orig == results.size())
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        socket_manager.close("test");
        std::cout << "[client]> closed..." << std::endl;

        serv_end.get_future().wait();
        _net.stop();

        EXPECT_EQ(results.size(), 2);
        EXPECT_EQ(results.at(0), "hello");
        EXPECT_EQ(results.at(1), "world");
    }
    _net.pop();
    std::cout << std::endl;
    auto* _netv = _net.get<tim::component::network_stats>();
    if(_netv)
    {
        std::cout << _net << std::endl;
        auto    _data = _netv->load();
        int64_t _sum  = 0;
        for(auto& itr : _data.get_data())
            _sum += itr;
        EXPECT_GT(_sum, 500) << *_netv;
    }
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    test_argc = argc;
    test_argv = argv;

    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();
    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
