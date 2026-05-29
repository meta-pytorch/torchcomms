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

#if defined(TIMEMORY_USE_MPI)
#    include <mpi.h>
#endif

#if !defined(TIMEMORY_TEST_NO_METRIC)
#    include "timemory/timemory.hpp"
#endif

#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;

namespace
{
#if !defined(DISABLE_TIMEMORY)
template <typename Tp>
inline std::string
as_string(const tim::node::result<Tp>& _obj)
{
    std::ostringstream _oss{};
    _oss << "tid=" << std::setw(2) << _obj.tid() << ", ";
    _oss << "pid=" << std::setw(6) << _obj.pid() << ", ";
    _oss << "depth=" << std::setw(2) << _obj.depth() << ", ";
    _oss << "hash=" << std::setw(21) << _obj.hash() << ", ";
    _oss << "prefix=" << _obj.prefix() << ", ";
    _oss << "data=" << _obj.data();
    return _oss.str();
}
#else
template <typename Tp>
inline std::string
as_string(const Tp&)
{
    return std::string{};
}
#endif
}  // namespace

#if !defined(DISABLE_TIMEMORY)
#    define TIMEMORY_TEST_ARGS                                                           \
        static int    test_argc = 0;                                                     \
        static char** test_argv = nullptr;

#    define TIMEMORY_TEST_MAIN                                                           \
        int main(int argc, char** argv)                                                  \
        {                                                                                \
            ::testing::InitGoogleTest(&argc, argv);                                      \
            test_argc = argc;                                                            \
            test_argv = argv;                                                            \
            return RUN_ALL_TESTS();                                                      \
        }

#    define TIMEMORY_TEST_DEFAULT_MAIN                                                   \
        TIMEMORY_TEST_ARGS                                                               \
        TIMEMORY_TEST_MAIN

#    define TIMEMORY_TEST_SUITE_SETUP(...)                                               \
    protected:                                                                           \
        static void SetUpTestSuite()                                                     \
        {                                                                                \
            puts("[SetupTestSuite] setup starting");                                     \
            tim::settings::verbose()     = 0;                                            \
            tim::settings::debug()       = false;                                        \
            tim::settings::json_output() = true;                                         \
            puts("[SetupTestSuite] initializing dmp");                                   \
            tim::dmp::initialize(test_argc, test_argv);                                  \
            puts("[SetupTestSuite] initializing timemory");                              \
            tim::timemory_init(test_argc, test_argv);                                    \
            puts("[SetupTestSuite] timemory initialized");                               \
            tim::settings::dart_output() = false;                                        \
            tim::settings::dart_count()  = 1;                                            \
            tim::settings::banner()      = false;                                        \
            __VA_ARGS__;                                                                 \
            puts("[SetupTestSuite] setup completed");                                    \
        }

#    define TIMEMORY_TEST_DEFAULT_SUITE_SETUP TIMEMORY_TEST_SUITE_SETUP({})

#    define TIMEMORY_TEST_SUITE_TEARDOWN(...)                                            \
    protected:                                                                           \
        static void TearDownTestSuite()                                                  \
        {                                                                                \
            __VA_ARGS__;                                                                 \
            tim::timemory_finalize();                                                    \
            if(tim::dmp::rank() == 0)                                                    \
                tim::signals::enable_signal_detection(                                   \
                    tim::signals::signal_settings::get_default());                       \
            tim::dmp::finalize();                                                        \
        }

#else
#    define TIMEMORY_TEST_ARGS
#    if defined(TIMEMORY_USE_MPI)
#        define TIMEMORY_TEST_MAIN                                                       \
            int main(int argc, char** argv)                                              \
            {                                                                            \
                MPI_Init(&argc, &argv);                                                  \
                ::testing::InitGoogleTest(&argc, argv);                                  \
                auto ret = RUN_ALL_TESTS();                                              \
                MPI_Finalize();                                                          \
                return ret;                                                              \
            }
#    else
#        define TIMEMORY_TEST_MAIN                                                       \
            int main(int argc, char** argv)                                              \
            {                                                                            \
                ::testing::InitGoogleTest(&argc, argv);                                  \
                return RUN_ALL_TESTS();                                                  \
            }
#    endif
#    define TIMEMORY_TEST_DEFAULT_MAIN TIMEMORY_TEST_MAIN
#    define TIMEMORY_TEST_SUITE_SETUP(...)
#    define TIMEMORY_TEST_DEFAULT_SUITE_SETUP
#    define TIMEMORY_TEST_SUITE_TEARDOWN(...)
#endif

#define TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN TIMEMORY_TEST_SUITE_TEARDOWN({})

#if !defined(DISABLE_TIMEMORY)
#    define TIMEMORY_TEST_CHECK_SKIP                                                     \
        if(tim::get_env<std::string>("TIMEMORY_TEST_SKIP", "")                           \
               .find(details::get_test_name()) != std::string::npos)                     \
        {                                                                                \
            GTEST_SKIP();                                                                \
        }
#else
#    define TIMEMORY_TEST_CHECK_SKIP
#endif

#define TIMEMORY_TEST_SETUP(...)                                                         \
protected:                                                                               \
    void SetUp() override                                                                \
    {                                                                                    \
        TIMEMORY_TEST_CHECK_SKIP;                                                        \
        printf("[##########] Executing %s ... \n", details::get_test_name().c_str());    \
        __VA_ARGS__;                                                                     \
    }

#define TIMEMORY_TEST_DEFAULT_SETUP TIMEMORY_TEST_SETUP({})

#define TIMEMORY_TEST_TEARDOWN(...)                                                      \
protected:                                                                               \
    void TearDown() override { __VA_ARGS__; }

#define TIMEMORY_TEST_DEFAULT_TEARDOWN TIMEMORY_TEST_TEARDOWN({})

#define TIMEMORY_TEST_DEFAULT_SUITE_BODY                                                 \
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP                                                    \
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN                                                 \
    TIMEMORY_TEST_DEFAULT_SETUP                                                          \
    TIMEMORY_TEST_DEFAULT_TEARDOWN
