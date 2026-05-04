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
// clang-format off
#include "gtest/gtest.h"

#include "timemory/timemory.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using namespace tim::quirk;

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using comp_lw_tuple_t = tim::lightweight_tuple<wall_clock, cpu_clock, cpu_util>;
using auto_lw_tuple_t =
    tim::lightweight_tuple<wall_clock, cpu_clock, cpu_util, auto_start>;

using auto_tuple_t = typename comp_lw_tuple_t::auto_type;
using comp_tuple_t = typename auto_lw_tuple_t::component_type;

//--------------------------------------------------------------------------------------//
//
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
void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // wait time
    auto wait = std::chrono::milliseconds(n);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + wait))
    {
    }
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class macro_tests : public ::testing::Test
{};

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, blank_marker)
{
    TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name());
    details::do_sleep(25);
    details::consume(75);
    timemory_variable_101.stop();
    auto              key = timemory_variable_101.key();
    std::stringstream expected;
    expected << details::get_test_name();
    if(key != expected.str())
        FAIL() << "expected key: \"" << expected.str() << "\" vs. actual key: \"" << key;
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, basic_marker)
{
    TIMEMORY_BASIC_MARKER(auto_tuple_t, details::get_test_name());
    details::do_sleep(25);
    details::consume(75);
    timemory_variable_118.stop();
    auto              key = timemory_variable_118.key();
    std::stringstream expected;
    expected << __FUNCTION__ << "/" << details::get_test_name();
    if(key != expected.str())
        FAIL() << "expected key: \"" << expected.str() << "\" vs. actual key: \"" << key;
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, marker)
{
    TIMEMORY_MARKER(auto_tuple_t, details::get_test_name());
    auto line = __LINE__ - 1;
    details::do_sleep(25);
    details::consume(75);
    timemory_variable_135.stop();
    auto              key = timemory_variable_135.key();
    std::stringstream expected;
    std::string       file = __FILE__;
    file = file.substr(file.find_last_of(TIMEMORY_OS_PATH_DELIMITER) + 1);
    expected << __FUNCTION__ << "@" << file << ":" << line << "/"
             << details::get_test_name();
    if(key != expected.str())
        FAIL() << "expected key: \"" << expected.str() << "\" vs. actual key: \"" << key;
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, marker_auto_type)
{
    TIMEMORY_BLANK_MARKER(comp_tuple_t, details::get_test_name());
    details::do_sleep(250);
    details::consume(750);
    timemory_variable_156.stop();
    auto              key = timemory_variable_156.key();
    std::stringstream expected;
    expected << details::get_test_name();

    using check_tuple_t = std::decay_t<decltype(timemory_variable_156)>;

    std::stringstream same;
    same << "\n";
    same << "comp_tuple  :: " << tim::demangle<comp_tuple_t>() << "\n";
    same << "auto_tuple  :: " << tim::demangle<auto_tuple_t>() << "\n";
    same << "check_tuple :: " << tim::demangle<check_tuple_t>() << "\n";
    same << "\n";
    same << timemory_variable_156 << std::endl;
    same << "\n";

    auto auto_check = std::is_same<auto_tuple_t, check_tuple_t>::value;
    auto comp_check = std::is_same<comp_tuple_t, check_tuple_t>::value;

    EXPECT_TRUE(auto_check) << same.str();
    EXPECT_FALSE(comp_check) << same.str();
    EXPECT_GE(TIMEMORY_ESC(timemory_variable_156.get<wall_clock>()->get()), 0.9);
    EXPECT_GE(TIMEMORY_ESC(timemory_variable_156.get<cpu_clock>()->get()), 0.65);
    EXPECT_GE(TIMEMORY_ESC(timemory_variable_156.get<cpu_util>()->get()), 65.0);

    if(key != expected.str())
        FAIL() << "expected key: \"" << expected.str() << "\" vs. actual key: \"" << key;
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, marker_comp_type)
{
    auto _obj = TIMEMORY_BLANK_HANDLE(comp_tuple_t, details::get_test_name());
    details::do_sleep(250);
    details::consume(750);
    _obj.stop();
    auto              key = _obj.key();
    std::stringstream expected;
    expected << details::get_test_name();

    using check_tuple_t = std::decay_t<decltype(_obj)>;

    std::stringstream same;
    same << "\n";
    same << "comp_tuple  :: " << tim::demangle<comp_tuple_t>() << "\n";
    same << "auto_tuple  :: " << tim::demangle<auto_tuple_t>() << "\n";
    same << "check_tuple :: " << tim::demangle<check_tuple_t>() << "\n";
    same << "\n";
    same << _obj << std::endl;
    same << "\n";

    auto auto_check = std::is_same<auto_tuple_t, check_tuple_t>::value;
    auto comp_check = std::is_same<comp_tuple_t, check_tuple_t>::value;

    auto tol = 1.0e-6;

    EXPECT_FALSE(auto_check) << same.str();
    EXPECT_TRUE(comp_check) << same.str();
    EXPECT_NEAR(TIMEMORY_ESC(_obj.get<wall_clock>()->get()), 0.0, tol);
    EXPECT_NEAR(TIMEMORY_ESC(_obj.get<cpu_clock>()->get()), 0.0, tol);
    EXPECT_NEAR(TIMEMORY_ESC(_obj.get<cpu_util>()->get()), 0.0, tol);

    if(key != expected.str())
        FAIL() << "expected key: \"" << expected.str() << "\" vs. actual key: \"" << key;
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, blank_handle)
{
    auto _obj = TIMEMORY_BLANK_HANDLE(auto_tuple_t, details::get_test_name());
    details::do_sleep(25);
    details::consume(75);
    _obj.stop();
    auto              key = _obj.key();
    std::stringstream expected;
    expected << details::get_test_name();
    if(key != expected.str())
        FAIL() << "expected key: \"" << expected.str() << "\" vs. actual key: \"" << key;
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, basic_handle)
{
    auto _obj = TIMEMORY_BASIC_HANDLE(auto_tuple_t, details::get_test_name());
    details::do_sleep(25);
    details::consume(75);
    _obj.stop();
    auto              key = _obj.key();
    std::stringstream expected;
    expected << __FUNCTION__ << "/" << details::get_test_name();
    if(key != expected.str())
        FAIL() << "expected key: \"" << expected.str() << "\" vs. actual key: \"" << key;
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

TEST_F(macro_tests, handle)
{
    auto _obj = TIMEMORY_HANDLE(auto_tuple_t, details::get_test_name());
    auto line = __LINE__ - 1;
    details::do_sleep(25);
    details::consume(75);
    _obj.stop();
    auto              key = _obj.key();
    std::stringstream expected;
    std::string       file = __FILE__;
    file = file.substr(file.find_last_of(TIMEMORY_OS_PATH_DELIMITER) + 1);
    expected << __FUNCTION__ << "@" << file << ":" << line << "/"
             << details::get_test_name();
    if(key != expected.str())
        FAIL() << "expected key: \"" << expected.str() << "\" vs. actual key: \"" << key;
    else
        SUCCEED();
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    tim::settings::verbose()     = 0;
    tim::settings::debug()       = false;
    tim::settings::json_output() = true;
    tim::timemory_init(&argc, &argv);
    tim::settings::dart_output() = true;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    tim::settings::dart_type() = "peak_rss";
    // TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE("PEAK_RSS", ::tim::component::peak_rss);
    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();
    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//
// clang-format on
