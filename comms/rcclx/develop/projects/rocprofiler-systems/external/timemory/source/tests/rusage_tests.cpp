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
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;
using floating_t     = double;

static const int64_t niter       = 20;
static const int64_t nelements   = 0.95 * (tim::units::get_page_size() * 500);
static const auto    memory_unit = std::pair<int64_t, string_t>(tim::units::KiB, "KiB");
static peak_rss      peak;
static page_rss      curr;
static current_peak_rss current_peak;
static read_bytes       rb;
static written_bytes    wb;
static auto             tot_size = nelements * sizeof(int64_t) / memory_unit.first;
static auto             tot_rw   = nelements * sizeof(floating_t) / memory_unit.first;

static const double peak_tolerance = 5 * tim::units::MiB;
static const double curr_tolerance = 5 * tim::units::MiB;
static const double byte_tolerance = tot_rw;  // macOS is not dependable

#define CHECK_AVAILABLE(type)                                                            \
    if(!tim::trait::is_available<type>::value)                                           \
    {                                                                                    \
        printf("skipping %s because %s is not available\n",                              \
               details::get_test_name().c_str(), tim::demangle<type>().c_str());         \
        return;                                                                          \
    }

//--------------------------------------------------------------------------------------//
namespace details
{
// this function consumes an unknown number of cpu resources
long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function ensures an allocation cannot be optimized
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
allocate()
{
    peak.reset();
    curr.reset();
    current_peak.reset();

    curr.start();
    peak.start();
    current_peak.start();

    std::vector<int64_t> v(nelements, 15);
    auto                 ret  = fibonacci(0);
    long                 nfib = details::random_entry(v);
    for(int64_t i = 0; i < niter; ++i)
    {
        nfib = details::random_entry(v);
        ret += details::fibonacci(nfib);
    }

    if(ret < 0)
        printf("fibonacci(%li) * %li = %li\n", (long) nfib, (long) niter, ret);

    current_peak.stop();
    curr.stop();
    peak.stop();
}

void
read_write()
{
    {
        std::ofstream ofs("/tmp/file.dat", std::ios::out | std::ios::binary);
        if(ofs)
        {
            for(auto i = 0; i < nelements; ++i)
            {
                floating_t val = static_cast<floating_t>(i);
                ofs.write((char*) &val, sizeof(floating_t));
            }
            return;
        }
        else
        {
            std::cerr << "Error opening '/tmp/file.dat'..." << std::endl;
        }
        ofs.close();
    }
    {
        std::vector<floating_t> data(nelements);
        std::ifstream           ifs("/tmp/file.dat", std::ios::in | std::ios::binary);
        if(ifs)
        {
            for(auto i = 0; i < nelements; ++i)
                ifs.read((char*) (data.data() + i), sizeof(floating_t));
        }
        {
            std::cerr << "Error opening '/tmp/file.dat'..." << std::endl;
        }
        ifs.close();
    }
}

template <typename Tp>
string_t
get_info(const Tp& obj)
{
    stringstream_t ss;
    auto           _unit = static_cast<double>(Tp::get_unit());
    ss << "value = " << obj.get_value() / _unit << " " << Tp::get_display_unit()
       << ", accum = " << obj.get_accum() / _unit << " " << Tp::get_display_unit()
       << std::endl;
    return ss.str();
}

string_t
get_info(const read_bytes& obj)
{
    stringstream_t ss;
    auto           _unit = std::get<0>(read_bytes::get_unit());
    ss << "value = " << std::get<0>(obj.get_value()) / _unit << " "
       << std::get<0>(read_bytes::get_display_unit())
       << ", accum = " << std::get<0>(obj.get_accum()) / _unit << " "
       << std::get<0>(read_bytes::get_display_unit()) << std::endl;
    return ss.str();
}

string_t
get_info(const written_bytes& obj)
{
    stringstream_t ss;
    auto           _unit = std::get<0>(written_bytes::get_unit());
    ss << "value = " << std::get<0>(obj.get_value()) / _unit << " "
       << std::get<0>(written_bytes::get_display_unit())
       << ", accum = " << std::get<0>(obj.get_accum()) / _unit << " "
       << std::get<0>(written_bytes::get_display_unit()) << std::endl;
    return ss.str();
}

string_t
get_info(const current_peak_rss& obj)
{
    stringstream_t ss;
    auto           _unit = std::get<0>(written_bytes::get_unit());
    ss << "value = " << std::get<0>(obj.get_value()) / _unit << " "
       << std::get<0>(written_bytes::get_display_unit())
       << ", accum = " << std::get<0>(obj.get_accum()) / _unit << " "
       << std::get<0>(written_bytes::get_display_unit()) << std::endl;
    return ss.str();
}

inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

template <typename Tp>
void
print_info(const Tp& obj, int64_t expected)
{
    std::cout << std::endl;
    std::cout << "[" << get_test_name() << "]>  measured : " << obj << std::endl;
    std::cout << "[" << get_test_name() << "]>  expected : " << expected << " "
              << memory_unit.second << std::endl;
    std::cout << "[" << get_test_name() << "]> data info : " << get_info(obj)
              << std::endl;
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class rusage_tests : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        tim::settings::verbose()      = 0;
        tim::settings::debug()        = false;
        tim::settings::json_output()  = true;
        tim::settings::mpi_thread()   = false;
        tim::settings::precision()    = 9;
        tim::settings::memory_units() = memory_unit.second;
        tim::dmp::initialize(test_argc, test_argv);
        tim::timemory_init(test_argc, test_argv);
        tim::settings::file_output() = false;

        // preform allocation only once here
        details::allocate();
    }

    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN
};

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, peak_rss)
{
    CHECK_AVAILABLE(peak_rss);
    details::print_info(peak, tot_size);
    ASSERT_NEAR(tot_size, peak.get(), peak_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, page_rss)
{
    CHECK_AVAILABLE(page_rss);
    details::print_info(curr, tot_size);
    ASSERT_NEAR(tot_size, curr.get(), curr_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, read_bytes)
{
    CHECK_AVAILABLE(read_bytes);
    details::print_info(rb, tot_rw);
    ASSERT_NEAR(tot_rw, std::get<0>(rb.get()), byte_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, written_bytes)
{
    CHECK_AVAILABLE(written_bytes);
    details::print_info(wb, tot_rw);
    ASSERT_NEAR(tot_rw, std::get<0>(wb.get()), byte_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(rusage_tests, current_peak_rss)
{
    CHECK_AVAILABLE(current_peak_rss);
    details::print_info(current_peak, tot_size);
    ASSERT_NEAR(tot_size, current_peak.get().second - current_peak.get().first,
                peak_tolerance);
}

//--------------------------------------------------------------------------------------//
