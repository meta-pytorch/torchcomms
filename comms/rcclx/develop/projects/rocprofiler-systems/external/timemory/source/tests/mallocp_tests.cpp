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
#include "timemory/tools/timemory-mallocp.h"

using namespace tim::component;

//--------------------------------------------------------------------------------------//

using data_type                  = int64_t;
static const int64_t nelements   = 0.95 * (tim::units::get_page_size() * 100);
static const auto    memory_unit = std::pair<int64_t, string_t>(tim::units::KiB, "KiB");
static const auto    tot_size    = nelements * sizeof(data_type) / memory_unit.first;
static const double  tolerance   = 25;

namespace details
{
//  Get the current tests name
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
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

auto
foo()
{
    uint64_t               n = 0;
    std::vector<data_type> _data(nelements, 0);
    std::generate(_data.begin(), _data.end(), [&n]() { return ++n; });
    return random_entry(_data);
}

auto
bar(size_t n)
{
    uint64_t mallocp_idx = UINT64_MAX;
    if(n % 2 == 1)
        mallocp_idx = timemory_start_mallocp();

    data_type tot = 0;
    for(size_t i = 0; i < n; ++i)
        tot += foo();

    if(n % 2 == 1)
        timemory_stop_mallocp(mallocp_idx);

    return tot;
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class mallocp_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

TEST_F(mallocp_tests, registration)
{
    ASSERT_EQ(tim::storage<malloc_gotcha>::instance()->size(), 0);

    tim::settings::memory_units() = "KiB";
    size_t                   idx  = 0;
    std::array<data_type, 4> ret{};

    timemory_register_mallocp();
    ret.at(idx++) = details::foo();
    ret.at(idx++) = details::bar(1);
    ret.at(idx++) = details::bar(2);
    ret.at(idx++) = details::foo();
    timemory_register_mallocp();

    for(auto& itr : ret)
    {
        EXPECT_GT(itr, 0);
    }

    auto _storage = tim::storage<malloc_gotcha>::instance()->get();

    ASSERT_GE(_storage.size(), 2);

    std::array<malloc_gotcha, 2> _data{};
    for(auto& itr : _storage)
    {
        if(itr.prefix().find("malloc") != std::string::npos)
            _data.at(0) += itr.data();
        else if(itr.prefix().find("calloc") != std::string::npos)
            _data.at(0) += itr.data();
        else if(itr.prefix().find("free") != std::string::npos)
            _data.at(1) += itr.data();
    }

    auto unit_str = memory_unit.second;
    auto scale    = static_cast<double>(memory_unit.first) / malloc_gotcha::get_unit();

    auto _alloc   = _data.at(0).get() / scale;
    auto _dealloc = _data.at(1).get() / scale;

    std::cout << "total size * 5  : " << 5 * tot_size << " " << unit_str << std::endl;
    std::cout << "allocated       : " << _alloc << " " << unit_str << std::endl;
    std::cout << "deallocated     : " << _dealloc << " " << unit_str << std::endl;

    EXPECT_NEAR(5 * tot_size, _alloc, tolerance) << _data.at(0) << std::endl;
    EXPECT_NEAR(5 * tot_size, _dealloc, tolerance) << _data.at(1) << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(mallocp_tests, start_stop)
{
    ASSERT_EQ(tim::storage<malloc_gotcha>::instance()->size(), 0);

    tim::settings::memory_units() = "KiB";
    size_t                   idx  = 0;
    std::array<data_type, 4> ret{};

    ret.at(idx++) = details::foo();
    ret.at(idx++) = details::bar(3);
    ret.at(idx++) = details::bar(2);
    ret.at(idx++) = details::foo();

    for(auto& itr : ret)
    {
        EXPECT_GT(itr, 0);
    }

    auto _storage = tim::storage<malloc_gotcha>::instance()->get();

    ASSERT_GE(_storage.size(), 2);

    std::array<malloc_gotcha, 2> _data{};
    for(auto& itr : _storage)
    {
        if(itr.prefix().find("malloc") != std::string::npos)
            _data.at(0) += itr.data();
        else if(itr.prefix().find("calloc") != std::string::npos)
            _data.at(0) += itr.data();
        else if(itr.prefix().find("free") != std::string::npos)
            _data.at(1) += itr.data();
    }

    auto unit_str = memory_unit.second;
    auto scale    = static_cast<double>(memory_unit.first) / malloc_gotcha::get_unit();

    auto _alloc   = _data.at(0).get() / scale;
    auto _dealloc = _data.at(1).get() / scale;

    std::cout << "total size * 3  : " << 3 * tot_size << " " << unit_str << std::endl;
    std::cout << "allocated       : " << _alloc << " " << unit_str << std::endl;
    std::cout << "deallocated     : " << _dealloc << " " << unit_str << std::endl;

    EXPECT_NEAR(3 * tot_size, _alloc, tolerance) << _data.at(0) << std::endl;
    EXPECT_NEAR(3 * tot_size, _dealloc, tolerance) << _data.at(1) << std::endl;
}

//--------------------------------------------------------------------------------------//
