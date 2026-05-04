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
void
random_fill(std::vector<Tp>& v, Tp scale, Tp offset = 1.0)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::generate(v.begin(), v.end(), [&]() {
        return (scale * std::generate_canonical<Tp, 10>(rng)) + offset;
    });
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

class io_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

using namespace tim::component;

using io_bundle_t =
    tim::component_tuple<wall_clock, read_char, written_char, read_bytes, written_bytes>;
using type = double;

//--------------------------------------------------------------------------------------//

TEST_F(io_tests, bytes)
{
    tim::settings::memory_units() = "kb";

    io_bundle_t  _obj{ details::get_test_name() };
    const size_t nsize = tim::units::get_page_size() / sizeof(type);
    const size_t nitr  = 10;

    for(size_t j = 0; j < nitr; ++j)
    {
        _obj.start();
        std::vector<type> _data(nsize, 0.0);
        details::random_fill(_data, 10.0, 1.0);
        {
            std::ofstream ofs{ TIMEMORY_JOIN("", ".", details::get_test_name(), "_data",
                                             j, ".txt"),
                               std::ios::out | std::ios::binary };
            for(auto& itr : _data)
                ofs.write((char*) &itr, sizeof(type));
        }
        auto _data_cpy = _data;
        _data.clear();
        _data.resize(nsize, 0.0);
        {
            std::ifstream ifs{ TIMEMORY_JOIN("", ".", details::get_test_name(), "_data",
                                             j, ".txt"),
                               std::ios::in | std::ios::binary };
            for(auto& itr : _data)
                ifs.read((char*) &itr, sizeof(type));
        }
        for(size_t i = 0; i < _data.size(); ++i)
            EXPECT_NEAR(_data.at(i), _data_cpy.at(i), 1.0e-3);
        auto val = details::random_entry(_data);
        EXPECT_GE(val, 1.0);
        EXPECT_LE(val, 11.0);
        _obj.stop();

        ASSERT_TRUE(_obj.get<read_bytes>() != nullptr)
            << " iteration " << j << " :: " << _obj;
        EXPECT_GT(std::get<0>(_obj.get<read_bytes>()->get()), 0)
            << " iteration " << j << " :: " << _obj;
        EXPECT_GT(std::get<1>(_obj.get<read_bytes>()->get()), 0)
            << " iteration " << j << " :: " << _obj;
        EXPECT_NEAR(std::get<0>(_obj.get<read_bytes>()->get()),
                    nsize * sizeof(type) / tim::units::KB,
                    std::get<0>(_obj.get<read_bytes>()->get()) * 0.05)
            << " iteration " << j << " :: " << _obj;

        ASSERT_TRUE(_obj.get<written_bytes>() != nullptr)
            << " iteration " << j << " :: " << _obj;
        EXPECT_GT(std::get<0>(_obj.get<written_bytes>()->get()), 0)
            << " iteration " << j << " :: " << _obj;
        EXPECT_GT(std::get<1>(_obj.get<written_bytes>()->get()), 0)
            << " iteration " << j << " :: " << _obj;
        EXPECT_NEAR(std::get<0>(_obj.get<written_bytes>()->get()),
                    nsize * sizeof(type) / tim::units::KB,
                    std::get<0>(_obj.get<written_bytes>()->get()) * 0.05)
            << " iteration " << j << " :: " << _obj;

        EXPECT_NEAR(std::get<0>(_obj.get<read_bytes>()->get()),
                    std::get<0>(_obj.get<written_bytes>()->get()), 1.0e-3)
            << " iteration " << j << " :: " << _obj;
    }
}

//--------------------------------------------------------------------------------------//
