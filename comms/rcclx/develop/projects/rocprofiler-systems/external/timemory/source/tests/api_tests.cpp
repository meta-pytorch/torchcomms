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

class api_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

namespace os       = ::tim::os;
namespace project  = ::tim::project;
namespace category = ::tim::category;
namespace tpls     = ::tim::tpls;
namespace trait    = ::tim::trait;

using namespace tim::component;

using tim_bundle_t  = tim::component_bundle<project::timemory, wall_clock, cpu_clock>;
using cali_bundle_t = tim::component_bundle<tpls::caliper, caliper_marker, wall_clock>;
using time_bundle_t = tim::component_tuple<wall_clock, cpu_clock, num_minor_page_faults,
                                           read_bytes, written_bytes>;
using os_bundle_t   = tim::component_bundle<os::agnostic, wall_clock, cpu_clock>;

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, enum_vs_macro)
{
    namespace component = tim::component;
    auto macro_sz = std::tuple_size<tim::type_list<TIMEMORY_COMPONENT_TYPES>>::value;
    auto enum_sz =
        TIMEMORY_NATIVE_COMPONENTS_END - TIMEMORY_NATIVE_COMPONENT_INTERNAL_SIZE;

    EXPECT_EQ(macro_sz, enum_sz);
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, placeholder)
{
    using nothing_placeholder = placeholder<nothing>;

    auto in_complete = tim::is_one_of<nothing_placeholder, tim::complete_types_t>::value;
    auto in_avail    = tim::is_one_of<nothing_placeholder, tim::available_types_t>::value;

    EXPECT_FALSE(in_complete) << "complete_types_t: "
                              << tim::demangle<tim::complete_types_t>() << std::endl;

    EXPECT_FALSE(in_avail) << "available_types_t: "
                           << tim::demangle<tim::available_types_t>() << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, timemory)
{
    using test_t  = project::timemory;
    auto incoming = trait::runtime_enabled<test_t>::get();
    trait::runtime_enabled<test_t>::set(false);

    tim_bundle_t _obj(details::get_test_name());
    _obj.start();
    details::consume(1000);
    _obj.stop();
    EXPECT_NEAR(_obj.get<wall_clock>()->get(), 0.0, 1.0e-6);
    EXPECT_NEAR(_obj.get<cpu_clock>()->get(), 0.0, 1.0e-6);

    trait::runtime_enabled<test_t>::set(incoming);
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, tpls)
{
    using test_t  = tpls::caliper;
    auto incoming = trait::runtime_enabled<test_t>::get();
    trait::runtime_enabled<test_t>::set(false);

    EXPECT_FALSE(trait::runtime_enabled<test_t>::get());
    EXPECT_FALSE(trait::runtime_enabled<caliper_marker>::get());

    cali_bundle_t _obj(details::get_test_name());
    _obj.start();
    details::consume(1000);
    _obj.stop();

    std::cout << "tpls::caliper available == " << std::boolalpha << "[compile-time "
              << tim::trait::is_available<test_t>::value << "] [run-time "
              << trait::runtime_enabled<test_t>::get() << "]\n";

    if(tim::trait::is_available<test_t>::value)
    {
        EXPECT_NEAR(_obj.get<wall_clock>()->get(), 0.0, 1.0e-6) << "obj: " << _obj;
    }
    else
    {
        EXPECT_EQ(_obj.get<wall_clock>(), nullptr) << "obj: " << _obj;
    }

    trait::runtime_enabled<test_t>::set(incoming);
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, category)
{
    // a type in a non-timing category that should still be enabled
#if defined(TIMEMORY_UNIX)
    using check_type = num_minor_page_faults;
#elif defined(TIMEMORY_WINDOWS)
    using check_type = read_bytes;
#endif

    using test_t  = category::timing;
    auto incoming = trait::runtime_enabled<test_t>::get();
    trait::runtime_enabled<test_t>::set(false);

    using wc_api_t   = trait::component_apis_t<wall_clock>;
    using wc_true_t  = tim::mpl::get_true_types_t<trait::runtime_configurable, wc_api_t>;
    using wc_false_t = tim::mpl::get_false_types_t<trait::runtime_configurable, wc_api_t>;
    using apitypes_t = typename trait::runtime_enabled<wall_clock>::category_type_list;

    puts("");
    TIMEMORY_PRINT_HERE("component-apis : %s", tim::demangle<wc_api_t>().c_str());
    TIMEMORY_PRINT_HERE("true-types     : %s", tim::demangle<wc_false_t>().c_str());
    TIMEMORY_PRINT_HERE("false-types    : %s", tim::demangle<wc_true_t>().c_str());
    TIMEMORY_PRINT_HERE("type-traits    : %s", tim::demangle<apitypes_t>().c_str());
    puts("");

    EXPECT_FALSE(trait::runtime_enabled<wall_clock>::get());
    EXPECT_FALSE(trait::runtime_enabled<cpu_clock>::get());
    EXPECT_TRUE(trait::runtime_enabled<check_type>::get());

    time_bundle_t _obj(details::get_test_name());
    _obj.start();
    details::consume(500);
    for(size_t i = 0; i < 10; ++i)
    {
        std::vector<double> _data(10000, 0.0);
        details::random_fill(_data, 10.0, 1.0);
        {
            std::ofstream ofs{ TIMEMORY_JOIN("", ".", details::get_test_name(), "_data",
                                             i, ".txt") };
            for(auto& itr : _data)
                ofs << std::fixed << std::setprecision(6) << itr << " ";
        }
        auto _data_cpy = _data;
        _data.clear();
        _data.resize(10000, 0.0);
        {
            std::ifstream ifs{ TIMEMORY_JOIN("", ".", details::get_test_name(), "_data",
                                             i, ".txt") };
            for(auto& itr : _data)
                ifs >> itr;
        }
        for(size_t j = 0; j < _data.size(); ++j)
            EXPECT_NEAR(_data.at(j), _data_cpy.at(j), 1.0e-3);
        auto val = details::random_entry(_data);
        EXPECT_GE(val, 1.0);
        EXPECT_LE(val, 11.0);
    }
    _obj.stop();

    ASSERT_TRUE(_obj.get<wall_clock>() != nullptr);
    ASSERT_TRUE(_obj.get<cpu_clock>() != nullptr);
    ASSERT_TRUE(_obj.get<check_type>() != nullptr);

    EXPECT_NEAR(_obj.get<wall_clock>()->get(), 0.0, 1.0e-6);
    EXPECT_NEAR(_obj.get<cpu_clock>()->get(), 0.0, 1.0e-6);
#if defined(TIMEMORY_UNIX)
    EXPECT_GT(_obj.get<check_type>()->get(), 0);
#elif defined(TIMEMORY_WINDOWS)
    EXPECT_GT(std::get<0>(_obj.get<check_type>()->get()), 0);
    EXPECT_GT(std::get<1>(_obj.get<check_type>()->get()), 0);
    EXPECT_NEAR(std::get<0>(_obj.get<read_bytes>()->get()),
                std::get<0>(_obj.get<written_bytes>()->get()), 1.0e-3);
#endif

    trait::runtime_enabled<test_t>::set(incoming);
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, os)
{
    trait::apply<trait::runtime_enabled>::set<os::supports_unix, os::supports_windows,
                                              cpu_clock>(false);

    os_bundle_t _obj(details::get_test_name());
    _obj.start();
    details::consume(1000);
    _obj.stop();
    EXPECT_NEAR(_obj.get<wall_clock>()->get(), 1.0, 0.1);
    EXPECT_NEAR(_obj.get<cpu_clock>()->get(), 0.0, 1.0e-6);

    trait::apply<trait::runtime_enabled>::set<os::supports_unix, os::supports_windows,
                                              cpu_clock>(true);
}

//--------------------------------------------------------------------------------------//
