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
#include "timemory/utility/signals.hpp"
#include "timemory/variadic/functional.hpp"

#include "gtest/gtest.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

TIMEMORY_DECLARE_COMPONENT(tst_roofline_flops)
TIMEMORY_DECLARE_COMPONENT(tst_roofline_sp_flops)
TIMEMORY_DECLARE_COMPONENT(tst_roofline_dp_flops)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::tst_roofline_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::tst_roofline_sp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::tst_roofline_dp_flops, false_type)

using namespace tim::component;

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

//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n)
{
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
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
}  // namespace details

//--------------------------------------------------------------------------------------//

class component_bundle_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, variadic)
{
    static constexpr size_t nz = 7;
    std::array<size_t, nz>  sizes;
    size_t                  n = 0;
    std::generate(sizes.begin(), sizes.end(), [&]() { return n++; });

    {
        using bundle_t = tim::auto_bundle<TIMEMORY_API, wall_clock, peak_rss>;
        sizes[0]       = bundle_t::size();
    }

    {
        using bundle_t = tim::component_bundle<TIMEMORY_API, wall_clock, peak_rss>;
        sizes[1]       = bundle_t::size();
    }

    {
        using bundle_t =
            tim::component_bundle_t<TIMEMORY_API, wall_clock, peak_rss,
                                    tst_roofline_dp_flops*, tst_roofline_flops*>;
        sizes[2] = bundle_t::size();
    }

    {
        using bundle_t =
            tim::auto_bundle_t<TIMEMORY_API, wall_clock, peak_rss, tst_roofline_dp_flops*,
                               tst_roofline_flops*, tst_roofline_sp_flops*>;
        sizes[3] = bundle_t::size();
    }

    {
        using bundle_t =
            tim::component_bundle_t<TIMEMORY_API, wall_clock, peak_rss,
                                    tst_roofline_dp_flops*, tst_roofline_flops,
                                    tst_roofline_sp_flops*>;
        sizes[4] = bundle_t::size();
    }

    {
        using bundle_t =
            tim::component_bundle_t<TIMEMORY_API, wall_clock, peak_rss,
                                    tst_roofline_dp_flops*, tst_roofline_flops*,
                                    tst_roofline_sp_flops*>;
        sizes[5] = bundle_t::size();
    }

    {
        using bundle_t =
            tim::auto_bundle_t<TIMEMORY_API, wall_clock, peak_rss, tst_roofline_dp_flops*,
                               tst_roofline_flops, tst_roofline_sp_flops*>;
        sizes[6] = bundle_t::size();
    }

    // std::cout << "\n";
    // for(size_t i = 0; i < nz; ++i)
    //     std::cout << "size[" << i << "] = " << sizes[i] << std::endl;
    // std::cout << "\n";

    EXPECT_EQ(sizes[0], sizes[1]);

    for(size_t i = 2; i < nz; ++i)
    {
        EXPECT_EQ(sizes[0], sizes[i]);
        EXPECT_EQ(sizes[1], sizes[i]);
    }

    using bundle_t = tim::auto_bundle<TIMEMORY_API, wall_clock, cpu_clock, peak_rss*>;

    auto hsize = bundle_t::size();
    EXPECT_EQ(hsize, 3);

    auto wsize = tim::storage<wall_clock>::instance()->size();
    auto csize = tim::storage<cpu_clock>::instance()->size();
    auto psize = tim::storage<peak_rss>::instance()->size();

    // std::cout << "\nbundle        : " << tim::demangle<bundle_t>() << "\n";
    // std::cout << "\n";
    long nfib = 30;
    long ival = 200;
    long nitr = 1000;
    for(long i = 0; i < nitr; ++i)
    {
        bundle_t bundle(details::get_test_name());
        auto     ret = details::fibonacci(nfib);
        if(i % ival == (ival - 1))
            printf("\nfibonacci(%li) = %li\n\n", (long int) nfib, (long int) ret);
    }

    wsize = tim::storage<wall_clock>::instance()->size() - wsize;
    csize = tim::storage<cpu_clock>::instance()->size() - csize;
    psize = tim::storage<peak_rss>::instance()->size() - psize;

    EXPECT_EQ(wsize, 1);
    EXPECT_EQ(csize, 1);
    EXPECT_EQ(psize, 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, get)
{
    tim::trait::runtime_enabled<cpu_roofline<float>>::set(false);
    tim::trait::runtime_enabled<gpu_roofline<float>>::set(false);
    tim::trait::runtime_enabled<cpu_roofline<double>>::set(false);
    tim::trait::runtime_enabled<gpu_roofline<double>>::set(false);

    using lhs_t =
        tim::component_bundle_t<TIMEMORY_API, wall_clock, user_clock*, system_clock,
                                cpu_roofline<double>*, gpu_roofline<float>>;
    using rhs_t = tim::auto_bundle_t<TIMEMORY_API, wall_clock*, cpu_clock*,
                                     cpu_roofline<float>*, gpu_roofline<double>>;

    // using lhs_data_t = typename lhs_t::data_type;
    // using rhs_data_t = typename rhs_t::data_type;
    // using rhs_comp_t = typename rhs_t::component_type;

    lhs_t::get_initializer() = [](auto& cl) {
        cl.template initialize<wall_clock, user_clock>();
    };
    rhs_t::get_initializer() = [](auto& cl) {
        cl.template initialize<wall_clock, cpu_clock>();
    };

    auto lhs = lhs_t(TIMEMORY_JOIN("/", details::get_test_name(), "lhs"));
    auto rhs = rhs_t(TIMEMORY_JOIN("/", details::get_test_name(), "rhs"));

    tim::invoke::start(std::tie(lhs, rhs));
    tim::invoke::mark_begin(std::tie(lhs, rhs));

    std::this_thread::sleep_for(std::chrono::seconds(1));
    details::consume(1000);

    tim::invoke::mark_end(std::tie(lhs, rhs));
    tim::invoke::stop(std::tie(lhs, rhs));

    auto cb = lhs.get();
    auto ab = rhs.get();

    // std::cout << "\n" << std::flush;

    // std::cout << "rhs_t      = " << tim::demangle<rhs_t>() << "\n";
    // std::cout << "lhs_t      = " << tim::demangle<lhs_t>() << "\n";
    // std::cout << "rhs_comp_t = " << tim::demangle<rhs_comp_t>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "lhs_data_t = " << tim::demangle<lhs_data_t>() << "\n";
    // std::cout << "rhs_data_t = " << tim::demangle<rhs_data_t>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "cb         = " << tim::demangle<decltype(cb)>() << "\n";
    // std::cout << "ab         = " << tim::demangle<decltype(ab)>() << "\n";
    // std::cout << "\n" << std::flush;

    tim::invoke::print(std::cout, lhs, rhs);

    EXPECT_NEAR(std::get<0>(cb), 2.0, 0.15);
    EXPECT_NEAR(std::get<0>(ab), 2.0, 0.15);
    EXPECT_NEAR(std::get<1>(cb) + std::get<2>(cb), 1.0, 0.6);
    EXPECT_NEAR(std::get<1>(ab), 1.0, 0.6);
}

//--------------------------------------------------------------------------------------//
//  these types are available on every OS
//
template <typename ApiT>
struct test
{
    using direct_auto_t =
        tim::auto_bundle<ApiT, wall_clock, cpu_clock, peak_rss, page_rss*, user_clock*>;
    using direct_comp_t = tim::component_bundle<ApiT, wall_clock, cpu_clock, peak_rss,
                                                page_rss*, user_clock*>;
    using derive_auto_t = typename direct_comp_t::auto_type;
    using derive_comp_t = typename direct_auto_t::component_type;

    using direct_auto_data_t = typename direct_auto_t::data_type;
    using direct_comp_data_t = typename direct_comp_t::data_type;
    using derive_auto_data_t = typename derive_auto_t::data_type;
    using derive_comp_data_t = typename derive_comp_t::data_type;
};

template <typename Tp>
constexpr auto
fixed_count(int) -> decltype(Tp::fixed_count(), uint64_t())
{
    return Tp::fixed_count();
}

template <typename Tp>
constexpr auto
fixed_count(long)
{
    return (tim::mpl::get_tuple_size<Tp>::value -
            tim::mpl::get_tuple_size<
                typename tim::mpl::get_true_types<tim::is_optional, Tp>::type>::value);
}

template <typename Tp>
constexpr auto
optional_count(int) -> decltype(Tp::optional_count(), uint64_t())
{
    return Tp::optional_count();
}

template <typename Tp>
constexpr auto
optional_count(long)
{
    return (tim::mpl::get_tuple_size<
            typename tim::mpl::get_true_types<tim::is_optional, Tp>::type>::value);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_type_fixed_size_check)
{
    using api_t = TIMEMORY_API;
    EXPECT_TRUE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(fixed_count<test<api_t>::direct_auto_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::direct_comp_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::derive_auto_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::derive_comp_t>(0), 3);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_data_fixed_size_check)
{
    using api_t = TIMEMORY_API;
    EXPECT_TRUE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(fixed_count<test<api_t>::direct_auto_data_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::direct_comp_data_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::derive_auto_data_t>(0), 3);
    EXPECT_EQ(fixed_count<test<api_t>::derive_comp_data_t>(0), 3);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_type_optional_size_check)
{
    using api_t = TIMEMORY_API;
    EXPECT_TRUE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(optional_count<test<api_t>::direct_auto_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::direct_comp_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::derive_auto_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::derive_comp_t>(0), 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_data_optional_size_check)
{
    using api_t = TIMEMORY_API;
    EXPECT_TRUE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(optional_count<test<api_t>::direct_auto_data_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::direct_comp_data_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::derive_auto_data_t>(0), 2);
    EXPECT_EQ(optional_count<test<api_t>::derive_comp_data_t>(0), 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_type_count_wo_init)
{
    using api_t = TIMEMORY_API;
    auto A      = test<api_t>::direct_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::direct_auto"));
    auto B = test<api_t>::direct_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::direct_comp"));
    auto C = test<api_t>::derive_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::derive_auto"));
    auto D = test<api_t>::derive_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::derive_comp"));

    EXPECT_TRUE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(A.count(), 3);
    EXPECT_EQ(B.count(), 3);
    EXPECT_EQ(C.count(), 3);
    EXPECT_EQ(D.count(), 3);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, native_type_count_w_init)
{
    using api_t                                   = TIMEMORY_API;
    test<api_t>::direct_auto_t::get_initializer() = [](auto& cb) {
        // initialize one of two pointers and a type which does not belong
        cb.template initialize<page_rss, system_clock>();
    };

    auto A = test<api_t>::direct_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::direct_auto"));
    auto C = test<api_t>::derive_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::derive_auto"));

    test<api_t>::derive_comp_t::get_initializer() = [](auto& cb) {
        // initialize both pointers
        cb.template initialize<page_rss, user_clock>();
    };

    auto B = test<api_t>::direct_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::direct_comp"));
    auto D = test<api_t>::derive_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<TIMEMORY_API>::derive_comp"));

    EXPECT_TRUE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(A.count(), 4);
    EXPECT_EQ(B.count(), 5);
    EXPECT_EQ(C.count(), 4);
    EXPECT_EQ(D.count(), 5);

    test<api_t>::derive_auto_t::get_initializer() = [](auto&) {};
    test<api_t>::direct_comp_t::get_initializer() = [](auto&) {};
}

//--------------------------------------------------------------------------------------//

// declare a new API
TIMEMORY_DEFINE_API(custom_tag)
// make the API unavailable
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, api::custom_tag, false_type)
// macro for API
#define CUSTOM_API tim::api::custom_tag

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_type_fixed_size_check)
{
    using api_t = CUSTOM_API;
    EXPECT_FALSE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(fixed_count<test<api_t>::direct_auto_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::direct_comp_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::derive_auto_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::derive_comp_t>(0), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_data_fixed_size_check)
{
    using api_t = CUSTOM_API;
    EXPECT_FALSE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(fixed_count<test<api_t>::direct_auto_data_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::direct_comp_data_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::derive_auto_data_t>(0), 0);
    EXPECT_EQ(fixed_count<test<api_t>::derive_comp_data_t>(0), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_type_optional_size_check)
{
    using api_t = CUSTOM_API;
    EXPECT_EQ(optional_count<test<api_t>::direct_auto_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::direct_comp_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::derive_auto_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::derive_comp_t>(0), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_data_optional_size_check)
{
    using api_t = CUSTOM_API;
    EXPECT_FALSE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(optional_count<test<api_t>::direct_auto_data_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::direct_comp_data_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::derive_auto_data_t>(0), 0);
    EXPECT_EQ(optional_count<test<api_t>::derive_comp_data_t>(0), 0);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_type_count_wo_init)
{
    using api_t = CUSTOM_API;
    auto A      = test<api_t>::direct_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::direct_auto"));
    auto B = test<api_t>::direct_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::direct_comp"));
    auto C = test<api_t>::derive_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::derive_auto"));
    auto D = test<api_t>::derive_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::derive_comp"));

    EXPECT_FALSE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(A.count(), 0);
    EXPECT_EQ(B.count(), 0);
    EXPECT_EQ(C.count(), 0);
    EXPECT_EQ(D.count(), 0);
}

//--------------------------------------------------------------------------------------//

template <typename T>
using get_is_invalid_t = tim::operation::get_is_invalid<T, false>;

TEST_F(component_bundle_tests, invalid)
{
    using bundle_t = tim::component_tuple<wall_clock, cpu_clock, cpu_util, peak_rss>;

    auto wc_size_orig = tim::storage<wall_clock>::instance()->size();
    auto cu_size_orig = tim::storage<cpu_util>::instance()->size();
    auto cc_size_orig = tim::storage<cpu_clock>::instance()->size();
    auto pr_size_orig = tim::storage<peak_rss>::instance()->size();

    long ret = 0;
    {
        bundle_t _instance{ details::get_test_name(),
                            tim::scope::flat{} + tim::scope::timeline{} };

        _instance.invoke<tim::operation::set_is_invalid>(true);
        _instance.get<cpu_clock>()->set_is_invalid(false);

        EXPECT_TRUE(_instance.get<wall_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_util>()->get_is_invalid());
        EXPECT_FALSE(_instance.get<cpu_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<peak_rss>()->get_is_invalid());

        EXPECT_EQ(_instance.get<wall_clock>()->get_is_invalid(),
                  get_is_invalid_t<wall_clock>{}(*_instance.get<wall_clock>()));
        EXPECT_EQ(_instance.get<cpu_util>()->get_is_invalid(),
                  get_is_invalid_t<cpu_util>{}(*_instance.get<cpu_util>()));
        EXPECT_EQ(_instance.get<cpu_clock>()->get_is_invalid(),
                  get_is_invalid_t<cpu_clock>{}(*_instance.get<cpu_clock>()));
        EXPECT_EQ(_instance.get<peak_rss>()->get_is_invalid(),
                  get_is_invalid_t<peak_rss>{}(*_instance.get<peak_rss>()));

        _instance.start();
        ret += details::fibonacci(35);
        _instance.stop();
    }

    {
        bundle_t _instance{ details::get_test_name(),
                            tim::scope::flat{} + tim::scope::timeline{} };

        _instance.invoke<tim::operation::set_is_invalid>(true);

        EXPECT_TRUE(_instance.get<wall_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_util>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<peak_rss>()->get_is_invalid());

        EXPECT_EQ(_instance.get<wall_clock>()->get_is_invalid(),
                  get_is_invalid_t<wall_clock>{}(*_instance.get<wall_clock>()));
        EXPECT_EQ(_instance.get<cpu_util>()->get_is_invalid(),
                  get_is_invalid_t<cpu_util>{}(*_instance.get<cpu_util>()));
        EXPECT_EQ(_instance.get<cpu_clock>()->get_is_invalid(),
                  get_is_invalid_t<cpu_clock>{}(*_instance.get<cpu_clock>()));
        EXPECT_EQ(_instance.get<peak_rss>()->get_is_invalid(),
                  get_is_invalid_t<peak_rss>{}(*_instance.get<peak_rss>()));

        _instance.start();

        EXPECT_TRUE(_instance.get<wall_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_util>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<peak_rss>()->get_is_invalid());
        EXPECT_TRUE(get_is_invalid_t<wall_clock>{}(*_instance.get<wall_clock>()));
        EXPECT_TRUE(get_is_invalid_t<cpu_util>{}(*_instance.get<cpu_util>()));
        EXPECT_TRUE(get_is_invalid_t<cpu_clock>{}(*_instance.get<cpu_clock>()));
        EXPECT_TRUE(get_is_invalid_t<peak_rss>{}(*_instance.get<peak_rss>()));

        ret += details::fibonacci(35);
        _instance.stop();

        EXPECT_TRUE(_instance.get<wall_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_util>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<peak_rss>()->get_is_invalid());
        EXPECT_TRUE(get_is_invalid_t<wall_clock>{}(*_instance.get<wall_clock>()));
        EXPECT_TRUE(get_is_invalid_t<cpu_util>{}(*_instance.get<cpu_util>()));
        EXPECT_TRUE(get_is_invalid_t<cpu_clock>{}(*_instance.get<cpu_clock>()));
        EXPECT_TRUE(get_is_invalid_t<peak_rss>{}(*_instance.get<peak_rss>()));
    }

    {
        bundle_t _instance{ details::get_test_name(),
                            tim::scope::flat{} + tim::scope::timeline{} };

        EXPECT_FALSE(_instance.get<wall_clock>()->get_is_invalid());
        EXPECT_FALSE(_instance.get<cpu_util>()->get_is_invalid());
        EXPECT_FALSE(_instance.get<cpu_clock>()->get_is_invalid());
        EXPECT_FALSE(_instance.get<peak_rss>()->get_is_invalid());
        EXPECT_FALSE(get_is_invalid_t<wall_clock>{}(*_instance.get<wall_clock>()));
        EXPECT_FALSE(get_is_invalid_t<cpu_util>{}(*_instance.get<cpu_util>()));
        EXPECT_FALSE(get_is_invalid_t<cpu_clock>{}(*_instance.get<cpu_clock>()));
        EXPECT_FALSE(get_is_invalid_t<peak_rss>{}(*_instance.get<peak_rss>()));

        _instance.invoke<tim::operation::set_is_invalid>(true);

        EXPECT_TRUE(_instance.get<wall_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_util>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<cpu_clock>()->get_is_invalid());
        EXPECT_TRUE(_instance.get<peak_rss>()->get_is_invalid());
        EXPECT_TRUE(get_is_invalid_t<wall_clock>{}(*_instance.get<wall_clock>()));
        EXPECT_TRUE(get_is_invalid_t<cpu_util>{}(*_instance.get<cpu_util>()));
        EXPECT_TRUE(get_is_invalid_t<cpu_clock>{}(*_instance.get<cpu_clock>()));
        EXPECT_TRUE(get_is_invalid_t<peak_rss>{}(*_instance.get<peak_rss>()));

        _instance.invoke<tim::operation::set_is_invalid>(false);

        EXPECT_FALSE(_instance.get<wall_clock>()->get_is_invalid());
        EXPECT_FALSE(_instance.get<cpu_util>()->get_is_invalid());
        EXPECT_FALSE(_instance.get<cpu_clock>()->get_is_invalid());
        EXPECT_FALSE(_instance.get<peak_rss>()->get_is_invalid());
        EXPECT_FALSE(get_is_invalid_t<wall_clock>{}(*_instance.get<wall_clock>()));
        EXPECT_FALSE(get_is_invalid_t<cpu_util>{}(*_instance.get<cpu_util>()));
        EXPECT_FALSE(get_is_invalid_t<cpu_clock>{}(*_instance.get<cpu_clock>()));
        EXPECT_FALSE(get_is_invalid_t<peak_rss>{}(*_instance.get<peak_rss>()));

        EXPECT_EQ(_instance.get<wall_clock>()->get_is_invalid(),
                  get_is_invalid_t<wall_clock>{}(*_instance.get<wall_clock>()));
        EXPECT_EQ(_instance.get<cpu_util>()->get_is_invalid(),
                  get_is_invalid_t<cpu_util>{}(*_instance.get<cpu_util>()));
        EXPECT_EQ(_instance.get<cpu_clock>()->get_is_invalid(),
                  get_is_invalid_t<cpu_clock>{}(*_instance.get<cpu_clock>()));
        EXPECT_EQ(_instance.get<peak_rss>()->get_is_invalid(),
                  get_is_invalid_t<peak_rss>{}(*_instance.get<peak_rss>()));

        _instance.start();
        ret += details::fibonacci(35);
        _instance.stop();
    }

    printf("fibonacci(35) = %li\n", ret);

    auto wc_n = wc_size_orig + 1;
    auto cu_n = cu_size_orig + 1;
    auto cc_n = cc_size_orig + 2;
    auto pr_n = pr_size_orig + 1;

    EXPECT_EQ(tim::storage<wall_clock>::instance()->size(), wc_n);
    EXPECT_EQ(tim::storage<cpu_util>::instance()->size(), cu_n);
    EXPECT_EQ(tim::storage<cpu_clock>::instance()->size(), cc_n);
    EXPECT_EQ(tim::storage<peak_rss>::instance()->size(), pr_n);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, custom_type_count_w_init)
{
    using api_t                                   = CUSTOM_API;
    test<api_t>::direct_auto_t::get_initializer() = [](auto& cb) {
        // initialize one of two pointers and a type which does not belong
        cb.template initialize<page_rss, system_clock>();
    };

    auto A = test<api_t>::direct_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<api_t>::direct_auto"));
    auto C = test<api_t>::derive_auto_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::derive_auto"));

    test<api_t>::derive_comp_t::get_initializer() = [](auto& cb) {
        // initialize both pointers
        cb.template initialize<page_rss, user_clock>();
    };

    auto B = test<api_t>::direct_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::direct_comp"));
    auto D = test<api_t>::derive_comp_t(
        TIMEMORY_JOIN("/", details::get_test_name(), "test<CUSTOM_API>::derive_comp"));

    EXPECT_FALSE(tim::trait::is_available<api_t>::value);
    EXPECT_EQ(A.count(), 0);
    EXPECT_EQ(B.count(), 0);
    EXPECT_EQ(C.count(), 0);
    EXPECT_EQ(D.count(), 0);

    test<api_t>::derive_auto_t::get_initializer() = [](auto&) {};
    test<api_t>::direct_comp_t::get_initializer() = [](auto&) {};
}

//--------------------------------------------------------------------------------------//

using namespace tim::component;
namespace quirk = tim::quirk;

template <typename BundleT>
using template_stop_last_instance_t =
    tim::convert_t<tim::type_list<wall_clock, quirk::stop_last_bundle>, BundleT>;

template <typename BundleT>
using stop_last_instance_t = tim::convert_t<tim::type_list<wall_clock>, BundleT>;

namespace details
{
template <typename BundleT, typename NextT = BundleT, typename... Args>
void
run(Args&&... args)
{
    BundleT foo{ details::get_test_name(), std::forward<Args>(args)... };
    foo.start();

    consume(50);

    // foo should be stopped by this
    NextT bar{ details::get_test_name(), std::forward<Args>(args)... };
    bar.start();

    consume(50);

    bar.stop();

    // this should not be included in any timing
    consume(100);

    // this should have no effect
    foo.stop();
    // std::cout << tim::demangle<BundleT>() << std::endl;
    // std::cout << foo << std::endl;
    // std::cout << bar << std::endl;
}
//
auto
get_substr(std::string _str)
{
    auto idx = _str.find(">>> ");
    if(idx == std::string::npos)
        return _str;
    return _str.substr(idx + 4);
}
//
}  // namespace details

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, dont_stop_last_instance)
{
    auto wc_beg = tim::storage<wall_clock>::instance()->get();

    auto quirk_cfg    = quirk::config<>{};
    auto _initializer = [](auto& cl) { cl.template initialize<wall_clock>(); };

    details::run<stop_last_instance_t<tim::component_tuple<>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::component_bundle<TIMEMORY_API>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::component_list<>>>(quirk_cfg, _initializer);
    details::run<stop_last_instance_t<tim::auto_tuple<>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::auto_bundle<TIMEMORY_API>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::auto_list<>>>(quirk_cfg, _initializer);

    auto wc_end = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(wc_beg.size() + 2, wc_end.size());
    EXPECT_EQ(details::get_substr(wc_end.back().prefix()),
              std::string{ "|_" } + details::get_test_name());
    EXPECT_EQ(wc_end.back().depth(), 1);
    EXPECT_EQ(wc_end.back().data().get_laps(), 6);
    EXPECT_NEAR((wc_end.back().data().get() * wall_clock::get_unit()) / tim::units::msec,
                300., 100.);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, template_stop_last_instance)
{
    auto wc_beg = tim::storage<wall_clock>::instance()->get();

    auto _initializer = [](auto& cl) { cl.template initialize<wall_clock>(); };

    details::run<template_stop_last_instance_t<tim::component_tuple<>>>();
    details::run<template_stop_last_instance_t<tim::component_bundle<TIMEMORY_API>>>();
    details::run<template_stop_last_instance_t<tim::component_list<>>>(quirk::config<>{},
                                                                       _initializer);
    details::run<template_stop_last_instance_t<tim::auto_tuple<>>>();
    details::run<template_stop_last_instance_t<tim::auto_bundle<TIMEMORY_API>>>();
    details::run<template_stop_last_instance_t<tim::auto_list<>>>(quirk::config<>{},
                                                                  _initializer);

    auto wc_end = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(wc_beg.size() + 1, wc_end.size());
    EXPECT_EQ(details::get_substr(wc_end.back().prefix()), details::get_test_name());
    EXPECT_EQ(wc_end.back().depth(), 0);
    EXPECT_EQ(wc_end.back().data().get_laps(), 12);
    EXPECT_NEAR((wc_end.back().data().get() * wall_clock::get_unit()) / tim::units::msec,
                600., 200.);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, ctor_stop_last_instance)
{
    auto wc_beg = tim::storage<wall_clock>::instance()->get();

    auto quirk_cfg    = quirk::config<quirk::stop_last_bundle>{};
    auto _initializer = [](auto& cl) { cl.template initialize<wall_clock>(); };

    details::run<stop_last_instance_t<tim::component_tuple<>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::component_bundle<TIMEMORY_API>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::component_list<>>>(quirk_cfg, _initializer);
    details::run<stop_last_instance_t<tim::auto_tuple<>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::auto_bundle<TIMEMORY_API>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::auto_list<>>>(quirk_cfg, _initializer);

    auto wc_end = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(wc_beg.size() + 1, wc_end.size());
    EXPECT_EQ(details::get_substr(wc_end.back().prefix()), details::get_test_name());
    EXPECT_EQ(wc_end.back().depth(), 0);
    EXPECT_EQ(wc_end.back().data().get_laps(), 12);
    EXPECT_NEAR((wc_end.back().data().get() * wall_clock::get_unit()) / tim::units::msec,
                600., 200.);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, both_stop_last_instance)
{
    auto wc_beg = tim::storage<wall_clock>::instance()->get();

    auto quirk_cfg    = quirk::config<quirk::stop_last_bundle>{};
    auto _initializer = [](auto& cl) { cl.template initialize<wall_clock>(); };

    details::run<template_stop_last_instance_t<tim::component_tuple<>>>(quirk_cfg);
    details::run<template_stop_last_instance_t<tim::component_bundle<TIMEMORY_API>>>(
        quirk_cfg);
    details::run<template_stop_last_instance_t<tim::component_list<>>>(quirk_cfg,
                                                                       _initializer);
    details::run<template_stop_last_instance_t<tim::auto_tuple<>>>(quirk_cfg);
    details::run<template_stop_last_instance_t<tim::auto_bundle<TIMEMORY_API>>>(
        quirk_cfg);
    details::run<template_stop_last_instance_t<tim::auto_list<>>>(quirk_cfg,
                                                                  _initializer);

    auto wc_end = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(wc_beg.size() + 1, wc_end.size());
    EXPECT_EQ(details::get_substr(wc_end.back().prefix()), details::get_test_name());
    EXPECT_EQ(wc_end.back().depth(), 0);
    EXPECT_EQ(wc_end.back().data().get_laps(), 12);
    EXPECT_NEAR((wc_end.back().data().get() * wall_clock::get_unit()) / tim::units::msec,
                600., 200.);
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, mixed_stop_last_instance)
{
    auto wc_beg = tim::storage<wall_clock>::instance()->get();

    auto quirk_cfg    = quirk::config<>{};
    auto _initializer = [](auto& cl) { cl.template initialize<wall_clock>(); };

    details::run<stop_last_instance_t<tim::component_tuple<>>,
                 template_stop_last_instance_t<tim::component_tuple<>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::component_bundle<TIMEMORY_API>>,
                 template_stop_last_instance_t<tim::component_bundle<TIMEMORY_API>>>(
        quirk_cfg);
    details::run<stop_last_instance_t<tim::component_list<>>,
                 template_stop_last_instance_t<tim::component_list<>>>(quirk_cfg,
                                                                       _initializer);
    details::run<stop_last_instance_t<tim::auto_tuple<>>,
                 template_stop_last_instance_t<tim::auto_tuple<>>>(quirk_cfg);
    details::run<stop_last_instance_t<tim::auto_bundle<TIMEMORY_API>>,
                 template_stop_last_instance_t<tim::auto_bundle<TIMEMORY_API>>>(
        quirk_cfg);
    details::run<stop_last_instance_t<tim::auto_list<>>,
                 template_stop_last_instance_t<tim::auto_list<>>>(quirk_cfg,
                                                                  _initializer);

    details::run<template_stop_last_instance_t<tim::component_tuple<>>,
                 stop_last_instance_t<tim::component_tuple<>>>(quirk_cfg);
    details::run<template_stop_last_instance_t<tim::component_bundle<TIMEMORY_API>>,
                 stop_last_instance_t<tim::component_bundle<TIMEMORY_API>>>(quirk_cfg);
    details::run<template_stop_last_instance_t<tim::component_list<>>,
                 stop_last_instance_t<tim::component_list<>>>(quirk_cfg, _initializer);
    details::run<template_stop_last_instance_t<tim::auto_tuple<>>,
                 stop_last_instance_t<tim::auto_tuple<>>>(quirk_cfg);
    details::run<template_stop_last_instance_t<tim::auto_bundle<TIMEMORY_API>>,
                 stop_last_instance_t<tim::auto_bundle<TIMEMORY_API>>>(quirk_cfg);
    details::run<template_stop_last_instance_t<tim::auto_list<>>,
                 stop_last_instance_t<tim::auto_list<>>>(quirk_cfg, _initializer);

    auto wc_end = tim::storage<wall_clock>::instance()->get();

    EXPECT_EQ(wc_beg.size() + 2, wc_end.size());
    EXPECT_EQ(details::get_substr(wc_end.back().prefix()),
              std::string{ "|_" } + details::get_test_name());
    EXPECT_EQ(wc_end.back().depth(), 1);
    EXPECT_EQ(wc_end.back().data().get_laps(), 12);
    EXPECT_NEAR((wc_end.back().data().get() * wall_clock::get_unit()) / tim::units::msec,
                600., 200.);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
rekey(const std::string& _initial)
{
    std::random_device              rd;
    std::mt19937                    gen{ rd() };
    std::uniform_int_distribution<> distrib{ 0, std::numeric_limits<char>::max() };

    auto get_random_character = [&distrib, &gen]() {
        static std::array<char, 2> buffer{};
        memset(buffer.data(), '\0', buffer.size() * sizeof(char));

        char c = '\0';
        do
        {
            c = distrib(gen);
        } while(!std::isalnum(c));
        buffer[0] = c;
        return buffer.data();
    };

    auto _other = _initial + "/";
    for(int i = 0; i < 8; ++i)
        _other += get_random_character();

    auto _init_hash  = tim::add_hash_id(_initial);
    auto _other_hash = tim::add_hash_id(_other);

    auto _init_srcloc =
        TIMEMORY_SOURCE_LOCATION(TIMEMORY_CAPTURE_MODE(blank), _initial.c_str());
    auto _other_srcloc =
        TIMEMORY_SOURCE_LOCATION(TIMEMORY_CAPTURE_MODE(blank), _other.c_str());

    // std::cout << "[" << tim::demangle<Tp>() << "] initial: " << _initial
    //           << ", random: " << _other << std::endl;

    {
        Tp _obj{ _initial };
        EXPECT_EQ(_obj.key(), _initial) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_EQ(_obj.hash(), _init_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;

        _obj.rekey(_other);
        EXPECT_EQ(_obj.key(), _other) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_NE(_obj.key(), _initial) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_EQ(_obj.hash(), _other_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_NE(_obj.hash(), _init_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;
    }

    {
        Tp _obj{ _init_hash };
        EXPECT_EQ(_obj.key(), _initial) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_EQ(_obj.hash(), _init_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;

        _obj.rekey(_other_hash);
        EXPECT_EQ(_obj.key(), _other) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_NE(_obj.key(), _initial) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_EQ(_obj.hash(), _other_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_NE(_obj.hash(), _init_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;
    }

    {
        Tp _obj{ _init_srcloc.get_captured() };
        EXPECT_EQ(_obj.key(), _initial) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_EQ(_obj.hash(), _init_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;

        _obj.rekey(_other_srcloc.get_captured());
        EXPECT_EQ(_obj.key(), _other) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_NE(_obj.key(), _initial) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_EQ(_obj.hash(), _other_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;
        EXPECT_NE(_obj.hash(), _init_hash) << "[" << tim::demangle<Tp>() << "] " << _obj;
    }

    // std::cout << std::endl;
}

template <typename BundleT>
using rekey_bundle_t =
    tim::convert_t<tim::type_list<wall_clock, quirk::explicit_start>, BundleT>;

TEST_F(component_bundle_tests, rekey)
{
    rekey<rekey_bundle_t<tim::component_tuple<>>>(details::get_test_name());
    rekey<rekey_bundle_t<tim::component_list<>>>(details::get_test_name());
    rekey<rekey_bundle_t<tim::component_bundle<TIMEMORY_API>>>(details::get_test_name());
    rekey<rekey_bundle_t<tim::lightweight_tuple<>>>(details::get_test_name());

    rekey<rekey_bundle_t<tim::auto_tuple<>>>(details::get_test_name());
    rekey<rekey_bundle_t<tim::auto_list<>>>(details::get_test_name());
    rekey<rekey_bundle_t<tim::auto_bundle<TIMEMORY_API>>>(details::get_test_name());
}

//--------------------------------------------------------------------------------------//

TEST_F(component_bundle_tests, piecewise_ignore)
{
    auto wc_beg = tim::storage<wall_clock>::instance()->get();
    auto cc_beg = tim::storage<cpu_clock>::instance()->get();
    auto tc_beg = tim::storage<trip_count>::instance()->get();

    static_assert(
        std::is_same<tim::mpl::subtract_t<
                         tim::mpl::implemented_t<wall_clock, cpu_clock, trip_count>,
                         tim::type_list<wall_clock, trip_count>>,
                     tim::type_list<cpu_clock>>::value,
        "Subtract error");

    static auto _ign = tim::mpl::piecewise_ignore<wall_clock, trip_count>{};
    auto        _run = [](auto& _obj) {
        _obj.start(_ign);
        details::consume(500);
        _obj.stop(_ign);
    };

    tim::component_tuple<wall_clock, cpu_clock, trip_count> _ct{
        details::get_test_name()
    };
    tim::component_list<wall_clock, cpu_clock, trip_count> _cl{
        details::get_test_name(), tim::scope::get_default(),
        [](auto& _v) { _v.template initialize<wall_clock, cpu_clock, trip_count>(); }
    };
    tim::component_bundle<TIMEMORY_API, wall_clock, cpu_clock, trip_count> _cb{
        details::get_test_name()
    };
    tim::lightweight_tuple<wall_clock, cpu_clock, trip_count> _lt{
        details::get_test_name()
    };

    _run(_ct);
    _run(_cl);
    _run(_cb);
    _run(_lt);

    ASSERT_TRUE(_cl.get<wall_clock>() != nullptr);
    ASSERT_TRUE(_cl.get<cpu_clock>() != nullptr);
    ASSERT_TRUE(_cl.get<trip_count>() != nullptr);

    auto _check = [](auto& _obj) {
        auto _name = tim::demangle<decltype(_obj)>();
        ASSERT_TRUE(_obj.template get<wall_clock>() != nullptr) << _name;
        ASSERT_TRUE(_obj.template get<cpu_clock>() != nullptr) << _name;
        ASSERT_TRUE(_obj.template get<trip_count>() != nullptr) << _name;

        EXPECT_EQ(_obj.template get<wall_clock>()->get_laps(), 0) << _name << " " << _obj;
        EXPECT_EQ(_obj.template get<cpu_clock>()->get_laps(), 1) << _name << " " << _obj;
        EXPECT_EQ(_obj.template get<trip_count>()->get_laps(), 0) << _name << " " << _obj;
    };

    _check(_ct);
    _check(_cl);
    _check(_cb);
    _check(_lt);

    auto wc_end = tim::storage<wall_clock>::instance()->get();
    auto cc_end = tim::storage<cpu_clock>::instance()->get();
    auto tc_end = tim::storage<trip_count>::instance()->get();

    EXPECT_EQ(wc_beg.size(), wc_end.size());
    EXPECT_NE(cc_beg.size(), cc_end.size());
    EXPECT_EQ(tc_beg.size(), tc_end.size());
}

//--------------------------------------------------------------------------------------//
