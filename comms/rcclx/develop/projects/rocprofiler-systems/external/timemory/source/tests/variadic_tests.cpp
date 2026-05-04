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

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#if defined(TIMEMORY_USE_PAPI)
#    undef TIMEMORY_USE_PAPI
#endif

#if defined(TIMEMORY_USE_CUPTI)
#    undef TIMEMORY_USE_CUPTI
#endif

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/component_hybrid.hpp"
#include "timemory/variadic/functional.hpp"

#include "gtest/gtest.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace tim::component;

//--------------------------------------------------------------------------------------//

namespace details
{
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

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

class variadic_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(variadic_tests, variadic)
{
    static constexpr size_t nz = 7;
    std::array<size_t, nz>  sizes;
    size_t                  n = 0;
    std::generate(sizes.begin(), sizes.end(), [&]() { return n++; });

    {
        using tuple_t  = tim::auto_tuple<wall_clock, peak_rss>;
        using list_t   = tim::auto_list<cpu_roofline_dp_flops, gpu_roofline_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[0]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::auto_list<cpu_roofline_dp_flops, gpu_roofline_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[1]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<cpu_roofline_dp_flops, gpu_roofline_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[2]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<cpu_roofline_dp_flops, gpu_roofline_flops,
                                           cpu_roofline_sp_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[3]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<gpu_roofline_dp_flops, cpu_roofline_flops,
                                           gpu_roofline_sp_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[4]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<gpu_roofline_dp_flops, gpu_roofline_flops,
                                           gpu_roofline_sp_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[5]       = hybrid_t::size();
    }

    {
        using tuple_t  = tim::component_tuple<wall_clock, peak_rss>;
        using list_t   = tim::component_list<cpu_roofline_dp_flops, cpu_roofline_flops,
                                           cpu_roofline_sp_flops>;
        using hybrid_t = tim::auto_hybrid<tuple_t, list_t>;
        sizes[6]       = hybrid_t::size();
    }

    // std::cout << "\n";
    // for(size_t i = 0; i < nz; ++i)
    //     std::cout << "size[" << i << "] = " << sizes[i] << std::endl;
    // std::cout << "\n";

    EXPECT_EQ(sizes[0], sizes[1]);
    EXPECT_EQ(sizes[0], sizes[2]);
    EXPECT_EQ(sizes[1], sizes[2]);

    for(size_t i = 3; i < nz; ++i)
    {
        ASSERT_LE(sizes[0], sizes[i]) << "i = " << i;
        ASSERT_LE(sizes[1], sizes[i]) << "i = " << i;
        ASSERT_LE(sizes[2], sizes[i]) << "i = " << i;
    }

    using tup_t    = tim::component_tuple<wall_clock, cpu_clock>;
    using lst_t    = tim::component_list<peak_rss, page_rss>;
    using hybrid_t = tim::auto_hybrid<tup_t, lst_t>;

    auto hsize = hybrid_t::size();
    // auto tsize = hybrid_t::tuple_t::size();
    // auto lsize = hybrid_t::list_t::size();

    // std::cout << "\nhybrid        : " << tim::demangle<hybrid_t>() << "\n";
    // std::cout << "\nhybrid tuple  : " << tim::demangle<typename hybrid_t::tuple_t>();
    // std::cout << "\nhybrid list   : " << tim::demangle<typename hybrid_t::list_t>();
    // std::cout << "\n";
    {
        hybrid_t hybrid(details::get_test_name());
        auto     ret = details::fibonacci(41);
        printf("\nfibonacci(41) = %li\n\n", (long int) ret);
    }

    EXPECT_EQ(hsize, 4);
    // EXPECT_EQ(tsize, 2);
    // EXPECT_EQ(lsize, 2);
}

//--------------------------------------------------------------------------------------//

TEST_F(variadic_tests, concat)
{
    using lhs_t = tim::component_tuple<wall_clock, system_clock>;
    using rhs_t = tim::component_tuple<wall_clock, cpu_clock>;

    using join_t0 = typename tim::component_tuple<lhs_t, rhs_t>::type;
    using join_t1 = typename tim::auto_tuple<lhs_t, rhs_t, user_clock>::type;
    using join_t2 =
        typename tim::auto_tuple<lhs_t, tim::component_list<rhs_t, user_clock>>::type;

    using comp_t0 = tim::mpl::remove_duplicates_t<join_t0>;
    using comp_t1 = tim::mpl::remove_duplicates_t<join_t1>;
    using comp_t2 = tim::mpl::remove_duplicates_t<join_t2>;

    using lhs_l = tim::convert_t<lhs_t, tim::component_list<>>;
    using rhs_l = tim::convert_t<rhs_t, tim::component_list<>>;

    using dbeg_t0 = typename tim::component_list<lhs_l, rhs_l>::data_type;
    using dbeg_t1 = typename tim::auto_list<lhs_l, rhs_l, user_clock>::data_type;
    using dbeg_t2 = typename tim::auto_list<rhs_l, user_clock>::data_type;
    using dbeg_t3 = typename tim::auto_list<lhs_l, dbeg_t2>::data_type;

    using data_t0 = tim::mpl::remove_duplicates_t<dbeg_t0>;
    using data_t1 = tim::mpl::remove_duplicates_t<dbeg_t1>;
    // using data_t2 = tim::mpl::remove_duplicates_t<dbeg_t2>;
    using data_t3 = tim::mpl::remove_duplicates_t<dbeg_t3>;

    // std::cout << "\n" << std::flush;

    // std::cout << "lhs_t = " << tim::demangle<lhs_t>() << "\n";
    // std::cout << "rhs_t = " << tim::demangle<rhs_t>() << "\n";
    // std::cout << "lhs_l = " << tim::demangle<lhs_l>() << "\n";
    // std::cout << "rhs_l = " << tim::demangle<rhs_l>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "join_t0 = " << tim::demangle<join_t0>() << "\n";
    // std::cout << "join_t1 = " << tim::demangle<join_t1>() << "\n";
    // std::cout << "join_t2 = " << tim::demangle<join_t2>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "comp_t0 = " << tim::demangle<comp_t0>() << "\n";
    // std::cout << "comp_t1 = " << tim::demangle<comp_t1>() << "\n";
    // std::cout << "comp_t2 = " << tim::demangle<comp_t2>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "dbeg_t0 = " << tim::demangle<dbeg_t0>() << "\n";
    // std::cout << "dbeg_t1 = " << tim::demangle<dbeg_t1>() << "\n";
    // std::cout << "dbeg_t2 = " << tim::demangle<dbeg_t2>() << "\n";
    // std::cout << "dbeg_t3 = " << tim::demangle<dbeg_t3>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "data_t0 = " << tim::demangle<data_t0>() << "\n";
    // std::cout << "data_t1 = " << tim::demangle<data_t1>() << "\n";
    // std::cout << "data_t2 = " << tim::demangle<data_t2>() << "\n";
    // std::cout << "data_t3 = " << tim::demangle<data_t3>() << "\n";
    // std::cout << "\n" << std::flush;

    EXPECT_EQ(comp_t0::size(), 3);
    EXPECT_EQ(comp_t1::size(), 4);
    EXPECT_EQ(comp_t2::size(), 4);

    EXPECT_EQ(std::tuple_size<data_t0>::value, 3);
    EXPECT_EQ(std::tuple_size<data_t1>::value, 4);
    EXPECT_EQ(std::tuple_size<data_t3>::value, 4);
}

//--------------------------------------------------------------------------------------//

TEST_F(variadic_tests, get)
{
    tim::trait::runtime_enabled<cpu_roofline<float>>::set(false);
    tim::trait::runtime_enabled<cpu_roofline<double>>::set(false);

    using lhs_t = tim::component_tuple<wall_clock, system_clock, cpu_roofline<double>>;
    using rhs_t = tim::component_tuple<wall_clock, cpu_clock, cpu_roofline<float>>;
    using lhs_l = tim::convert_t<lhs_t, tim::component_list<>>;
    using rhs_l = tim::convert_t<rhs_t, tim::component_list<>>;

    using join_t0 = typename tim::component_tuple<lhs_t, rhs_t>::type;
    using join_t1 = typename tim::auto_tuple<lhs_t, rhs_t, user_clock>::type;
    using join_t2 =
        typename tim::auto_tuple<lhs_t, tim::component_list<rhs_t, user_clock>>::type;

    using comp_t0 = tim::mpl::remove_duplicates_t<join_t0>;
    using comp_t1 = tim::mpl::remove_duplicates_t<join_t1>;
    using comp_t2 = tim::mpl::remove_duplicates_t<join_t2>;

    using comp_t2_base_type      = typename comp_t2::base_type;
    using comp_t2_component_type = typename comp_t2::component_type;
    using comp_t2_tuple_type     = typename comp_t2::tuple_type;
    using comp_t2_data_type      = typename comp_t2::data_type;
    using comp_t2_bundle_type    = typename comp_t2::bundle_type;
    using comp_t2_this_type      = typename comp_t2::this_type;

#define PRINT_TYPE(TYPE)                                                                 \
    std::cout << std::right << std::setw(24) << #TYPE << " = " << tim::demangle<TYPE>()  \
              << "\n";

    std::cout << "\n" << std::flush;
    PRINT_TYPE(comp_t2_base_type)
    PRINT_TYPE(comp_t2_component_type)
    PRINT_TYPE(comp_t2_tuple_type)
    PRINT_TYPE(comp_t2_data_type)
    PRINT_TYPE(comp_t2_bundle_type)
    PRINT_TYPE(comp_t2_this_type)
    std::cout << "\n" << std::flush;

    using list_t0 = tim::mpl::remove_duplicates_t<tim::component_list_t<lhs_l, rhs_l>>;
    using list_t1 =
        tim::mpl::remove_duplicates_t<tim::auto_list_t<lhs_l, rhs_l, user_clock>>;
    using list_t2 = tim::mpl::remove_duplicates_t<tim::auto_list_t<rhs_l, user_clock>>;
    using list_t3 = tim::mpl::remove_duplicates_t<tim::auto_list_t<lhs_l, list_t2>>;

    // std::cout << "\n" << std::flush;

    // std::cout << "lhs_t = " << tim::demangle<lhs_t>() << "\n";
    // std::cout << "rhs_t = " << tim::demangle<rhs_t>() << "\n";
    // std::cout << "lhs_l = " << tim::demangle<lhs_l>() << "\n";
    // std::cout << "rhs_l = " << tim::demangle<rhs_l>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "join_t0 = " << tim::demangle<join_t0>() << "\n";
    // std::cout << "join_t1 = " << tim::demangle<join_t1>() << "\n";
    // std::cout << "join_t2 = " << tim::demangle<join_t2>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "comp_t0 = " << tim::demangle<comp_t0>() << "\n";
    // std::cout << "comp_t1 = " << tim::demangle<comp_t1>() << "\n";
    // std::cout << "comp_t2 = " << tim::demangle<comp_t2>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "list_t0 = " << tim::demangle<list_t0>() << "\n";
    // std::cout << "list_t1 = " << tim::demangle<list_t1>() << "\n";
    // std::cout << "list_t2 = " << tim::demangle<list_t2>() << "\n";
    // std::cout << "list_t3 = " << tim::demangle<list_t3>() << "\n";
    // std::cout << "\n" << std::flush;

    list_t0::get_initializer() = [](auto& cl) { cl.template initialize<wall_clock>(); };
    list_t1::get_initializer() = [](auto& cl) { cl.template initialize<wall_clock>(); };
    list_t2::get_initializer() = [](auto& cl) { cl.template initialize<wall_clock>(); };
    list_t3::get_initializer() = [](auto& cl) { cl.template initialize<wall_clock>(); };

    auto ct0 = comp_t0("ct0");
    auto ct1 = comp_t1("ct1");
    auto ct2 = comp_t2("ct2");
    auto cl0 = list_t0("cl0");
    auto cl1 = list_t1("cl1");
    auto cl2 = list_t2("cl2");
    auto cl3 = list_t3("cl3");

    // auto ct3 =
    //    tim::auto_tuple<tim::component::wall_clock, tim::component::system_clock,
    //                    tim::component::cpu_clock, tim::component::user_clock>{ "ct3" };

    namespace disjoint = tim::invoke::disjoint;

    disjoint::start(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));
    disjoint::mark_begin(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    disjoint::store(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3), 10);
    disjoint::mark(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3), 10);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    disjoint::record(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3), 10);
    disjoint::measure(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    disjoint::mark_end(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));
    disjoint::stop(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    auto dt0 = ct0.get();
    auto dt1 = ct1.get();
    auto dt2 = ct2.get();

    auto dl0 = cl0.get();
    auto dl1 = cl1.get();
    auto dl2 = cl2.get();
    auto dl3 = cl3.get();

    // std::cout << "dt0 = " << tim::demangle<decltype(dt0)>() << "\n";
    // std::cout << "dt1 = " << tim::demangle<decltype(dt1)>() << "\n";
    // std::cout << "dt2 = " << tim::demangle<decltype(dt2)>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "dl0 = " << tim::demangle<decltype(dl0)>() << "\n";
    // std::cout << "dl1 = " << tim::demangle<decltype(dl1)>() << "\n";
    // std::cout << "dl2 = " << tim::demangle<decltype(dl2)>() << "\n";
    // std::cout << "dl3 = " << tim::demangle<decltype(dl3)>() << "\n";
    // std::cout << "\n" << std::flush;

    tim::invoke::print(std::cout, ct0, ct1, ct2, cl0, cl1, cl2, cl3);

    // std::cout << '\n';
    disjoint::invoke(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3),
                     [](auto& itr) { std::cout << itr << std::endl; });

#if defined(TIMEMORY_MACOS)
    // GitHub CI for macOS is very noisy
    constexpr auto dtolerance = 0.15;
#else
    constexpr auto dtolerance = 0.10;
#endif

    // std::cout << "\nCheck get<0>()" << std::endl;
    EXPECT_NEAR(std::get<0>(dt0), 1.0, dtolerance);
    EXPECT_NEAR(std::get<0>(dt1), 1.0, dtolerance);
    EXPECT_NEAR(std::get<0>(dt2), 1.0, dtolerance);

    EXPECT_NEAR(std::get<0>(dl0), 1.0, dtolerance);
    EXPECT_NEAR(std::get<0>(dl1), 1.0, dtolerance);
    EXPECT_NEAR(std::get<0>(dl2), 1.0, dtolerance);
    EXPECT_NEAR(std::get<0>(dl3), 1.0, dtolerance);

    // std::cout << "disjoint::reset" << std::endl;
    disjoint::reset(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    // std::cout << "disjoint::set_prefix" << std::endl;
    disjoint::set_prefix(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3), "set_prefix");

    // std::cout << "disjoint::set_scope" << std::endl;
    disjoint::set_scope(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3),
                        tim::scope::flat{} + tim::scope::timeline{});

    // std::cout << "disjoint::push" << std::endl;
    disjoint::push(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    // std::cout << "disjoint::assemble" << std::endl;
    disjoint::assemble(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    // std::cout << "disjoint::audit" << std::endl;
    disjoint::audit(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3), 10);

    // std::cout << "disjoint::derive" << std::endl;
    disjoint::derive(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    // std::cout << "disjoint::add_secondary" << std::endl;
    disjoint::add_secondary(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    // std::cout << "disjoint::pop" << std::endl;
    disjoint::pop(std::tie(ct0, ct1, ct2, cl0, cl1, cl2, cl3));

    // std::cout << "\nCheck get()" << std::endl;
    dt0 = ct0.get();
    dt1 = ct1.get();
    dt2 = ct2.get();

    dl0 = cl0.get();
    dl1 = cl1.get();
    dl2 = cl2.get();
    dl3 = cl3.get();

    EXPECT_NEAR(std::get<0>(dt0), 0.0, 1.e-3);
    EXPECT_NEAR(std::get<0>(dt1), 0.0, 1.e-3);
    EXPECT_NEAR(std::get<0>(dt2), 0.0, 1.e-3);

    EXPECT_NEAR(std::get<0>(dl0), 0.0, 1.e-3);
    EXPECT_NEAR(std::get<0>(dl1), 0.0, 1.e-3);
    EXPECT_NEAR(std::get<0>(dl2), 0.0, 1.e-3);
    EXPECT_NEAR(std::get<0>(dl3), 0.0, 1.e-3);

    // std::cout << "Done" << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(variadic_tests, piecewise_ignore)
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

    static auto _ign = tim::mpl::piecewise_ignore<cpu_clock, trip_count>{};
    auto        _run = [](auto& _obj) {
        auto _name = tim::demangle<decltype(_obj)>();

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

        EXPECT_NEAR(_obj.template get<wall_clock>()->get(), 0.5, 0.2)
            << _name << " " << _obj;
        EXPECT_NEAR(_obj.template get<cpu_clock>()->get(), 0.0, 1.0e-6)
            << _name << " " << _obj;
        EXPECT_EQ(_obj.template get<trip_count>()->get(), 0) << _name << " " << _obj;
    };

    _check(_ct);
    _check(_cl);
    _check(_cb);
    _check(_lt);

    auto wc_end = tim::storage<wall_clock>::instance()->get();
    auto cc_end = tim::storage<cpu_clock>::instance()->get();
    auto tc_end = tim::storage<trip_count>::instance()->get();

    EXPECT_NE(wc_beg.size(), wc_end.size());
    EXPECT_EQ(cc_beg.size(), cc_end.size());
    EXPECT_EQ(tc_beg.size(), tc_end.size());
}

//--------------------------------------------------------------------------------------//
