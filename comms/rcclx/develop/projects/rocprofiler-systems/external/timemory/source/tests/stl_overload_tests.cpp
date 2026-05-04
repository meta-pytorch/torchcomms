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
#include "timemory/math/assign.hpp"
#include "timemory/mpl/concepts.hpp"

#include <cstdint>
#include <limits>
#include <sstream>

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using namespace tim;
using namespace tim::stl;
using namespace tim::stl::ostream;

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
inline TIMEMORY_HOT void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds{ n });
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class stl_overload_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

#define GET(VAR, N) std::get<N>(VAR)

constexpr size_t N         = 2;
const double     tolerance = static_cast<double>(std::numeric_limits<float>::epsilon());

using int_type    = int;
using real_type   = double;
using vector_type = std::vector<real_type>;
using array_type  = std::array<int_type, N>;
using pair_type   = std::pair<vector_type, array_type>;
using tuple_type  = std::tuple<int_type, real_type, pair_type>;

//--------------------------------------------------------------------------------------//

TEST_F(stl_overload_tests, divide)
{
    int_type    int_v    = 4;
    real_type   real_v   = 4.0;
    vector_type vector_v = { real_v, 1.5 * real_v };
    array_type  array_v  = { { int_v, 3 * int_v } };
    pair_type   pair_v   = { vector_v, array_v };

    int_type    int_op    = 2;
    real_type   real_op   = 2.0;
    vector_type vector_op = { real_op, real_op };
    array_type  array_op  = { { int_op, int_op } };
    pair_type   pair_op   = { vector_op, array_op };

    tuple_type init_v{ int_v, real_v, pair_v };
    tuple_type init_op{ int_op, real_op, pair_op };

    tuple_type same_ret = init_v / init_op;
    tuple_type fund_ret = init_v / 2;

    std::stringstream ss;
    ss.precision(2);
    ss.width(4);
    ss.setf(std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint);
    ss << '\n';
    ss << "INIT VAL : " << init_v << '\n';
    ss << "INIT OP  : " << init_op << '\n';
    ss << "SAME RET : " << same_ret << '\n';
    ss << "FUND RET : " << fund_ret << '\n';
    ss << '\n';
    std::cout << ss.str();

    // check int
    ASSERT_EQ(GET(same_ret, 0), 2);
    // check double
    ASSERT_NEAR(GET(same_ret, 1), 2.0, tolerance);
    // check vector
    ASSERT_NEAR(GET(same_ret, 2).first[0], 2.0, tolerance);
    ASSERT_NEAR(GET(same_ret, 2).first[1], 3.0, tolerance);
    // check array
    ASSERT_EQ(GET(same_ret, 2).second[0], 2);
    ASSERT_EQ(GET(same_ret, 2).second[1], 6);

    // check int
    ASSERT_EQ(GET(fund_ret, 0), 2);
    // check double
    ASSERT_NEAR(GET(fund_ret, 1), 2.0, tolerance);
    // check vector
    ASSERT_NEAR(GET(fund_ret, 2).first[0], 2.0, tolerance);
    ASSERT_NEAR(GET(fund_ret, 2).first[1], 3.0, tolerance);
    // check array
    ASSERT_EQ(GET(fund_ret, 2).second[0], 2);
    ASSERT_EQ(GET(fund_ret, 2).second[1], 6);
}

//--------------------------------------------------------------------------------------//

TEST_F(stl_overload_tests, multiply)
{
    int_type    int_v    = 4;
    real_type   real_v   = 4.0;
    vector_type vector_v = { real_v, 1.5 * real_v };
    array_type  array_v  = { { int_v, 3 * int_v } };
    pair_type   pair_v   = { vector_v, array_v };

    int_type    int_op    = 2;
    real_type   real_op   = 2.0;
    vector_type vector_op = { real_op, real_op };
    array_type  array_op  = { { int_op, int_op } };
    pair_type   pair_op   = { vector_op, array_op };

    tuple_type init_v{ int_v, real_v, pair_v };
    tuple_type init_op{ int_op, real_op, pair_op };

    tuple_type same_ret = init_v * init_op;
    tuple_type fund_ret = init_v * 2;

    std::stringstream ss;
    ss.precision(2);
    ss.width(4);
    ss.setf(std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint);
    ss << '\n';
    ss << "INIT VAL : " << init_v << '\n';
    ss << "INIT OP  : " << init_op << '\n';
    ss << "SAME RET : " << same_ret << '\n';
    ss << "FUND RET : " << fund_ret << '\n';
    ss << '\n';
    std::cout << ss.str();

    // check int
    ASSERT_EQ(GET(same_ret, 0), 8);
    // check double
    ASSERT_NEAR(GET(same_ret, 1), 8.0, tolerance);
    // check vector
    ASSERT_NEAR(GET(same_ret, 2).first[0], 8.0, tolerance);
    ASSERT_NEAR(GET(same_ret, 2).first[1], 12.0, tolerance);
    // check array
    ASSERT_EQ(GET(same_ret, 2).second[0], 8);
    ASSERT_EQ(GET(same_ret, 2).second[1], 24);

    // check int
    ASSERT_EQ(GET(fund_ret, 0), 8);
    // check double
    ASSERT_NEAR(GET(fund_ret, 1), 8.0, tolerance);
    // check vector
    ASSERT_NEAR(GET(fund_ret, 2).first[0], 8.0, tolerance);
    ASSERT_NEAR(GET(fund_ret, 2).first[1], 12.0, tolerance);
    // check array
    ASSERT_EQ(GET(fund_ret, 2).second[0], 8);
    ASSERT_EQ(GET(fund_ret, 2).second[1], 24);
}

//--------------------------------------------------------------------------------------//

TEST_F(stl_overload_tests, statistics)
{
    int_type    int_v    = 4;
    real_type   real_v   = 4.0;
    vector_type vector_v = { real_v, 1.5 * real_v };
    array_type  array_v  = { { int_v, 3 * int_v } };
    pair_type   pair_v   = { vector_v, array_v };

    int_type    int_op    = 2;
    real_type   real_op   = 2.0;
    vector_type vector_op = { real_op, real_op };
    array_type  array_op  = { { int_op, int_op } };
    pair_type   pair_op   = { vector_op, array_op };

    auto apply = [](statistics<tuple_type>& stat_v, const tuple_type& init_v) {
        std::stringstream ss;
        ss.precision(2);
        ss.width(4);
        ss.setf(std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint);

        stat_v += init_v;

        tuple_type sum_ret  = stat_v.get_sum();
        tuple_type sqr_ret  = stat_v.get_sqr();
        tuple_type mean_ret = stat_v.get_mean();
        tuple_type min_ret  = stat_v.get_min();
        tuple_type max_ret  = stat_v.get_max();
        tuple_type var_ret  = stat_v.get_variance();
        tuple_type std_ret  = stat_v.get_stddev();

        ss << '\n';
        ss << "ANSWER   :\n";
        ss << "    INIT : " << init_v << '\n';
        ss << "     SUM : " << sum_ret << '\n';
        ss << "     SQR : " << sqr_ret << '\n';
        ss << "    MEAN : " << mean_ret << '\n';
        ss << "     MIN : " << min_ret << '\n';
        ss << "     MAX : " << max_ret << '\n';
        ss << "     VAR : " << var_ret << '\n';
        ss << "  STDDEV : " << std_ret << '\n';
        ss << "    DATA : " << stat_v << '\n';

        std::cout << ss.str();
    };

    tuple_type init_v{ int_v, real_v, pair_v };
    tuple_type init_op{ int_op, real_op, pair_op };

    statistics<tuple_type> stat_v;

    stat_v += init_v;
    stat_v -= init_v;

    tuple_type sum_ret  = stat_v.get_sum();
    tuple_type sqr_ret  = stat_v.get_sqr();
    tuple_type mean_ret = stat_v.get_mean();
    tuple_type min_ret  = stat_v.get_min();
    tuple_type max_ret  = stat_v.get_max();
    tuple_type var_ret  = stat_v.get_variance();
    tuple_type std_ret  = stat_v.get_stddev();

    std::stringstream ss;
    ss.precision(2);
    ss.width(4);
    ss.setf(std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint);
    ss << '\n';
    ss << "INITIAL  :\n";
    ss << "     SUM : " << sum_ret << '\n';
    ss << "     SQR : " << sqr_ret << '\n';
    ss << "    MEAN : " << mean_ret << '\n';
    ss << "     MIN : " << min_ret << '\n';
    ss << "     MAX : " << max_ret << '\n';
    ss << "     VAR : " << var_ret << '\n';
    ss << "  STDDEV : " << std_ret << '\n';
    ss << "    DATA : " << stat_v << '\n';
    std::cout << ss.str();

    apply(stat_v, init_op);
    for(int i = 0; i < 10; ++i)
        apply(stat_v, init_v);

    std::cout << '\n';
}

//--------------------------------------------------------------------------------------//
#if defined(CXX17)

template <typename... Tp>
auto get_variant(type_list<Tp...>)
{
    return std::variant<Tp...>{};
}

template <typename Up, typename... Tp>
auto
get_variant(type_list<Tp...>, Up _v)
{
    std::variant<Tp...> _var;
    return math::assign(_var, _v);
}

template <typename Up, typename... Tp>
auto
test_variant(int line_no, const std::variant<Tp...>& _var, Up _val,
             double _tol = 2.0f * std::numeric_limits<float>::round_error())
{
    using types           = type_list<Tp...>;
    constexpr auto _index = tim::index_of<Up, types>::value;

    auto _orig = std::string{};
    std::visit(
        [&_orig](auto _v) {
            _orig = TIMEMORY_JOIN("", std::setprecision(3), std::fixed, _v);
        },
        _var);

    auto _msg = TIMEMORY_JOIN("", "[", details::get_test_name(), "][", __FILE__, ":",
                              line_no, "] original value: ", _orig, " :: ");

    EXPECT_EQ(_var.index(), _index)
        << _msg << "type: " << demangle<Up>() << ", types: " << demangle<types>();

    auto _func = [_val, _tol, _msg](auto _v) {
        using type    = concepts::unqualified_type_t<decltype(_v)>;
        auto _is_same = std::is_same<Up, type>::value;
        ASSERT_TRUE(_is_same) << _msg << "Up: " << demangle<Up>()
                              << ", type: " << demangle<type>();
        if constexpr(std::is_integral<Up>::value)
        {
            EXPECT_EQ(_v, _val)
                << _msg << "Up: " << demangle<Up>() << ", type: " << demangle<type>();
        }
        else
        {
            EXPECT_NEAR(_v, _val, _tol)
                << _msg << "Up: " << demangle<Up>() << ", type: " << demangle<type>();
        }
        (void) _tol;
    };

    std::visit(_func, _var);
}

#    define TEST_VARIANT(TYPE, ...) test_variant<TYPE>(__LINE__, __VA_ARGS__)

TEST_F(stl_overload_tests, variant)
{
    using types = type_list<int32_t, uint32_t, float, int64_t, uint64_t, double>;
    using variant_type =
        std::variant<int32_t, uint32_t, float, int64_t, uint64_t, double>;

    using value_type = concepts::unqualified_type_t<decltype(get_variant(types{}))>;
    auto _is_same    = std::is_same<variant_type, value_type>::value;
    ASSERT_TRUE(_is_same) << "variant_type: " << tim::demangle<variant_type>()
                          << ", value_type: " << tim::demangle<value_type>();

    constexpr size_t _size = std::tuple_size<types>::value;
    ASSERT_EQ(_size, 6);

    auto _data  = std::array<value_type, _size>{};
    _data.at(0) = get_variant<int32_t>(types{}, -1);
    _data.at(1) = get_variant<uint32_t>(types{}, -1);
    _data.at(2) = get_variant<float>(types{}, -1.0);
    _data.at(3) = get_variant<int64_t>(types{}, -1);
    _data.at(4) = get_variant<uint64_t>(types{}, -1);
    _data.at(5) = get_variant<double>(types{}, -1.0f);

    TEST_VARIANT(int32_t, _data.at(0), -1);
    TEST_VARIANT(uint32_t, _data.at(1), std::numeric_limits<uint32_t>::max());
    TEST_VARIANT(float, _data.at(2), -1.0f);
    TEST_VARIANT(int64_t, _data.at(3), -1);
    TEST_VARIANT(uint64_t, _data.at(4), std::numeric_limits<uint64_t>::max());
    TEST_VARIANT(double, _data.at(5), -1.0);

    TEST_VARIANT(int32_t, math::abs(_data.at(0)), 1);
    TEST_VARIANT(uint32_t, math::abs(_data.at(1)), std::numeric_limits<uint32_t>::max());
    TEST_VARIANT(float, math::abs(_data.at(2)), 1.0f);
    TEST_VARIANT(int64_t, math::abs(_data.at(3)), 1);
    TEST_VARIANT(uint64_t, math::abs(_data.at(4)), std::numeric_limits<uint64_t>::max());
    TEST_VARIANT(double, math::abs(_data.at(5)), 1.0);

    auto _reset_values = [](auto& _d, auto _v) {
        _d.at(0) = get_variant<int32_t>(types{}, _v);
        _d.at(1) = get_variant<uint32_t>(types{}, _v);
        _d.at(2) = get_variant<float>(types{}, _v);
        _d.at(3) = get_variant<int64_t>(types{}, _v);
        _d.at(4) = get_variant<uint64_t>(types{}, _v);
        _d.at(5) = get_variant<double>(types{}, _v);
    };

    _reset_values(_data, 9.0);

    TEST_VARIANT(int32_t, _data.at(0), 9);
    TEST_VARIANT(uint32_t, _data.at(1), 9);
    TEST_VARIANT(float, _data.at(2), 9.0f);
    TEST_VARIANT(int64_t, _data.at(3), 9);
    TEST_VARIANT(uint64_t, _data.at(4), 9);
    TEST_VARIANT(double, _data.at(5), 9.0);

    TEST_VARIANT(int32_t, math::sqr(_data.at(0)), 81);
    TEST_VARIANT(uint32_t, math::sqr(_data.at(1)), 81);
    TEST_VARIANT(float, math::sqr(_data.at(2)), 81.0f);
    TEST_VARIANT(int64_t, math::sqr(_data.at(3)), 81);
    TEST_VARIANT(uint64_t, math::sqr(_data.at(4)), 81);
    TEST_VARIANT(double, math::sqr(_data.at(5)), 81.0);

    TEST_VARIANT(int32_t, math::sqrt(_data.at(0)), 3);
    TEST_VARIANT(uint32_t, math::sqrt(_data.at(1)), 3);
    TEST_VARIANT(float, math::sqrt(_data.at(2)), 3.0f);
    TEST_VARIANT(int64_t, math::sqrt(_data.at(3)), 3);
    TEST_VARIANT(uint64_t, math::sqrt(_data.at(4)), 3);
    TEST_VARIANT(double, math::sqrt(_data.at(5)), 3.0);

    TEST_VARIANT(int32_t, math::pow(_data.at(0), 3), 729);
    TEST_VARIANT(uint32_t, math::pow(_data.at(1), 3), 729);
    TEST_VARIANT(float, math::pow(_data.at(2), 3), 729.0f);
    TEST_VARIANT(int64_t, math::pow(_data.at(3), 3), 729);
    TEST_VARIANT(uint64_t, math::pow(_data.at(4), 3), 729);
    TEST_VARIANT(double, math::pow(_data.at(5), 3), 729.0);

    auto _data_min = _data;
    _reset_values(_data_min, -729.0);

    TEST_VARIANT(int32_t, math::min(_data.at(0), _data_min.at(0)), -729);
    // TEST_VARIANT(uint32_t, math::min(_data.at(1), _data_min.at(1)), 9);
    TEST_VARIANT(float, math::min(_data.at(2), _data_min.at(2)), -729.0f);
    TEST_VARIANT(int64_t, math::min(_data.at(3), _data_min.at(3)), -729);
    // TEST_VARIANT(uint64_t, math::min(_data.at(4), _data_min.at(4)), 9);
    TEST_VARIANT(double, math::min(_data.at(5), _data_min.at(5)), -729.0);

    auto _data_max = _data;
    _reset_values(_data_max, 729.0);

    TEST_VARIANT(int32_t, math::max(_data.at(0), _data_max.at(0)), 729);
    TEST_VARIANT(uint32_t, math::max(_data.at(1), _data_max.at(1)), 729);
    TEST_VARIANT(float, math::max(_data.at(2), _data_max.at(2)), 729.0f);
    TEST_VARIANT(int64_t, math::max(_data.at(3), _data_max.at(3)), 729);
    TEST_VARIANT(uint64_t, math::max(_data.at(4), _data_max.at(4)), 729);
    TEST_VARIANT(double, math::max(_data.at(5), _data_max.at(5)), 729.0);

    _reset_values(_data, 9.0);

    TEST_VARIANT(int32_t, math::plus(_data.at(0), 9), 18);
    TEST_VARIANT(uint32_t, math::plus(_data.at(1), 9), 18);
    TEST_VARIANT(float, math::plus(_data.at(2), 9.0f), 18.0f);
    TEST_VARIANT(int64_t, math::plus(_data.at(3), 9), 18);
    TEST_VARIANT(uint64_t, math::plus(_data.at(4), 9), 18);
    TEST_VARIANT(double, math::plus(_data.at(5), 9.0), 18.0);

    _reset_values(_data, 9.0);

    TEST_VARIANT(int32_t, math::minus(_data.at(0), 4), 5);
    TEST_VARIANT(uint32_t, math::minus(_data.at(1), 4), 5);
    TEST_VARIANT(float, math::minus(_data.at(2), 4.0f), 5.0f);
    TEST_VARIANT(int64_t, math::minus(_data.at(3), 4), 5);
    TEST_VARIANT(uint64_t, math::minus(_data.at(4), 4), 5);
    TEST_VARIANT(double, math::minus(_data.at(5), 4.0), 5.0);

    _reset_values(_data, 9.0);

    TEST_VARIANT(int32_t, math::multiply(_data.at(0), 6), 54);
    TEST_VARIANT(uint32_t, math::multiply(_data.at(1), 6), 54);
    TEST_VARIANT(float, math::multiply(_data.at(2), 6.0f), 54.0f);
    TEST_VARIANT(int64_t, math::multiply(_data.at(3), 6), 54);
    TEST_VARIANT(uint64_t, math::multiply(_data.at(4), 6), 54);
    TEST_VARIANT(double, math::multiply(_data.at(5), 6.0), 54.0);

    _reset_values(_data, 9.0);

    TEST_VARIANT(int32_t, math::divide(_data.at(0), 2), 4);
    TEST_VARIANT(uint32_t, math::divide(_data.at(1), 2), 4);
    TEST_VARIANT(float, math::divide(_data.at(2), 2.0f), 4.5f);
    TEST_VARIANT(int64_t, math::divide(_data.at(3), 2), 4);
    TEST_VARIANT(uint64_t, math::divide(_data.at(4), 2), 4);
    TEST_VARIANT(double, math::divide(_data.at(5), 2.0), 4.5);

    _reset_values(_data, 9.0);

    TEST_VARIANT(int32_t, _data.at(0) * 4, 36);
    TEST_VARIANT(uint32_t, _data.at(1) * 4, 36);
    TEST_VARIANT(float, _data.at(2) * 4.0f, 36.0f);
    TEST_VARIANT(int64_t, _data.at(3) * 4, 36);
    TEST_VARIANT(uint64_t, _data.at(4) * 4, 36);
    TEST_VARIANT(double, _data.at(5) * 4.0, 36.0);

    TEST_VARIANT(int32_t, _data.at(0) / 2, 4);
    TEST_VARIANT(uint32_t, _data.at(1) / 2, 4);
    TEST_VARIANT(float, _data.at(2) / 2.0f, 4.5f);
    TEST_VARIANT(int64_t, _data.at(3) / 2, 4);
    TEST_VARIANT(uint64_t, _data.at(4) / 2, 4);
    TEST_VARIANT(double, _data.at(5) / 2.0, 4.5);
}

#endif

//--------------------------------------------------------------------------------------//

namespace comp = tim::component;

TEST_F(stl_overload_tests, update_statistics)
{
    using bundle_t = tim::lightweight_tuple<comp::wall_clock, comp::data_tracker_integer>;
    bundle_t _bundle{ details::get_test_name() };
    _bundle.push();
    for(int64_t i = 0; i < 10; ++i)
    {
        auto _v = 4 + ((i % 2 == 0) ? 0 : 2);
        _bundle.start().store(std::plus<int64_t>{}, _v);
        details::do_sleep(10 * _v);
        _bundle.stop().update_statistics(true);
        EXPECT_EQ(_bundle.get<comp::data_tracker_integer>()->get_last(), _v);
    }
    _bundle.pop();

    auto _storage = tim::storage<comp::data_tracker_integer>::instance()->get();
    int  _checks  = 0;
    for(const auto& itr : _storage)
    {
        if(itr.prefix().find(details::get_test_name()) != std::string::npos)
        {
            ++_checks;
            std::ostringstream _msg;
            _msg << "data: " << itr.data() << ", stats: " << itr.stats();
            const auto& _stats = itr.stats();
            EXPECT_EQ(_stats.get_sum(), 50) << _msg.str();
            EXPECT_EQ(_stats.get_count(), 10) << _msg.str();
            EXPECT_EQ(_stats.get_mean(), 5) << _msg.str();
            EXPECT_EQ(_stats.get_min(), 4) << _msg.str();
            EXPECT_EQ(_stats.get_max(), 6) << _msg.str();
            EXPECT_EQ(_stats.get_stddev(), 1) << _msg.str();
            EXPECT_EQ(_stats.get_variance(), 1) << _msg.str();
        }
    }
    ASSERT_EQ(_checks, 1) << "Test did not perform appropriate number of checks";
}

//--------------------------------------------------------------------------------------//
