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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <x86intrin.h>

using namespace tim::component;
using mutex_t   = std::mutex;
using lock_t    = std::unique_lock<mutex_t>;
using condvar_t = std::condition_variable;
using device_t  = tim::device::cpu;

static constexpr uint64_t FLOPS  = 16;
static constexpr uint64_t TRIALS = 2;
static constexpr uint64_t SIZE   = 1024 * 1024 * 64;

#if defined(__GNUG__) && !defined(__OPTIMIZE__)
// for code coverage (always GNU C++), allow very high differences in debug mode
static const int64_t ops_tolerance  = (TRIALS * SIZE * FLOPS) * 2000;
static const int64_t lst_tolerance  = (TRIALS * SIZE * FLOPS) * 2000;
static const double  rate_tolerance = 500.0;
#else
// tolerance of 5.0e-5
static const int64_t ops_tolerance  = (TRIALS * SIZE * FLOPS) / 2000;
static const int64_t lst_tolerance  = (TRIALS * SIZE * FLOPS) / 2000;
static const double  rate_tolerance = 0.05;
#endif

static std::function<void()> begin_cb = [] {};
static std::function<void()> end_cb   = [] {};

#define CHECK_WORKING()                                                                  \
    if(!tim::papi::working())                                                            \
    {                                                                                    \
        printf("Skipping %s because PAPI is not working\n",                              \
               details::get_test_name().c_str());                                        \
        return;                                                                          \
    }

#define CHECK_AVAILABLE(type)                                                            \
    CHECK_WORKING();                                                                     \
    if(!tim::trait::is_available<type>::value)                                           \
    {                                                                                    \
        printf("skipping %s because %s is not available\n",                              \
               details::get_test_name().c_str(), tim::demangle<type>().c_str());         \
        return;                                                                          \
    }

#define CHECK_PRESET(preset)                                                             \
    if(!details::preset_is_available(preset, #preset))                                   \
    {                                                                                    \
        std::cerr << "Warning! " << details::get_test_name() << " skipped because "      \
                  << #preset << " is not available" << std::endl;                        \
        return;                                                                          \
    }

template <typename Tp>
using ptr_t = std::shared_ptr<Tp>;
template <typename Tp>
using return_type = std::tuple<ptr_t<Tp>, int64_t, int64_t>;

namespace details
{
//--------------------------------------------------------------------------------------//
bool
preset_is_available(int _preset, const std::string& _preset_name)
{
    for(const auto& itr : tim::papi::available_events_info())
    {
        if(_preset == itr.id())
            return itr.available();
    }
    throw std::runtime_error("Error! No matching preset " + _preset_name);
    return false;
}
//--------------------------------------------------------------------------------------//
template <typename Tp, int64_t Nunroll, typename ComponentT, typename... ArgsT>
return_type<ComponentT>
run_cpu_ops_kernel(int64_t ntrials, int64_t nsize, ArgsT&&... _args)
{
    // auto op_func = [](Tp& a, const Tp& b, const Tp& c) { a = b + c; };
    auto op_func             = [](Tp& a, const Tp& b, const Tp& c) { a = a * b + c; };
    auto store_func          = [](Tp& a, const Tp& b) { a = b; };
    auto bytes_per_elem      = sizeof(Tp);
    auto lst_per_vec         = (TIMEMORY_VEC / sizeof(Tp));
    auto cpubit_factor       = 8 / sizeof(Tp);
    auto mem_access_per_elem = 2;

    int64_t nops         = Nunroll;
    int64_t working_size = nsize * ntrials;
    int64_t total_bytes  = working_size * bytes_per_elem * mem_access_per_elem;
    int64_t total_lst    = total_bytes / lst_per_vec * cpubit_factor;
    int64_t total_ops    = working_size * nops;
    int64_t kB           = tim::units::kilobyte;

    std::cout << "\n";
    std::cout << "               FLOPs:  " << nops << "\n";
    std::cout << "              trials:  " << ntrials << "\n";
    std::cout << "        working size:  " << working_size / kB << " kB\n";
    std::cout << "   bytes per element:  " << bytes_per_elem << "\n";
    std::cout << "   accesses per elem:  " << mem_access_per_elem << "\n";
    std::cout << "         total bytes:  " << total_bytes << "\n";
    std::cout << "    total operations:  " << total_ops << "\n";
    std::cout << "    total load/store:  " << total_lst << "\n";

    // Tp* array = new Tp[nsize];
    std::vector<Tp, tim::ert::aligned_allocator<Tp, 64>> array(nsize);
    std::memset(array.data(), 0, nsize * sizeof(Tp));

    ptr_t<ComponentT> _obj;
    {
        using measure_t = tim::auto_bundle<TIMEMORY_API, ComponentT>;
        measure_t obj(std::forward<ArgsT>(_args)...);
        begin_cb();
        tim::ert::ops_kernel<Nunroll, device_t>(ntrials, nsize, array.data(), op_func,
                                                store_func);
        end_cb();
        obj.stop();
        _obj = std::make_shared<ComponentT>(*obj.template get<ComponentT>());
    }

    return return_type<ComponentT>(_obj, total_ops, total_lst);
}
//--------------------------------------------------------------------------------------//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}
//--------------------------------------------------------------------------------------//
template <typename Up, typename Tp>
void
report(const Tp& measured_count, const Tp& explicit_count, const Tp& tolerance,
       const std::string& label)
{
    Tp     diff  = measured_count - explicit_count;
    double err   = (diff / static_cast<double>(explicit_count)) * 100.0;
    double ratio = measured_count / static_cast<double>(explicit_count);
    std::cout << "\n";
    std::cout << "\t   Test name:  " << get_test_name() << std::endl;
    std::cout << "\t   Data Type:  " << tim::demangle(typeid(Up).name()) << std::endl;
    std::cout << "\t    Counters:  " << label << std::endl;
    std::cout << "\t    Measured:  " << measured_count << std::endl;
    std::cout << "\t    Expected:  " << explicit_count << std::endl;
    std::cout << "\t   Tolerance:  " << tolerance << std::endl;
    std::cout << "\t  Difference:  " << diff << std::endl;
    std::cout << "\t       Ratio:  " << ratio << std::endl;
    std::cout << "\t   Abs Error:  " << err << " %" << std::endl;
    std::cout << "\n";
}
//--------------------------------------------------------------------------------------//
}  // namespace details

//--------------------------------------------------------------------------------------//

class papi_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN
    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, print_hw_info) { tim::papi::print_hw_info(); }

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, tuple_single_precision_ops)
{
    using test_type = papi_tuple<PAPI_SP_OPS>;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_SP_OPS)

    auto ret = details::run_cpu_ops_kernel<float, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>().at(0);

    // int idx = 0;
    // for(auto itr : test_type::get_overhead())
    //    std::cout << "Overhead for counter " << idx++ << " is " << itr << std::endl;

    details::report<float>(total_measured, total_expected, ops_tolerance,
                           "PAPI float ops");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, array_single_precision_ops)
{
    using test_type = papi_array_t;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_SP_OPS)

    test_type::get_initializer() = []() { return std::vector<int>({ PAPI_SP_OPS }); };
    auto ret                     = details::run_cpu_ops_kernel<float, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>().at(0);

    details::report<float>(total_measured, total_expected, ops_tolerance,
                           "PAPI float ops");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, vector_single_precision_ops)
{
    using test_type = papi_vector;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_SP_OPS)

    test_type::get_initializer() = []() { return std::vector<int>({ PAPI_SP_OPS }); };
    auto ret                     = details::run_cpu_ops_kernel<float, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>().at(0);

    details::report<float>(total_measured, total_expected, ops_tolerance,
                           "PAPI float ops");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

//--------------------------------------------------------------------------------------//

#if !defined(DISABLE_DP) || (defined(DISABLE_DP) && DISABLE_DP == 0)

TEST_F(papi_tests, tuple_double_precision_ops)
{
    using test_type = papi_tuple<PAPI_DP_OPS>;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_DP_OPS)

    auto ret = details::run_cpu_ops_kernel<double, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>().at(0);

    details::report<double>(total_measured, total_expected, ops_tolerance,
                            "PAPI double ops");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, array_double_precision_ops)
{
    using test_type = papi_array_t;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_DP_OPS)

    test_type::get_initializer() = []() { return std::vector<int>({ PAPI_DP_OPS }); };
    auto ret                     = details::run_cpu_ops_kernel<double, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>().at(0);

    details::report<double>(total_measured, total_expected, ops_tolerance,
                            "PAPI double ops");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, vector_double_precision_ops)
{
    using test_type = papi_vector;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_DP_OPS)

    test_type::get_initializer() = []() { return std::vector<int>({ PAPI_DP_OPS }); };
    auto ret                     = details::run_cpu_ops_kernel<double, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<1>(ret);
    auto total_measured = obj->get<int64_t>().at(0);

    details::report<double>(total_measured, total_expected, ops_tolerance,
                            "PAPI double ops");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

#endif  // !defined(DISABLE_DP) || (defined(DISABLE_DP) && DISABLE_DP == 0)

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, tuple_load_store_ins)
{
    using test_type = papi_tuple<PAPI_LD_INS, PAPI_SR_INS, PAPI_LST_INS>;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_LD_INS)
    CHECK_PRESET(PAPI_SR_INS)
    CHECK_PRESET(PAPI_LST_INS)

    auto ret = details::run_cpu_ops_kernel<float, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<2>(ret);
    auto total_measured = obj->get<int64_t>().at(0) + obj->get<int64_t>().at(1);

    auto indiv = obj->get<int64_t>().at(0) + obj->get<int64_t>().at(1);
    auto accum = obj->get<int64_t>().at(2);
    EXPECT_EQ(indiv, accum);
    EXPECT_GT(indiv, 0);
    EXPECT_GT(accum, 0);

    details::report<float>(total_measured, total_expected, ops_tolerance,
                           "PAPI load/store");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, array_load_store_ins)
{
    using test_type = papi_array_t;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_LD_INS)
    CHECK_PRESET(PAPI_SR_INS)
    CHECK_PRESET(PAPI_LST_INS)

    test_type::get_initializer() = []() {
        return std::vector<int>({ PAPI_LD_INS, PAPI_SR_INS, PAPI_LST_INS });
    };
    auto ret = details::run_cpu_ops_kernel<double, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<2>(ret);
    auto total_measured = obj->get<int64_t>().at(2);

    auto indiv = obj->get<int64_t>().at(0) + obj->get<int64_t>().at(1);
    auto accum = obj->get<int64_t>().at(2);
    EXPECT_EQ(indiv, accum);
    EXPECT_GT(indiv, 0);
    EXPECT_GT(accum, 0);

    details::report<double>(total_measured, total_expected, lst_tolerance,
                            "PAPI load/store");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, vector_load_store_ins)
{
    using test_type = papi_vector;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_LD_INS)
    CHECK_PRESET(PAPI_SR_INS)
    CHECK_PRESET(PAPI_LST_INS)

    test_type::get_initializer() = []() {
        return std::vector<int>({ PAPI_LD_INS, PAPI_SR_INS, PAPI_LST_INS });
    };
    auto ret = details::run_cpu_ops_kernel<double, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj            = std::get<0>(ret);
    auto total_expected = std::get<2>(ret);
    auto total_measured = obj->get<int64_t>().at(2);

    auto indiv = obj->get<int64_t>().at(0) + obj->get<int64_t>().at(1);
    auto accum = obj->get<int64_t>().at(2);
    EXPECT_EQ(indiv, accum);
    EXPECT_GT(indiv, 0);
    EXPECT_GT(accum, 0);

    details::report<double>(total_measured, total_expected, lst_tolerance,
                            "PAPI load/store");
    EXPECT_NEAR(total_measured, total_expected, ops_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, tuple_flops_rate)
{
    using check_type = papi_tuple<PAPI_SP_OPS>;
    using test_type  = papi_rate_tuple<wall_clock, PAPI_SP_OPS>;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_SP_OPS)

    tim::lightweight_tuple<check_type, wall_clock> check{};

    // call-backs around algorithm
    begin_cb = [&check]() { check.start(); };
    end_cb   = [&check]() { check.stop(); };

    auto ret = details::run_cpu_ops_kernel<float, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj = std::get<0>(ret);

    auto sp_ops_rate = obj->get().at(0);

    auto* pt = check.get<check_type>();
    auto* wc = check.get<wall_clock>();

    auto _wc_val = wc->get();
    auto _pt_val = pt->get();

    auto _sp_ops = sp_ops_rate * _wc_val;

    std::cout << "\n";
    std::cout << "RATE_TUPLE : " << *obj << std::endl;
    std::cout << "TUPLE      : " << *pt << std::endl;
    std::cout << "WALL-CLOCK : " << *wc << std::endl;

    EXPECT_NEAR(_sp_ops, _pt_val.at(0), 0.1 * _pt_val.at(0));

    auto total_expected  = std::get<1>(ret) / _wc_val;
    auto total_measured  = obj->get<double>().at(0);
    auto total_tolerance = rate_tolerance * total_expected;
    details::report<double>(total_measured, total_expected,
                            static_cast<double>(total_tolerance), "PAPI SP flops rate");

#if !defined(TIMEMORY_USE_COVERAGE)
    EXPECT_NEAR(total_measured, total_expected, total_tolerance);
#endif
}

//--------------------------------------------------------------------------------------//

TEST_F(papi_tests, tuple_load_store_ins_rate)
{
    using check_type = papi_tuple<PAPI_LD_INS, PAPI_SR_INS, PAPI_LST_INS>;
    using test_type = papi_rate_tuple<wall_clock, PAPI_LD_INS, PAPI_SR_INS, PAPI_LST_INS>;
    CHECK_AVAILABLE(test_type);
    CHECK_PRESET(PAPI_LD_INS)
    CHECK_PRESET(PAPI_SR_INS)
    CHECK_PRESET(PAPI_LST_INS)

    tim::lightweight_tuple<check_type, wall_clock> check{};

    // call-backs around algorithm
    begin_cb = [&check]() { check.start(); };
    end_cb   = [&check]() { check.stop(); };

    auto ret = details::run_cpu_ops_kernel<float, FLOPS, test_type>(
        TRIALS, SIZE, details::get_test_name());

    auto obj = std::get<0>(ret);

    auto ld_ins_rate  = obj->get().at(0);
    auto sr_ins_rate  = obj->get().at(1);
    auto lst_ins_rate = obj->get().at(2);

    auto* pt = check.get<check_type>();
    auto* wc = check.get<wall_clock>();

    auto _wc_val = wc->get();
    auto _pt_val = pt->get();

    auto _ld_ins  = ld_ins_rate * _wc_val;
    auto _sr_ins  = sr_ins_rate * _wc_val;
    auto _lst_ins = lst_ins_rate * _wc_val;

    std::cout << "\n";
    std::cout << "RATE_TUPLE : " << *obj << std::endl;
    std::cout << "TUPLE      : " << *pt << std::endl;
    std::cout << "WALL-CLOCK : " << *wc << std::endl;

    EXPECT_NEAR(_ld_ins, _pt_val.at(0), 0.1 * _pt_val.at(0));
    EXPECT_NEAR(_sr_ins, _pt_val.at(1), 0.1 * _pt_val.at(1));
    EXPECT_NEAR(_lst_ins, _pt_val.at(2), 0.1 * _pt_val.at(2));

    auto indiv = obj->get().at(0) + obj->get().at(1);
    auto accum = obj->get().at(2);
    EXPECT_NEAR(indiv, accum, 50);
    EXPECT_GT(indiv, 0);
    EXPECT_GT(accum, 0);

    auto total_expected  = std::get<2>(ret) / _wc_val;
    auto total_measured  = obj->get<double>().at(2);
    auto total_tolerance = rate_tolerance * total_expected;
    details::report<double>(total_measured, total_expected,
                            static_cast<double>(total_tolerance), "PAPI load/store rate");

#if !defined(TIMEMORY_USE_COVERAGE)
    EXPECT_NEAR(total_measured, total_expected, total_tolerance);
#endif
}

//--------------------------------------------------------------------------------------//

using instruction_rate_t =
    papi_rate_tuple<wall_clock, PAPI_LD_INS, PAPI_SR_INS, PAPI_LST_INS>;
using instruction_tuple_t = papi_tuple<PAPI_LD_INS, PAPI_SR_INS, PAPI_LST_INS>;

TIMEMORY_INITIALIZE_STORAGE(instruction_rate_t, instruction_tuple_t);

//--------------------------------------------------------------------------------------//
