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
using namespace tim::stl;
using namespace tim::stl::ostream;

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using papi_tuple_t = papi_tuple<PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_LST_INS>;

using auto_tuple_t =
    tim::auto_tuple<wall_clock, thread_cpu_clock, thread_cpu_util, process_cpu_clock,
                    process_cpu_util, peak_rss, page_rss>;

using full_measurement_t =
    tim::component_tuple<peak_rss, page_rss, stack_rss, data_rss, num_swap, num_io_in,
                         num_io_out, num_minor_page_faults, num_major_page_faults,
                         num_msg_sent, num_msg_recv, num_signals,
                         voluntary_context_switch, priority_context_switch, papi_tuple_t>;

using measurement_t =
    tim::component_tuple<wall_clock, system_clock, user_clock, cpu_clock, cpu_util,
                         thread_cpu_clock, thread_cpu_util, process_cpu_clock,
                         process_cpu_util, monotonic_clock, monotonic_raw_clock,
                         papi_tuple_t>;

using printed_t = tim::component_tuple<wall_clock, system_clock, user_clock, cpu_clock,
                                       thread_cpu_clock, process_cpu_clock>;

//--------------------------------------------------------------------------------------//
// dummy component which tests that a component which does not implement anything can
// compile
//
struct dummy_component : public base<dummy_component, void>
{
    using value_type = void;
    using this_type  = dummy_component;
    using base_type  = base<this_type, value_type>;
    static std::string label() { return "dummy_component"; }
    static std::string description() { return "dummy component"; }
};

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
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}
template <typename Tp>
void
serialize(const std::string& fname, const std::string& title, const Tp& obj)
{
    static constexpr auto spacing =
        tim::cereal::JSONOutputArchive::Options::IndentChar::space;
    std::stringstream ss;
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        tim::cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        tim::cereal::JSONOutputArchive          oa(ss, opts);
        oa(tim::cereal::make_nvp(title, obj));
    }
    std::ofstream ofs(fname.c_str());
    ofs << ss.str() << std::endl;
}
//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n)
{
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, n);
        return (n < 2) ? n : fibonacci(n - 1, cutoff) + fibonacci(n - 2, cutoff);
    }
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
//--------------------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
int64_t
time_fibonacci(int32_t n)
{
    TIMEMORY_MARKER(auto_tuple_t, "");
    return fibonacci(n);
}
//--------------------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
int64_t
time_fibonacci(int32_t n, int32_t cutoff)
{
    TIMEMORY_MARKER(auto_tuple_t, "");
    return fibonacci(n, cutoff);
}
//--------------------------------------------------------------------------------------//
// this function consumes approximately "n" milliseconds of real time
void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}
//--------------------------------------------------------------------------------------//
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

class tuple_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, usage)
{
    auto test_1_marker = TIMEMORY_BLANK_HANDLE(auto_tuple_t, details::get_test_name());

    full_measurement_t _use_beg("test_1_usage_begin");
    full_measurement_t _use_delta("test_1_usage_delta");
    full_measurement_t _use_end("test_1_usage_end");

    auto n = 5000000;
    _use_beg.record();
    _use_delta.start();
    std::vector<int64_t> v(n, 30);
    long                 nfib = details::random_entry(v);
    details::time_fibonacci(nfib);
    _use_delta.stop();
    _use_end.record();

    // std::cout << "usage (begin): " << _use_beg << std::endl;
    // std::cout << "usage (delta): " << _use_delta << std::endl;
    // std::cout << "usage (end):   " << _use_end << std::endl;

    std::vector<std::pair<std::string, full_measurement_t>> measurements = {
        { "begin", _use_beg },
        { "delta", _use_delta },
        { "end", _use_end }
    };
    // serialize("rusage.json", "usage", measurements);
    test_1_marker.stop();
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, all_threads)
{
    using auto_types_t                = tim::convert_t<auto_tuple_t, tim::type_list<>>;
    tim::settings::collapse_threads() = false;
    auto manager                      = tim::manager::instance();
    tim::manager::get_storage<auto_types_t>::clear(manager);
    auto starting_storage_size = tim::manager::get_storage<auto_types_t>::size(manager);
    auto data_size             = auto_tuple_t::size();

    using pair_t = std::pair<std::string, measurement_t>;

    mutex_t              mtx;
    std::vector<pair_t>  measurements;
    measurement_t        runtime("", false);
    printed_t            runtime_printed("", false);
    std::atomic<int64_t> ret;
    std::stringstream    lambda_ss;

    {
        TIMEMORY_BLANK_MARKER(auto_tuple_t, details::get_test_name());

        auto run_fib = [&](long n, std::promise<void>* p) {
            std::stringstream ss;
            ss << "fibonacci(" << n << ")";

            TIMEMORY_BLANK_MARKER(auto_tuple_t, "run_fib");
            if(p != nullptr)
                p->set_value();
            measurement_t _tm("thread " + ss.str(), false);
            _tm.start();
            ret += details::time_fibonacci(n, n - 2);
            _tm.stop();

            lock_t lk(mtx);
            measurements.emplace_back(ss.str(), _tm);
            lambda_ss << _tm << std::endl;
        };

        runtime_printed.start();
        runtime.start();
        {
            std::promise<void> _p1;
            std::promise<void> _p2;
            std::future<void>  _f1 = _p1.get_future();
            std::future<void>  _f2 = _p2.get_future();
            std::thread        _t1(run_fib, 42, &_p1);
            std::thread        _t2(run_fib, 42, &_p2);

            _f1.wait();
            _f2.wait();

            run_fib(40, nullptr);

            _t1.join();
            _t2.join();
        }
        runtime.stop();
        runtime_printed.stop();
    }

    std::cout << "\n" << lambda_ss.str() << '\n';
    std::cout << "total runtime: " << runtime << '\n';
    std::cout << "std::get: " << std::get<0>(runtime) << '\n';
    std::cout << "fibonacci total: " << ret.load() << "\n" << '\n';
    std::cout << "runtime process cpu time: " << runtime.get<process_cpu_clock>() << "\n";
    std::cout << "measured data: " << runtime_printed.get() << '\n';

    measurements.insert(measurements.begin(), pair_t("run", runtime));
    // serialize("timing.json", "runtime", measurements);

    // auto _test = std::tuple<int, double, std::string>{ 0, 0.2, "test" };
    // std::cout << "\nVARIADIC TUPLE PRINT: " << _test << "\n" << std::endl;

    auto final_storage_size = tim::manager::get_storage<auto_types_t>::size(manager);
    auto expected           = (final_storage_size - starting_storage_size);

    EXPECT_EQ(expected, 13 * data_size);

    const size_t store_size = 13;

    if(tim::trait::is_available<wall_clock>::value)
    {
        EXPECT_EQ(tim::storage<wall_clock>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<thread_cpu_clock>::value)
    {
        EXPECT_EQ(tim::storage<thread_cpu_clock>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<thread_cpu_util>::value)
    {
        EXPECT_EQ(tim::storage<thread_cpu_util>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<process_cpu_clock>::value)
    {
        EXPECT_EQ(tim::storage<process_cpu_clock>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<process_cpu_util>::value)
    {
        EXPECT_EQ(tim::storage<process_cpu_util>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<peak_rss>::value)
    {
        EXPECT_EQ(tim::storage<peak_rss>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<page_rss>::value)
    {
        EXPECT_EQ(tim::storage<page_rss>::instance()->get().size(), store_size);
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, collapsed_threads)
{
    using auto_types_t                = tim::convert_t<auto_tuple_t, tim::type_list<>>;
    tim::settings::collapse_threads() = true;
    auto manager                      = tim::manager::instance();
    tim::manager::get_storage<auto_types_t>::clear(manager);
    auto starting_storage_size = tim::manager::get_storage<auto_types_t>::size(manager);
    auto data_size             = auto_tuple_t::size();

    std::atomic<int64_t> ret;
    // accumulate metrics on full run
    TIMEMORY_BASIC_CALIPER(tot, auto_tuple_t, "[total]");

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // run a fibonacci calculation and accumulate metric
    auto run_fibonacci = [&](long n) {
        TIMEMORY_BLANK_MARKER(auto_tuple_t, "run_fibonacci");
        ret += details::time_fibonacci(n, n - 2);
    };

    {
        // run longer fibonacci calculations on two threads
        TIMEMORY_BLANK_CALIPER(master_thread_a, auto_tuple_t, details::get_test_name(),
                               "/[master_thread]/0");
        {
            std::thread t1(run_fibonacci, 40);
            t1.join();
            std::thread t2(run_fibonacci, 41);
            t2.join();
        }
        TIMEMORY_CALIPER_APPLY0(master_thread_a, stop);
    }

    {
        // run longer fibonacci calculations on two threads
        TIMEMORY_BLANK_CALIPER(master_thread_a, auto_tuple_t, details::get_test_name(),
                               "/[master_thread]/1");

        std::thread t1(run_fibonacci, 41);
        std::thread t2(run_fibonacci, 40);
        // run_fibonacci(42);

        t1.join();
        t2.join();

        TIMEMORY_CALIPER_APPLY0(master_thread_a, stop);
    }

    TIMEMORY_CALIPER_APPLY0(tot, stop);

    std::cout << "\nfibonacci total: " << ret.load() << "\n" << std::endl;

    auto final_storage_size = tim::manager::get_storage<auto_types_t>::size(manager);
    auto expected           = (final_storage_size - starting_storage_size);

    EXPECT_EQ(expected, 19 * data_size);

    const size_t store_size = 15;

    if(tim::trait::is_available<wall_clock>::value)
    {
        EXPECT_EQ(tim::storage<wall_clock>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<thread_cpu_clock>::value)
    {
        EXPECT_EQ(tim::storage<thread_cpu_clock>::instance()->get().size(),
                  store_size + 4);
    }

    if(tim::trait::is_available<thread_cpu_util>::value)
    {
        EXPECT_EQ(tim::storage<thread_cpu_util>::instance()->get().size(),
                  store_size + 4);
    }

    if(tim::trait::is_available<process_cpu_clock>::value)
    {
        EXPECT_EQ(tim::storage<process_cpu_clock>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<process_cpu_util>::value)
    {
        EXPECT_EQ(tim::storage<process_cpu_util>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<peak_rss>::value)
    {
        EXPECT_EQ(tim::storage<peak_rss>::instance()->get().size(), store_size);
    }

    if(tim::trait::is_available<page_rss>::value)
    {
        EXPECT_EQ(tim::storage<page_rss>::instance()->get().size(), store_size);
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, measure)
{
    tim::component_tuple<page_rss, peak_rss> prss(details::get_test_name() + "/" +
                                                  TIMEMORY_LABEL(""));
    {
        TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE(details::get_test_name() + "/[init]", page_rss,
                                           peak_rss);
        // just record the peak rss
        prss.measure();
        std::cout << "  Current rss: " << prss << std::endl;
    }

    {
        TIMEMORY_VARIADIC_AUTO_TUPLE(details::get_test_name() + "/[delta]", page_rss,
                                     peak_rss);
        // do something, where you want delta peak rss
        auto                 n = 10000000;
        std::vector<int64_t> v(n, 10);
        long                 nfib = details::random_entry(v);
        details::fibonacci(nfib);
        prss.stop();
        std::cout << "Change in rss: " << prss << std::endl;
    }

    // prss.reset();
    prss.measure();
    std::cout << "  Current rss: " << prss << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, concat)
{
    using lhs_t = tim::component_tuple<wall_clock, system_clock>;
    using rhs_t = tim::component_tuple<wall_clock, cpu_clock>;

    using comp_t0 =
        tim::mpl::remove_duplicates_t<typename tim::component_tuple<lhs_t, rhs_t>::type>;
    using comp_t1 = tim::mpl::remove_duplicates_t<
        typename tim::auto_tuple<lhs_t, rhs_t, user_clock>::type>;

    using lhs_l = tim::convert_t<lhs_t, tim::component_list<>>;
    using rhs_l = tim::convert_t<rhs_t, tim::component_list<>>;

    using data_t0 = tim::mpl::remove_duplicates_t<
        typename tim::component_list<lhs_l, rhs_l>::data_type>;
    using data_t1 = tim::mpl::remove_duplicates_t<
        typename tim::auto_list<lhs_l, rhs_l, user_clock>::data_type>;

    // std::cout << "\n" << std::flush;
    // std::cout << "comp_t0 = " << tim::demangle<comp_t0>() << "\n";
    // std::cout << "comp_t1 = " << tim::demangle<comp_t1>() << "\n";
    // std::cout << "\n" << std::flush;

    // std::cout << "data_t0 = " << tim::demangle<data_t0>() << "\n";
    // std::cout << "data_t1 = " << tim::demangle<data_t1>() << "\n";
    // std::cout << "\n" << std::flush;

    EXPECT_EQ(comp_t0::size(), 3);
    EXPECT_EQ(comp_t1::size(), 4);

    EXPECT_EQ(std::tuple_size<data_t0>::value, 3);
    EXPECT_EQ(std::tuple_size<data_t1>::value, 4);
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, get)
{
    using get_tuple_t = tim::component_tuple<wall_clock, dummy_component, cpu_clock>;

    get_tuple_t obj(details::get_test_name());
    obj.start();
    details::consume(1000);
    details::do_sleep(1000);
    obj.stop();

    using label_t                       = std::tuple<std::string, double>;
    std::tuple<double, double>   data   = obj.get();
    std::tuple<label_t, label_t> labels = obj.get_labeled();

    double wc_v;
    double cc_v;
    std::tie(wc_v, cc_v) = data;

    label_t wc_l;
    label_t cc_l;
    std::tie(wc_l, cc_l) = labels;

    // std::cout << "\n" << std::flush;
    // std::cout << std::fixed;
    // std::cout.precision(6);
    // std::cout << std::setw(12) << std::get<0>(wc_l) << " = " << wc_v << "\n";
    // std::cout << std::setw(12) << std::get<0>(cc_l) << " = " << cc_v << "\n";
    // std::cout << "\n" << std::flush;

    ASSERT_TRUE(std::get<0>(wc_l) == "wall_clock");
    ASSERT_TRUE(std::get<0>(cc_l) == "cpu_clock");

#if defined(TIMEMORY_MACOS)
    // GitHub CI for macOS is very noisy
    EXPECT_NEAR(wc_v, 2.0, 5.0e-1);
    EXPECT_NEAR(cc_v, 1.0, 5.0e-1);
#else
    EXPECT_NEAR(wc_v, 2.0, 5.0e-2);
    EXPECT_NEAR(cc_v, 1.0, 5.0e-2);
#endif
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, explicit_start)
{
    using namespace tim::quirk;
    using config_t     = config<explicit_start>;
    using config_type  = typename config_t::type;
    using this_tuple_t = tim::auto_tuple_t<wall_clock, cpu_clock, cpu_util, config_t>;

    auto ex_check_start_t = this_tuple_t::quirk_config<explicit_start>::value;
    auto ex_check_stop_t  = this_tuple_t::quirk_config<explicit_stop>::value;

    std::cout << "\n" << std::flush;
    std::cout << "config_t     : " << tim::demangle<config_t>() << "\n";
    std::cout << "config_type  : " << tim::demangle<config_type>() << "\n";
    std::cout << "this_tuple_t : " << tim::demangle<this_tuple_t>() << "\n";
    std::cout << std::boolalpha << "start check : " << ex_check_start_t << "\n";
    std::cout << std::boolalpha << "stop check : " << ex_check_stop_t << "\n";

    std::array<double, 3> value;
    {
        this_tuple_t obj(details::get_test_name());
        details::consume(1000);
        details::do_sleep(1000);
        obj.stop();
        value[0] = obj.get<wall_clock>()->get();
        value[1] = obj.get<cpu_clock>()->get();
        value[2] = obj.get<cpu_util>()->get();
        // std::cout << "\n" << std::flush;
        // std::cout << obj << "\n";
        // std::cout << "\n" << std::flush;
    }

    EXPECT_EQ(ex_check_start_t, true);
    EXPECT_EQ(ex_check_stop_t, false);
    EXPECT_NEAR(value[0], 0.0, 1.0e-6);
    EXPECT_NEAR(value[1], 0.0, 1.0e-6);
    EXPECT_NEAR(value[2], 0.0, 1.0e-6);
}

//--------------------------------------------------------------------------------------//

TEST_F(tuple_tests, auto_start)
{
    using namespace tim::quirk;
    using config_t    = config<auto_start>;
    using config_type = typename config_t::type;
    using this_tuple_t =
        tim::component_tuple_t<wall_clock, cpu_clock, cpu_util, config_t>;

    auto ex_check_start_t = this_tuple_t::quirk_config<auto_start>::value;
    auto ex_check_stop_t  = this_tuple_t::quirk_config<auto_stop>::value;

    std::cout << "\n" << std::flush;
    std::cout << "config_t     : " << tim::demangle<config_t>() << "\n";
    std::cout << "config_type  : " << tim::demangle<config_type>() << "\n";
    std::cout << "this_tuple_t : " << tim::demangle<this_tuple_t>() << "\n";
    std::cout << std::boolalpha << "start check : " << ex_check_start_t << "\n";
    std::cout << std::boolalpha << "stop check : " << ex_check_stop_t << "\n";

    std::array<double, 3> value;
    {
        this_tuple_t obj(details::get_test_name());
        details::consume(1000);
        details::do_sleep(1000);
        obj.stop();
        value[0] = obj.get<wall_clock>()->get();
        value[1] = obj.get<cpu_clock>()->get();
        value[2] = obj.get<cpu_util>()->get();
        // std::cout << "\n" << std::flush;
        // std::cout << obj << "\n";
        // std::cout << "\n" << std::flush;
    }

    EXPECT_EQ(ex_check_start_t, true);
    EXPECT_EQ(ex_check_stop_t, false);
#if defined(TIMEMORY_MACOS)
    // GitHub CI for macOS is very noisy
    EXPECT_NEAR(value[0], 2.0, 5.0e-1);
    EXPECT_NEAR(value[1], 1.0, 5.0e-1);
    EXPECT_NEAR(value[2], 50.0, 10.0);
#else
    EXPECT_NEAR(value[0], 2.0, 5.0e-2);
    EXPECT_NEAR(value[1], 1.0, 5.0e-2);
    EXPECT_NEAR(value[2], 50.0, 5.0);
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
auto
run(const std::string& lbl, int n)
{
    Tp _one{ details::get_test_name() + "/" + lbl,
             tim::quirk::config<tim::quirk::explicit_pop>{} };
    _one.start();
    details::do_sleep(n / 2);
    _one.stop();
    Tp _two{ "none", tim::quirk::config<tim::quirk::explicit_push>{} };
    _two.start();
    details::do_sleep(n / 2);
    return (_one += _two.stop()).pop();
}

template <typename Tp>
auto
validate(const std::string& lbl, int n)
{
    // std::cout << "\n##### " << lbl << " #####\n";
    std::shared_ptr<Tp> obj{};
    double              val = 0.0;
    {
        auto tmp      = run<Tp>(lbl, n);
        std::tie(val) = tmp.get();
        val *= wall_clock::get_unit() / static_cast<double>(tim::units::msec);
#if defined(TIMEMORY_MACOS)
        EXPECT_NEAR(val, n, 300) << tmp;
#else
        EXPECT_NEAR(val, n, 150) << tmp;
#endif
        EXPECT_EQ(tmp.laps(), 2) << tmp;
        obj = std::make_shared<Tp>(details::get_test_name() + "/" + lbl,
                                   tim::scope::config{} + tim::scope::timeline{});
        obj->push();
        *obj += tmp;
        obj->pop();
    }
    double old    = val;
    std::tie(val) = obj->get();
    val *= wall_clock::get_unit() / static_cast<double>(tim::units::msec);
    EXPECT_NEAR(val, old, 10) << *obj;
    EXPECT_EQ(obj->laps(), 2) << *obj;
}

TEST_F(tuple_tests, addition_tests)
{
    auto _initializer = [](auto& _obj) { _obj.template initialize<wall_clock>(); };

    tim::component_list<wall_clock>::get_initializer()                  = _initializer;
    tim::component_bundle<TIMEMORY_API, wall_clock*>::get_initializer() = _initializer;

    validate<tim::lightweight_tuple<wall_clock>>("lightweight_tuple", 1000);
    validate<tim::component_tuple<wall_clock>>("component_tuple", 1000);
    validate<tim::component_list<wall_clock>>("component_list", 1000);
    validate<tim::component_bundle<TIMEMORY_API, wall_clock>>("component_bundle", 1000);
    validate<tim::component_bundle<TIMEMORY_API, wall_clock*>>("component_bundle*", 1000);

    validate<tim::auto_tuple<wall_clock>>("auto_tuple", 1000);
    validate<tim::auto_list<wall_clock>>("auto_list", 1000);
    validate<tim::auto_bundle<TIMEMORY_API, wall_clock>>("auto_bundle", 1000);
    validate<tim::auto_bundle<TIMEMORY_API, wall_clock*>>("auto_bundle*", 1000);

    auto wc_storage = tim::storage<wall_clock>::instance()->get();

    for(auto& itr : wc_storage)
    {
        if(itr.prefix().find(details::get_test_name()) == std::string::npos)
            continue;
#if defined(TIMEMORY_MACOS)
        // GitHub CI for macOS is noisy
        EXPECT_NEAR(itr.data().get(), (1.0 * tim::units::sec) / wall_clock::get_unit(),
                    (5.0e-1 * tim::units::sec) / wall_clock::get_unit())
            << itr.data();
#else
        EXPECT_NEAR(itr.data().get(), (1.0 * tim::units::sec) / wall_clock::get_unit(),
                    (5.0e-2 * tim::units::sec) / wall_clock::get_unit())
            << itr.data();
#endif
        EXPECT_EQ(itr.data().get_laps(), 2) << itr.data();
    }
}

//--------------------------------------------------------------------------------------//
