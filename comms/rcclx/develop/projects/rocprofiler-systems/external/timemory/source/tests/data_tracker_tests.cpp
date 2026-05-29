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
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

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

// random number generator
template <typename T = std::mt19937>
T&
get_rng(size_t initial_seed = 0)
{
    static std::atomic<int> _cnt{ 0 };
    auto                    _tid      = ++_cnt;
    static thread_local T   _instance = [=]() {
        T _rng;
        _rng.seed(((initial_seed == 0) ? std::random_device()() : initial_seed) *
                  (10 * _tid));
        return _rng;
    }();
    return _instance;
}

// random integer
template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
T
get_random_value(T beg, T end)
{
    std::uniform_int_distribution<T> dist(beg, end);
    return dist(get_rng());
}

template <typename T>
struct identity
{
    using type = T;
};

template <typename T>
using identity_t = typename identity<T>::type;

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
T
get_random_value(identity_t<T> beg, T end)
{
    std::uniform_real_distribution<T> dist(beg, end);
    return dist(get_rng());
}

// get a random entry from vector
template <typename Tp>
Tp
random_entry(const std::vector<Tp>& v)
{
    return v.at(get_random_value(0, v.size() - 1));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class data_tracker_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(data_tracker_tests, iteration_tracker)
{
    tim::settings::collapse_processes() = true;
    tim::settings::collapse_threads()   = true;
    tim::settings::add_secondary()      = true;

    struct iteration_tag
    {};

    using iteration_value_tracker_t = data_tracker<double, iteration_tag>;
    using iteration_count_tracker_t = data_tracker<uint64_t, iteration_tag>;
    using iteration_value_handle_t  = data_handler_t<iteration_value_tracker_t>;

    iteration_value_tracker_t::label()       = "iteration_value";
    iteration_value_tracker_t::description() = "Iteration value tracker";
    iteration_count_tracker_t::label()       = "iteration_count";
    iteration_count_tracker_t::description() = "Iteration count tracker";

    using tuple_t =
        tim::auto_tuple<wall_clock, iteration_count_tracker_t, iteration_value_tracker_t>;

    double       err      = 5.e3;
    const double tol      = 1.0e-3;
    uint64_t     num_iter = 0;

    tuple_t t{ details::get_test_name() };
    while(err > tol)
    {
        err -= details::get_random_value<double>(0.0, err * tol);

        // test that ((1 + 1 - 1) * 2) / 2 == 1
        auto _last = t.store(std::plus<uint64_t>{}, 1)
                         .store(std::plus<uint64_t>{}, 1)
                         .store(std::minus<uint64_t>{}, 1)
                         .get<iteration_count_tracker_t>()
                         ->get();

        auto _curr = t.store([](uint64_t lhs, uint64_t rhs) { return lhs * rhs; },
                             static_cast<uint64_t>(2))
                         .get<iteration_count_tracker_t>()
                         ->get();
        EXPECT_NE(_last, _curr) << "num_iter = " << num_iter << ", " << t;

        uint64_t v = 2;
        _last =
            t.store(std::divides<uint64_t>{}, v).get<iteration_count_tracker_t>()->get();
        EXPECT_NE(_last, _curr) << "num_iter = " << num_iter << ", " << t;

        t.store(iteration_value_handle_t{}, err)
            .add_secondary("error_per_iter", iteration_value_handle_t{},
                           err / (num_iter + 1));
        ++num_iter;
    }
    t.stop();

    std::cout << "\n" << t << std::endl;
    std::cout << "num_iter : " << num_iter << std::endl;
    std::cout << "error    : " << err << "\n" << std::endl;

    EXPECT_NE(num_iter, 0);
    EXPECT_EQ(t.get<iteration_count_tracker_t>()->get(), num_iter);
    EXPECT_LT(t.get<iteration_value_tracker_t>()->get(), tol);
    EXPECT_NEAR(t.get<iteration_value_tracker_t>()->get(), err, 1.0e-6);

    auto itrv_storage = tim::storage<iteration_value_tracker_t>::instance()->get();
    EXPECT_EQ(itrv_storage.size(), 2);
    EXPECT_EQ(itrv_storage.at(0).prefix().substr(4), details::get_test_name());
    EXPECT_EQ(itrv_storage.at(1).prefix(), std::string{ ">>> |_error_per_iter" });
    EXPECT_EQ(itrv_storage.at(1).depth(), 1);
}

//--------------------------------------------------------------------------------------//

struct myproject : tim::concepts::api
{};
struct myproject_debug : tim::concepts::api
{};

using itr_tracker_type   = data_tracker<int64_t, myproject>;
using err_tracker_type   = data_tracker<double, myproject>;
using err_tracker_type_d = data_tracker<double, myproject_debug>;

TIMEMORY_DECLARE_COMPONENT(myproject_logger)

// set the label and descriptions
TIMEMORY_METADATA_SPECIALIZATION(itr_tracker_type, "myproject_iterations",
                                 "Number of iterations", "Iteration count in loops")

TIMEMORY_METADATA_SPECIALIZATION(err_tracker_type, "myproject_error",
                                 "Iteration error delta", "Amount of error reduced")

TIMEMORY_METADATA_SPECIALIZATION(err_tracker_type_d, "myproject_error_timeline",
                                 "Timeline of all iteration errors",
                                 "Value of error over time")

TIMEMORY_METADATA_SPECIALIZATION(myproject_logger, "myproject_logger", "Logs messages",
                                 "")

// add statistics capabilities
TIMEMORY_STATISTICS_TYPE(itr_tracker_type, int64_t)
TIMEMORY_STATISTICS_TYPE(err_tracker_type, double)

#if defined(NDEBUG)
#    undef NDEBUG
#endif

// always uses timeline storage
TIMEMORY_DEFINE_CONCRETE_TRAIT(timeline_storage, err_tracker_type_d, true_type)
// never instantiate err_tracker_type_d when NDEBUG is defined
#if defined(NDEBUG)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, err_tracker_type_d, false_type)
#else
// default to turned off at runtime
TIMEMORY_DEFINE_CONCRETE_TRAIT(default_runtime_enabled, err_tracker_type_d, false_type)
#endif

TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, myproject_logger, false_type)
// TIMEMORY_DEFINE_CONCRETE_TRAIT(report_depth, myproject_logger, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, myproject_logger, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_mean, myproject_logger, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_self, myproject_logger, false_type)
// TIMEMORY_DEFINE_CONCRETE_TRAIT(report_sum, myproject_logger, false_type)

namespace tim
{
namespace component
{
struct myproject_logger : base<myproject_logger, short>
{
    void start() {}
    void stop() {}
};
}  // namespace component
//
namespace operation
{
// disable starting and stopping err_tracker_type_d completely
template <typename ApiT>
struct generic_operator<err_tracker_type_d, start<err_tracker_type_d>, ApiT>
{
    template <typename... Args>
    generic_operator(Args&&...)
    {}
    template <typename... Args>
    void operator()(Args&&...)
    {}
};
template <typename ApiT>
struct generic_operator<err_tracker_type_d, stop<err_tracker_type_d>, ApiT>
{
    template <typename... Args>
    generic_operator(Args&&...)
    {}
    template <typename... Args>
    void operator()(Args&&...)
    {}
};
}  // namespace operation
}  // namespace tim

// this is the generic bundle pairing a timer with an iteration tracker
// using this and not updating the iteration tracker will create entries
// in the call-graph with zero iterations.
using bundle_t = tim::auto_tuple<wall_clock, itr_tracker_type>;

// this is a dedicated bundle for adding data-tracker entries. This style
// can also be used with the iteration tracker or you can bundle
// both trackers together. The auto_tuple will call start on construction
// and stop on destruction so once can construct a nameless temporary of the
// this bundle type and call store(...) on the nameless tmp. This will
// ensure that the statistics are updated for each entry
//
using err_bundle_t   = tim::component_bundle_t<myproject, err_tracker_type>;
using err_bundle_t_d = tim::component_bundle_t<myproject_debug, err_tracker_type_d>;
using log_bundle_t   = tim::component_bundle_t<myproject, wall_clock, myproject_logger>;

TEST_F(data_tracker_tests, convergence_test)
{
    tim::settings::timing_precision()         = 9;
    size_t                           nthreads = 4;
    std::vector<std::thread>         threads{};
    std::vector<std::vector<double>> err_diffs(nthreads);
    std::vector<int64_t>             num_iters(nthreads, 0);

    // if NDEBUG is defined, this does nothing.
    if(tim::settings::debug() || tim::get_env<bool>("DEBUG", false))
        tim::trait::runtime_enabled<err_tracker_type_d>::set(true);

    auto _run = [&err_diffs, &num_iters, nthreads](size_t i, auto lbl, size_t nitr,
                                                   size_t /*modulo*/) {
        details::get_rng().seed(1000 * i);
        auto _name = TIMEMORY_JOIN('/', details::get_test_name(), lbl);

        for(size_t j = 0; j < nitr; ++j)
        {
            double            err      = 5.e3;
            const double      tol      = 1.0e-3;
            int64_t           num_iter = 0;
            std::stringstream _tcout{};  // thread output
            auto&             _err_diff = err_diffs.at(i);

            bundle_t _bundle{ _name };
            while(err > tol)
            {
                // this will get optimized away when NDEBUG is defined
                err_bundle_t_d{ _name }.store(err);
                // mark the starting number of iterations
                _bundle.mark_begin(num_iter++);

                tim::auto_tuple<wall_clock> _compute_timer{ "compute" };

                log_bundle_t{ _name + "/ignore" }
                    .push(tim::mpl::piecewise_select<myproject_logger>{},
                          tim::scope::config{} + tim::scope::flat{})
                    .start(tim::mpl::piecewise_select<myproject_logger>{})
                    .stop(tim::mpl::piecewise_select<myproject_logger>{})
                    .pop(tim::mpl::piecewise_select<myproject_logger>{});

                auto initial_err = err;
                err -= details::get_random_value<double>(
                    0.0, std::min<double>(nthreads - i, err));

                // mark the ending number of iterations
                _bundle.mark_end(std::plus<int64_t>{}, num_iter);

                // compute the change in error
                auto _err_delta = initial_err - err;
                err_bundle_t{ _name }.push().store(_err_delta).print(_tcout, true).pop();
                _err_diff.emplace_back(_err_delta);
            }
            num_iters.at(i) += num_iter;
            _bundle.stop();

            // if(j % modulo == (modulo - 1))
            // {
            //     tim::auto_lock_t _lk{ tim::type_mutex<decltype(std::cout)>() };
            //     if(tim::settings::debug())
            //         std::cout << "\niteration " << i << "\n" << _tcout.str() <<
            //         std::endl;
            //     std::cout << "\niteration " << i << "\n\t" << _bundle
            //               << "\n\tnum_iter : " << num_iter << "\n\terror    : " << err
            //               << "\n"
            //               << std::endl;
            // }
        }
    };

    for(size_t i = 0; i < nthreads; ++i)
    {
        threads.emplace_back(_run, i, i, 10, 10);
    }

    for(auto& itr : threads)
        itr.join();

    auto&& _compare = [](auto& lhs, auto& rhs) {
        return [](const std::string& _lhs, const std::string& _rhs) {
            auto lidx = _lhs.find_last_of("0123456789");
            auto ridx = _rhs.find_last_of("0123456789");
            if(lidx == std::string::npos || ridx == std::string::npos)
                return (lidx == ridx) ? (_lhs < _rhs) : (lidx < ridx);
            return std::stoi(_lhs.substr(lidx, lidx + 1)) <
                   std::stoi(_rhs.substr(ridx, ridx + 1));
        }(lhs.prefix(), rhs.prefix());
    };

    auto itr_storage = tim::storage<itr_tracker_type>::instance()->get();
    auto err_storage = tim::storage<err_tracker_type>::instance()->get();
    auto wc_storage  = tim::storage<wall_clock>::instance()->get();

    // for(auto& itr : itr_storage)
    //     std::cout << itr.prefix() << " :: " << itr.data() << std::endl;
    for(auto& itr : wc_storage)
    {
        // std::cout << itr.prefix() << " :: " << itr.data() << std::endl;
        ASSERT_EQ(itr.prefix().find("ignore"), std::string::npos)
            << itr.prefix() << " :: " << itr.data() << "\n";
    }

    ASSERT_EQ(itr_storage.size(), nthreads);
    ASSERT_EQ(err_storage.size(), nthreads);

    // for(size_t i = 0; i < nthreads; ++i)
    // {
    //     std::cout << itr_storage.at(i).data() << std::endl;
    // }

    std::sort(itr_storage.begin(), itr_storage.end(), _compare);
    std::sort(err_storage.begin(), err_storage.end(), _compare);

    for(size_t i = 0; i < nthreads; ++i)
    {
        auto _true_iter = num_iters.at(i);
        auto _meas_iter = itr_storage.at(i).data().get();
        EXPECT_EQ(_true_iter, _meas_iter) << itr_storage.at(i).data();
    }

    for(size_t i = 0; i < nthreads; ++i)
    {
        auto _true_err =
            std::accumulate(err_diffs.at(i).begin(), err_diffs.at(i).end(), 0.0);
        auto _meas_err = err_storage.at(i).data().get();
        EXPECT_NEAR(_true_err, _meas_err, 1.0e-3) << err_storage.at(i).data();
    }
}

TIMEMORY_INITIALIZE_STORAGE(wall_clock, itr_tracker_type, err_tracker_type,
                            myproject_logger)

//--------------------------------------------------------------------------------------//
