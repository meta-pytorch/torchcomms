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
//

#include "timemory/api.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DEFINE_CONCRETE_TRAIT(supports_runtime_enabled, TIMEMORY_API, false_type)

#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/invoker.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/testing.hpp"

using namespace tim::component;

using papi_tuple_t = papi_array<8>;
using auto_tuple_t = tim::auto_bundle_t<TIMEMORY_API, wall_clock>;
using timer_tuple_t =
    tim::component_tuple_t<wall_clock, cpu_clock, peak_rss, papi_tuple_t, likwid_marker>;
using global_tuple_t = tim::auto_tuple_t<wall_clock, cpu_clock, cpu_util, peak_rss,
                                         page_rss, papi_tuple_t, likwid_marker>;

static int64_t ncount       = 0;
static int64_t nmeasure     = 0;
static int64_t toolkit_size = 2;
using result_type           = std::tuple<timer_tuple_t, int64_t, int64_t>;

namespace mode
{
struct basic
{};
struct blank
{};
struct none
{};
struct basic_pointer
{};
struct blank_pointer
{};
struct invoke
{};
struct chained
{};
struct measure
{};
struct count
{};
struct single
{};
struct manual
{};
struct static_string
{};
}  // namespace mode

//======================================================================================//

static bool&
do_print_result()
{
    static bool _instance = true;
    return _instance;
}

//======================================================================================//

void
print_result(const std::string& prefix, int64_t result, int64_t num_meas,
             int64_t num_uniq)
{
    std::cout << std::setw(20) << prefix << " answer : " << result
              << " (# measurements: " << num_meas << ", # unique: " << num_uniq << ")"
              << std::endl;
}

//======================================================================================//

int64_t
fibonacci(int64_t n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t,
          tim::enable_if_t<std::is_same<Tp, mode::none>::value, int> = 0)
{
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::measure>::value, int> = 0)
{
    if(n > cutoff)
    {
        nmeasure += toolkit_size;
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::count>::value, int> = 0)
{
    ncount += 1;
    return (n < 2) ? n : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::static_string>::value, int> = 0)
{
    if(n > cutoff)
    {
        auto_tuple_t _marker{ tim::static_string{ __FUNCTION__ } };
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::blank>::value, int> = 0)
{
    if(n > cutoff)
    {
        TIMEMORY_BLANK_MARKER(auto_tuple_t, __FUNCTION__);
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::basic>::value, int> = 0)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_MARKER(auto_tuple_t, "[", n, "]");
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::blank_pointer>::value, int> = 0)
{
    if(n > cutoff)
    {
        TIMEMORY_BLANK_POINTER(auto_tuple_t, __FUNCTION__);
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::basic_pointer>::value, int> = 0)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_POINTER(auto_tuple_t, "[", n, "]");
        return (n < 2) ? n
                       : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::chained>::value, int> = 0)
{
    if(n > cutoff)
    {
        // TIMEMORY_STATIC_BASIC_CALIPER(fib, auto_tuple_t, "");
        // TIMEMORY_CALIPER_AUTO_SCOPE(fib);
        static auto _srcloc =
            TIMEMORY_SOURCE_LOCATION(TIMEMORY_CAPTURE_MODE(basic), "").get_captured();
        return auto_tuple_t{ _srcloc }
            .start()
            .execute([n, cutoff]() {
                return (n < 2) ? n
                               : (fibonacci<Tp>(n - 1, cutoff) +
                                  fibonacci<Tp>(n - 2, cutoff));
            })
            .stop()
            .return_result();
    }
    return fibonacci(n);
}

//======================================================================================//

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::invoke>::value, int> = 0)
{
    if(n > cutoff)
    {
        auto labeler = [](int _n) { return TIMEMORY_JOIN("", "fibonacci[", _n, "]"); };
        auto fib     = [](int _n, int _cutoff) {
            return (_n < 2) ? _n
                            : (fibonacci<Tp>(_n - 1, _cutoff) +
                               fibonacci<Tp>(_n - 2, _cutoff));
        };
        return tim::runtime::invoke<auto_tuple_t>(labeler(n), fib, n, cutoff);
    }
    return fibonacci(n);
}

//======================================================================================//

using single_tuple_t = tim::lightweight_tuple<wall_clock>;
static single_tuple_t single_sum{ "single" };

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::single>::value, int> = 0)
{
    single_tuple_t _scoped{};
    _scoped.start();
    auto _result =
        (n < 2) ? n : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    single_sum += _scoped.stop();
    return _result;
}

//======================================================================================//

static int64_t nseconds = 0;

template <typename Tp>
int64_t
fibonacci(int64_t n, int64_t cutoff,
          tim::enable_if_t<std::is_same<Tp, mode::manual>::value, int> = 0)
{
    using clock_type    = std::chrono::steady_clock;
    using duration_type = std::chrono::duration<clock_type::rep, std::nano>;

    auto _start =
        std::chrono::duration_cast<duration_type>(clock_type::now().time_since_epoch())
            .count();
    auto _result =
        (n < 2) ? n : (fibonacci<Tp>(n - 1, cutoff) + fibonacci<Tp>(n - 2, cutoff));
    nseconds +=
        std::chrono::duration_cast<duration_type>(clock_type::now().time_since_epoch())
            .count() -
        _start;
    return _result;
}

//======================================================================================//

template <typename Tp>
result_type
run(int64_t n, int64_t cutoff, bool store = true)
{
    // bool is_none  = std::is_same<Tp, mode::none>::value;
    bool is_blank = std::is_same<Tp, mode::blank>::value ||
                    std::is_same<Tp, mode::blank_pointer>::value ||
                    std::is_same<Tp, mode::chained>::value ||
                    std::is_same<Tp, mode::static_string>::value;
    bool is_basic = std::is_same<Tp, mode::basic>::value ||
                    std::is_same<Tp, mode::basic_pointer>::value ||
                    std::is_same<Tp, mode::invoke>::value;

    // bool        with_timing = !(is_none);
    // std::string space       = (with_timing) ? " " : "";
    auto signature =
        TIMEMORY_LABEL(" [with timing = ", tim::demangle(typeid(Tp).name()), "]");

    nmeasure = 0;
    fibonacci<mode::measure>(n, cutoff);

    timer_tuple_t timer{ signature, store };
    int64_t       result =
        timer.start().execute(&fibonacci<Tp>, n, cutoff, 0).stop().return_result();

    int64_t nuniq =
        (is_blank) ? ((n - cutoff) * toolkit_size) : (is_basic) ? nmeasure : 0;

    if(std::is_same<Tp, mode::single>::value || std::is_same<Tp, mode::manual>::value)
    {
        nuniq    = toolkit_size;
        nmeasure = ncount;
    }

    auto _alt = timer;
    if(do_print_result())
        print_result(signature, result, nmeasure, nuniq);
    return result_type(_alt, nmeasure, nuniq);
}

//======================================================================================//

template <typename Tp>
void
launch(const int nitr, const int nfib, const int cutoff, int64_t& ex_measure,
       int64_t& ex_unique, std::vector<timer_tuple_t>& timer_list)
{
    int64_t nmeas = 0;
    int64_t nuniq = 0;

    do_print_result() = true;
    for(int i = 0; i < nitr; ++i)
    {
        auto&& ret = run<Tp>(nfib, cutoff);
        if(i == 0)
        {
            timer_list.push_back(std::get<0>(ret));
            nmeas = std::get<1>(ret);
            nuniq = std::get<2>(ret);
            ex_measure += std::get<1>(ret);
            ex_unique += std::get<2>(ret);
            do_print_result() = false;
        }
        else
        {
            timer_list.back() += std::get<0>(ret);
        }
    }

    std::string prefix = std::to_string(nuniq) + " unique measurements and " +
                         std::to_string(nmeas) + " total measurements (" +
                         tim::demangle(typeid(Tp).name()) + ")";
    timer_list.push_back((timer_list.back() / nitr) - (timer_list.at(0) / nitr));
    timer_list.push_back(timer_list.back() / nmeas);
    timer_list.at(timer_list.size() - 2).rekey("difference vs. " + prefix);
    timer_list.at(timer_list.size() - 1).rekey("average overhead of " + prefix);
}
//======================================================================================//

int
main(int argc, char** argv)
{
    tim::settings::timing_scientific() = true;
    tim::settings::auto_output()       = true;
    tim::settings::json_output()       = false;
    tim::settings::text_output()       = true;
    tim::settings::memory_units()      = "kB";
    tim::settings::memory_precision()  = 3;
    tim::settings::width()             = 10;
    tim::settings::timing_precision()  = 3;

    using parser_t = tim::argparse::argument_parser;
    parser_t _parser{ "ex_cxx_overhead [<fibonnaci value> <fibonacci cutoff> <num "
                      "iterations> <warmup count>]" };

    tim::timemory_init(argc, argv);
    tim::timemory_argparse(&argc, &argv, &_parser);
    tim::settings::cout_output() = false;

    // default calc: fibonacci(43)
    int nfib = 35;
    if(argc > 1)
        nfib = std::stoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 15;
    if(argc > 2)
        cutoff = std::stoi(argv[2]);

    int nitr = 1;
    if(argc > 3)
        nitr = std::stoi(argv[3]);

    int nwarmup = std::max<int>(nitr, 2);
    if(argc > 4)
        nwarmup = std::stoi(argv[4]);

    std::vector<timer_tuple_t> timer_list{};
    std::cout << "\nRunning fibonacci(n = " << nfib << ", cutoff = " << cutoff << ")...\n"
              << std::endl;

    tim::auto_tuple<> empty_test("test");
    tim::consume_parameters(empty_test);

    std::cout << "[begin warmup]\n";
    do_print_result() = false;
    run<mode::count>(nfib, nfib, false);
    for(int i = 0; i < nwarmup; ++i)
    {
        auto warmup = run<mode::none>(nfib, nfib, false);
        std::cout << std::get<0>(warmup) << "\n";
    }
    std::cout << "[end warmup]\n";

    int64_t ex_measure = 0;
    int64_t ex_unique  = 0;

    TIMEMORY_CALIPER(global, global_tuple_t, "[", argv[0], "]");

    std::cout << std::endl;

    //----------------------------------------------------------------------------------//
    //      run without timing
    //----------------------------------------------------------------------------------//
    do_print_result() = true;
    for(int i = 0; i < nitr; ++i)
    {
        auto&& ret = run<mode::none>(nfib, nfib);
        if(i == 0)
        {
            timer_list.push_back(std::get<0>(ret));
            ex_measure += std::get<1>(ret);
            ex_unique += std::get<2>(ret);
            do_print_result() = false;
        }
        else
        {
            timer_list.at(0) += std::get<0>(ret);
        }
    }

    //----------------------------------------------------------------------------------//
    //      run various modes
    //----------------------------------------------------------------------------------//
    single_sum.push();

    launch<mode::manual>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::single>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::static_string>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::blank>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::blank_pointer>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::chained>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::basic>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::basic_pointer>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);
    launch<mode::invoke>(nitr, nfib, cutoff, ex_measure, ex_unique, timer_list);

    TIMEMORY_CALIPER_APPLY(global, stop);

    single_sum.pop();

    std::stringstream _sum_msg{};
    _sum_msg.setf(wall_clock::get_format_flags());
    _sum_msg << ">>> manual  : " << std::setprecision(wall_clock::get_precision())
             << (nseconds / static_cast<double>(std::nano::den)) << " sec\n";
    _sum_msg << single_sum << "\n";
    std::cout << "\n" << _sum_msg.str() << std::endl;
    assert(single_sum.get<wall_clock>()->get() > 0.0);

    std::cout << "\nReport from " << ex_measure << " total measurements and " << ex_unique
              << " unique measurements: " << std::endl;

    int nc = -1;

    // discarded print to set max width
    std::stringstream _discard{};
    for(auto& itr : timer_list)
        _discard << "    " << itr << "\n";

    for(auto& itr : timer_list)
    {
        std::cout << "    " << itr << std::endl;
        auto _nc = nc++;
        if(_nc % 3 == 2 || _nc < 0)
            std::cout << "\n";
    }

    auto l1_size  = tim::ert::cache_size::get<1>();
    auto l2_size  = tim::ert::cache_size::get<2>();
    auto l3_size  = tim::ert::cache_size::get<3>();
    auto max_size = tim::ert::cache_size::get_max();
    std::cout << "\n[INFO]> L1 cache size: " << (l1_size / tim::units::kilobyte)
              << " KB, L2 cache size: " << (l2_size / tim::units::kilobyte)
              << " KB, L3 cache size: " << (l3_size / tim::units::kilobyte)
              << " KB, max cache size: " << (max_size / tim::units::kilobyte) << " KB\n"
              << std::endl;

    // int ret = 0;

    if(!tim::settings::enabled())
    {
        printf("timemory was disabled.\n");
        // ret = EXIT_SUCCESS;
    }
    else if(tim::settings::flat_profile())
    {
        ex_unique = ((nfib - cutoff) + 1) * toolkit_size;
        int64_t rc_unique =
            (tim::storage<wall_clock>::instance()->size() - 6) * toolkit_size;
        printf("Expected size: %li, actual size: %li\n", (long) ex_unique,
               (long) rc_unique);
        // ret = (rc_unique == ex_unique) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    else
    {
        int64_t rc_unique =
            (tim::storage<wall_clock>::instance()->size() - 5) * toolkit_size - 8;
        printf("Expected size: %li, actual size: %li\n", (long) ex_unique,
               (long) rc_unique);
        // ret = (rc_unique == ex_unique) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    tim::timemory_finalize();

    return 0;
}
