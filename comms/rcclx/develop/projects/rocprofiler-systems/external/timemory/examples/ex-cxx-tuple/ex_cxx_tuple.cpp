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

#define TIMEMORY_INPUT_ARCHIVE ::tim::cereal::XMLInputArchive
#define TIMEMORY_OUTPUT_ARCHIVE ::tim::cereal::XMLOutputArchive

#include "timemory/mpl/types.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/testing.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iterator>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace tim::stl;
using namespace tim::component;

using papi_tuple_t = papi_tuple<PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_LST_INS>;

using auto_tuple_t =
    tim::auto_tuple_t<wall_clock, system_clock, thread_cpu_clock, thread_cpu_util,
                      process_cpu_clock, process_cpu_util, papi_tuple_t, tau_marker>;

using comp_tuple_t = tim::convert_t<auto_tuple_t, tim::component_tuple<>>;

using measurement_t =
    tim::component_tuple_t<peak_rss, page_rss, virtual_memory, num_major_page_faults,
                           num_minor_page_faults, priority_context_switch,
                           voluntary_context_switch, tau_marker>;

//--------------------------------------------------------------------------------------//
// fibonacci calculation
int64_t
fibonacci(int32_t n)
{
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

void
print_info(const std::string&);
void
print_string(const std::string& str);
void
test_1_usage();
void
test_2_timing();
void
test_3_measure();
void
print_mpi_storage();

//======================================================================================//

int
main(int argc, char** argv)
{
    tim::settings::banner()      = true;
    tim::settings::json_output() = true;
    tim::signals::enable_signal_detection();
    tim::dmp::initialize(argc, argv);
    tim::timemory_init(argc, argv);

    tim::component_tuple_t<papi_tuple_t> m("PAPI measurements");
    m.start();

    CONFIGURE_TEST_SELECTOR(3);

    int num_fail = 0;
    int num_test = 0;

    std::cout << "# tests: " << tests.size() << std::endl;
    try
    {
        RUN_TEST(1, test_1_usage, num_test, num_fail);
        RUN_TEST(2, test_2_timing, num_test, num_fail);
        RUN_TEST(3, test_3_measure, num_test, num_fail);
    } catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    m.stop();

    std::cout << "\n" << m << std::endl;

    TEST_SUMMARY(argv[0], num_test, num_fail);

    print_mpi_storage();

    tim::timemory_finalize();
    tim::dmp::finalize();

    exit(num_fail);
}

//======================================================================================//

void
print_info(const std::string& func)
{
    if(tim::dmp::rank() == 0)
    {
        std::cout << "\n[" << tim::dmp::rank() << "]\e[1;33m TESTING \e[0m["
                  << "\e[1;36m" << func << "\e[0m"
                  << "]...\n"
                  << std::endl;
    }
}

//======================================================================================//

void
print_string(const std::string& str)
{
    std::stringstream _ss;
    _ss << "[" << tim::dmp::rank() << "] " << str << std::endl;
    std::cout << _ss.str();
}

//======================================================================================//

template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

//======================================================================================//

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

//======================================================================================//

void
test_1_usage()
{
    print_info(__FUNCTION__);
    auto test_1_marker = TIMEMORY_HANDLE(auto_tuple_t, "");

    measurement_t _use_beg("test_1_usage_begin");
    measurement_t _use_delta("test_1_usage_delta");
    measurement_t _use_end("test_1_usage_end");

    auto n = 5000000;
    _use_beg.record();
    _use_delta.start();
    std::vector<int64_t> v(n, 30);
    long                 nfib = random_entry(v);
    time_fibonacci(nfib);
    _use_delta.stop();
    _use_end.record();

    std::cout << "usage (begin): " << _use_beg << std::endl;
    std::cout << "usage (delta): " << _use_delta << std::endl;
    std::cout << "usage (end):   " << _use_end << std::endl;

    std::vector<std::pair<std::string, measurement_t>> measurements = {
        { "begin", _use_beg },
        { "delta", _use_delta },
        { "end", _use_end }
    };
    // serialize("rusage.json", "usage", measurements);
    test_1_marker.stop();
}

//======================================================================================//

void
test_2_timing()
{
    print_info(__FUNCTION__);

    comp_tuple_t         runtime(TIMEMORY_JOIN("_", __func__, "runtime"),
                         tim::scope::config{ tim::scope::tree{} });
    std::atomic<int64_t> ret;
    {
        TIMEMORY_MARKER(auto_tuple_t, "");

        auto run_fib = [&](long n) {
            TIMEMORY_BLANK_MARKER(auto_tuple_t, "run_fib/", n);
            ret += time_fibonacci(n);
        };

        std::vector<std::thread> threads;
        runtime.start();
        {
            for(int i = 0; i < 7; ++i)
                threads.emplace_back(run_fib, 35);
            threads.emplace_back(run_fib, 43);

            run_fib(40);

            for(auto& itr : threads)
                itr.join();
        }
        runtime.stop();
    }

    print_string(TIMEMORY_JOIN("/", __FUNCTION__, __LINE__));
    std::cout << "total runtime: " << runtime << std::endl;
    std::cout << "std::get: " << std::get<0>(runtime) << std::endl;
    std::cout << "fibonacci total: " << ret.load() << "\n" << std::endl;
    std::cout << "runtime process cpu time: " << runtime.get<process_cpu_clock>() << "\n";
    print_string(TIMEMORY_JOIN("/", __FUNCTION__, __LINE__));
}

//======================================================================================//

void
test_3_measure()
{
    print_info(__FUNCTION__);

    tim::component_tuple<page_rss, peak_rss> prss(TIMEMORY_LABEL(""));
    {
        TIMEMORY_VARIADIC_BASIC_AUTO_TUPLE("[init]", page_rss, peak_rss);
        // just record the peak rss
        prss.measure();
        std::cout << "  Current rss: " << prss << std::endl;
    }

    {
        TIMEMORY_VARIADIC_AUTO_TUPLE("[delta]", page_rss, peak_rss);
        // do something, where you want delta peak rss
        auto                 n = 10000000;
        std::vector<int64_t> v(n, 10);
        long                 nfib = random_entry(v);
        fibonacci(nfib);
        prss.stop();
        std::cout << "Change in rss: " << prss << std::endl;
    }

    // prss.reset();
    prss.measure();
    std::cout << "  Current rss: " << prss << std::endl;
}

//======================================================================================//

void
print_mpi_storage()
{
    auto ret = tim::storage<tim::component::wall_clock>::instance()->mpi_get();
    if(tim::dmp::rank() != 0)
        return;

    uint64_t _w = 0;
    for(const auto& jitr : ret)
        for(const auto& iitr : jitr)
        {
            _w = std::max<uint64_t>(_w, iitr.prefix().length());
        }

    for(uint64_t j = 0; j < ret.size(); ++j)
    {
        std::cout << "[RANK: " << j << "]\n";
        std::stringstream ss;
        for(const auto& itr : ret[j])
        {
            ss << "\t" << std::setw(_w) << std::left << itr.prefix() << " : "
               << itr.data() << "\n";
        }
        std::cout << ss.str();
    }
}

//======================================================================================//
