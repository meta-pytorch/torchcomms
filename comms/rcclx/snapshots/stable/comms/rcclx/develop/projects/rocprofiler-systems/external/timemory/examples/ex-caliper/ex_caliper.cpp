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

#include "timemory/timemory.hpp"

#include <chrono>
#include <thread>

using namespace tim::component;

using auto_tuple_t =
    tim::auto_tuple_t<wall_clock, caliper_marker, user_clock, system_clock, cpu_util>;

using loop_tuple_t = tim::auto_tuple_t<caliper_loop_marker>;

intmax_t time_fibonacci(intmax_t);
intmax_t
fibonacci(intmax_t n);
intmax_t
ex_caliper(intmax_t n, const std::string& scope_tag);
void
print_info(const std::string& func, const std::string& scope_tag);

//======================================================================================//

int
main(int argc, char** argv)
{
    tim::settings::banner()            = false;
    tim::settings::timing_units()      = "sec";
    tim::settings::timing_width()      = 12;
    tim::settings::timing_precision()  = 6;
    tim::settings::timing_scientific() = false;
    tim::settings::memory_units()      = "KB";
    tim::settings::memory_width()      = 12;
    tim::settings::memory_precision()  = 3;
    tim::settings::memory_scientific() = false;
    tim::timemory_init(argc, argv);

    caliper_config cfg{};
    cfg.configure({ "mem.highwatermark" },
                  { { "runtime-report", "output=stdout" },
                    { "event-trace", "output=trace.cali,trace.io" } });

    cfg.start();

    std::vector<long> fibvalues;
    for(int i = 1; i < argc; ++i)
        fibvalues.push_back(atol(argv[i]));

    if(fibvalues.empty())
    {
        fibvalues.resize(tim::get_env<size_t>("NUM_FIBONACCI", 10));
        long n = tim::get_env("FIBONACCI_MIN", 33);
        std::generate_n(fibvalues.data(), fibvalues.size(), [&]() { return n++; });
    }

    // create a component tuple (does not auto-start)
    auto execute_test = [&](const std::string& scope_tag) {
        printf("caliper: scope = '%s', attributes = %i\n", scope_tag.c_str(),
               caliper::get_attributes());
        print_info("execute_test", scope_tag);
        intmax_t     ret = 0;
        loop_tuple_t _loop{ "execute_test" };
        for(auto n : fibvalues)
        {
            _loop.mark_begin();
            ret += ex_caliper(n, scope_tag);
            _loop.mark_end();
        }
        std::cout << "fibonacci " << scope_tag << " : " << ret << std::endl;
    };

    auto scope = tim::get_env<std::string>("CALIPER_SCOPE", "process");

    if(scope == "process")
    {
        caliper_common::enable_process_scope();
        execute_test("process");
    }
    else if(scope == "thread")
    {
        caliper_common::enable_thread_scope();
        execute_test("thread");
    }
    else if(scope == "task")
    {
        caliper_common::enable_task_scope();
        execute_test("task");
    }
    else
    {
        throw std::runtime_error(
            "Error!! CALIPER_SCOPE must be one of: 'process', 'thread', 'task'");
    }

    cfg.stop();

    tim::timemory_finalize();
}

//======================================================================================//

intmax_t
fibonacci(intmax_t n)
{
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

//======================================================================================//

intmax_t
time_fibonacci(intmax_t n, const std::string& scope_tag, const std::string& type_tag)
{
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "[", scope_tag, "-", type_tag, "]");
    return fibonacci(n);
}

//======================================================================================//

void
print_info(const std::string& func, const std::string& scope_tag)
{
    if(tim::dmp::rank() == 0)
    {
        std::cout << "[" << tim::dmp::rank() << "]\e[1;33m TESTING \e[0m["
                  << "\e[1;36m" << func << " ["
                  << "\e[1;37m"
                  << "scope: " << scope_tag << "\e[1;36m"
                  << "]"
                  << "\e[0m"
                  << "]..." << std::endl;
    }
}

//======================================================================================//

intmax_t
ex_caliper(intmax_t nfib, const std::string& scope_tag)
{
    std::atomic<int64_t> ret;
    // accumulate metrics on full run
    TIMEMORY_BASIC_CALIPER(tot, auto_tuple_t, "[total-", scope_tag, "-scope]");

    // run a fibonacci calculation and accumulate metric
    auto run_fibonacci = [&](long n, const std::string& type_tag) {
        ret += time_fibonacci(n, scope_tag, type_tag);
    };

    // run longer fibonacci calculations on two threads
    TIMEMORY_BASIC_CALIPER(worker_thread, auto_tuple_t, "[worker-thread-", scope_tag,
                           "-scope]");
    std::thread t(run_fibonacci, nfib, "worker");
    TIMEMORY_CALIPER_APPLY(worker_thread, stop);

    // run shorter fibonacci calculation on main thread
    TIMEMORY_BASIC_CALIPER(master_thread, auto_tuple_t, "[master-thread-", scope_tag,
                           "-scope]");
    run_fibonacci(nfib - 1, "master");
    TIMEMORY_CALIPER_APPLY(master_thread, stop);

    // wait to finish
    t.join();

    // stop total
    TIMEMORY_CALIPER_APPLY(tot, stop);

    return ret.load();
}

//======================================================================================//
