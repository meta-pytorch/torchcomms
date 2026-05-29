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
#include "timemory/utility/signals.hpp"

#include <chrono>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;

// a bundle of components which starts via ctor and stops via dtor
using auto_tuple_t = tim::auto_tuple_t<wall_clock, cpu_clock, cpu_util, peak_rss,
                                       papi_vector, caliper, tau_marker>;
// the component_tuple equivalent (requires explicit start/stop)
using comp_tuple_t = typename auto_tuple_t::component_type;
// a bundle of components which are optional at runtime
using auto_list_t = tim::auto_list_t<wall_clock, cpu_clock, cpu_util, peak_rss, caliper,
                                     tau_marker, papi_vector>;
// a pre-configured auto_bundle
using auto_timer_t = tim::auto_timer;

// forward declaration
void
foo();
void
bar();
void
raz();
intmax_t
fibonacci(intmax_t n);

//
// Usage example:
//
//  ./ex_cxx_basic 10 15 20
//  ./ex_cxx_basic --timemory-enabled=no -- 10 15
//  ./ex_cxx_basic --timemory-enabled=no 10 15
//  ./ex_cxx_basic --timemory-option components="wall_clock,page_rss" precision=12 -- 10
//
int
main(int argc, char** argv)
{
    // default settings for PAPI
    if(tim::settings::papi_events().empty())
        tim::settings::papi_events() = "PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_LST_INS";

    // it is recommended to do this outside of the initializer (or make variables static).
    // If done inside the initializer, this will result in unnecessary overhead
    const std::string default_env = "wall_clock,cpu_clock,cpu_util,caliper,papi_vector";
    auto              env         = tim::get_env("TIMEMORY_COMPONENTS", default_env);
    auto              env_enum    = tim::enumerate_components(tim::delimit(env));

    // runtime customization of auto_list_t initialization
    auto_list_t::get_initializer() = [env_enum](auto& al) {
        tim::initialize(al, env_enum);
        al.report_at_exit(true);
    };

    // default settings which can be overridden by environment
    tim::settings::width()                 = 16;
    tim::settings::timing_units()          = "sec";
    tim::settings::timing_width()          = 16;
    tim::settings::timing_precision()      = 6;
    tim::settings::timing_scientific()     = false;
    tim::settings::memory_units()          = "KB";
    tim::settings::memory_width()          = 16;
    tim::settings::memory_precision()      = 3;
    tim::settings::memory_scientific()     = false;
    tim::settings::enable_signal_handler() = true;

    // initialize timemory
    tim::timemory_init(argc, argv);

    // parse the timemory-specific command line options (if any)
    // to distinguish timemory arguments from user arguments, use the syntax:
    //    ./<exe> <timemory-options> -- <user-options>
    // after the function before, argc and argv will result in:
    //    ./<exe> <user-options>
    // Using the syntaxes:
    //    ./<exe> <user-options>
    //    ./<exe> <timemory-options>
    //    ./<exe> <timemory-options> <user-options> (intermixed)
    // will not remove any timemory options.
    // In other words, only timemory_argparse will remove all arguments after
    // <exe> until the first "--" if reached and everything after the "--"
    // will be placed in argv[1:]
    tim::timemory_argparse(&argc, &argv);

    // print out the signal handler settings
    std::cout << tim::signals::signal_settings::str() << std::endl;

    //
    //  Provide some work
    //
    std::vector<long> fibvalues;
    for(int i = 1; i < argc; ++i)
    {
        if(std::string(argv[i]).find_first_not_of("0123456789") == std::string::npos)
            fibvalues.push_back(atol(argv[i]));
    }
    if(fibvalues.empty())
        fibvalues = { 10, 15, 20 };

    // use the auto-timer
    auto_timer_t main(argv[0]);
    for(auto n : fibvalues)
    {
        // create a caliper handle to an auto_tuple_t and have it report when destroyed
        TIMEMORY_BLANK_CALIPER(fib, auto_tuple_t, "fibonacci(", n, ")");
        TIMEMORY_CALIPER_APPLY(fib, report_at_exit, true);
        // run calculation
        auto ret = fibonacci(n);
        // manually stop the auto_tuple_t
        TIMEMORY_CALIPER_APPLY(fib, stop);
        // this is necessary to avoid optimizing away the fibonacci call
        printf("\nfibonacci(%li) = %li\n", n, (long int) ret);
    }
    // stop and print
    main.stop();
    std::cout << "\n" << main << std::endl;

    foo();
    bar();
    raz();

    // finalize timemory (produces output)
    tim::timemory_finalize();

    std::stringstream cmd;
    std::stringstream filler;
    std::stringstream breaker;

    cmd << "#  Command executed: \"";
    for(int i = 0; i < argc; ++i)
        cmd << argv[i] << ((i + 1 < argc) ? " " : "");
    cmd << "\"  #" << std::flush;
    breaker << "#" << std::setw(cmd.str().length() - 2) << ""
            << "#";
    filler.fill('#');
    filler << std::setw(cmd.str().length()) << "";
    std::cout << "\n\t" << filler.str() << "\n\t" << breaker.str() << "\n\t" << cmd.str()
              << "\n\t" << breaker.str() << "\n\t" << filler.str() << "\n\n"
              << std::flush;

    return 0;
}

// recursive instrumentation
intmax_t
fibonacci(intmax_t n)
{
    TIMEMORY_BASIC_MARKER(auto_tuple_t, "");
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

void
foo()
{
    // use the component tuple (does not auto-start/stop)
    comp_tuple_t ct("foo");
    ct.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ct.stop();
}

void
bar()
{
    auto_list_t al("bar");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void
raz()
{
    using int_dist_t = std::uniform_int_distribution<std::mt19937::result_type>;

    // convert the component tuple into a lightweight_tuple
    // the lightweight tuple is designed to provide as little implicit overhead
    // as possible. For example, it does not automatically, push and pop from
    // storage, thus start/stop will not show up in the report unless push/pop are called
    using lw_tuple_t = tim::convert_t<comp_tuple_t, tim::lightweight_tuple<>>;

    lw_tuple_t lw("raz");
    lw.push();
    lw.start();

    // consume some memory
    auto v   = std::vector<double>(100000);
    auto rng = std::mt19937{};
    rng.seed(std::random_device()());
    // consume some cpu-time
    std::generate(v.begin(), v.end(),
                  [&]() { return (std::generate_canonical<double, 10>(rng)); });
    // get a random index
    auto idx = v.at(int_dist_t(0, v.size() - 1)(rng));
    // scale up the random value to [100, 600) milliseconds
    auto val = static_cast<size_t>(v.at(idx) * 500 + 100);
    // consume some wall-time
    std::this_thread::sleep_for(std::chrono::milliseconds(val));

    lw.stop();
    lw.pop();
}
