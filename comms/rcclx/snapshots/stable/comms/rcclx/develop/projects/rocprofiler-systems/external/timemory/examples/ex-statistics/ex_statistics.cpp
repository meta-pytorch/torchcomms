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
// include these headers for declaration w/o instantiation
//
#include "timemory/mpl/macros.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DEFINE_CONCRETE_TRAIT(pretty_archive, void, true_type)

//
// include these headers for declaration w/o instantiation
//
#include "timemory/components/types.hpp"

//
// configure these two types to always record statistics
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(flat_storage, component::monotonic_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::wall_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::cpu_clock, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::current_peak_rss, true_type)

//--------------------------------------------------------------------------------------//
// include rest of headers
//
#include "timemory/timemory.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>

//
// shorthand
//
using namespace tim::component;
using namespace tim;

//
// bundle of tools
//
using tuple_t = auto_tuple_t<wall_clock, monotonic_clock, cpu_clock, current_peak_rss>;

//--------------------------------------------------------------------------------------//

long
fib(long n)
{
    return (n < 2) ? n : (fib(n - 1) + fib(n - 2));
}

//--------------------------------------------------------------------------------------//

long
inst_fib(long n, bool encode_n)
{
    TIMEMORY_BASIC_MARKER(tuple_t, (encode_n) ? std::to_string(n) : "");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return (n < 2) ? n : (inst_fib(n - 1, encode_n) + fib(n - 2));
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    settings::destructor_report() = true;
    tim::timemory_init(argc, argv);
    tim::signals::enable_signal_detection();

    long nfib = (argc > 1) ? atol(argv[1]) : 40;
    int  nitr = (argc > 2) ? atoi(argv[2]) : 10;

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> size_dist(50000, 1000000);
    std::uniform_int_distribution<> value_dist(0, 1000000);

    for(int i = 0; i < nitr; ++i)
    {
        TIMEMORY_MARKER(tuple_t, "total");
        long ans = fib(nfib);

        TIMEMORY_BLANK_MARKER(tuple_t, "nested/", i % 5);
        ans += fib(nfib + (i % 3));

        // allocate some memory
        std::vector<int32_t> v(size_dist(gen), value_dist(gen));
        // use the memory
        ans *= v.at(value_dist(gen) % v.size());

        // generate a complex tail of instrumentation
        ans += inst_fib((nfib / ((i % 3) + 2)) + (nfib % (i + 1)), false);

        // occasionally create markers
        TIMEMORY_CONDITIONAL_BASIC_MARKER(i % 3 == 2, tuple_t, "occasional/", i % 3);
        ans += fib(nfib - 1);
        if(i % 3 == 2)
            ans += inst_fib(5 + (i % 4), true);

        printf("Answer = %li\n", ans);

        if(i > 0)
            tim::settings::destructor_report() = false;
    }

    tim::timemory_finalize();
    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//
