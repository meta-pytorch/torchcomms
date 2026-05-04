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

#include "timemory/timemory.hpp"

#include <cstdio>
#include <cstdlib>

//
// shorthand
//
using namespace tim::component;
using tim::type_list;
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DECLARE_COMPONENT(derived_cpu_util)
TIMEMORY_STATISTICS_TYPE(component::derived_cpu_util, double)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::derived_cpu_util, true_type)

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
template <>
struct derivation_types<derived_cpu_util>
{
    using type = type_list<type_list<wall_clock, cpu_clock>,
                           type_list<wall_clock, user_clock, system_clock>>;
};
}  // namespace trait

namespace component
{
struct derived_cpu_util : public base<derived_cpu_util, double>
{
    using ratio_t    = std::nano;
    using value_type = double;
    using this_type  = derived_cpu_util;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "derived_cpu_util"; }
    static std::string description() { return "cpu utilization (derived)"; }

    double get() const { return base_type::load(); }
    double get_display() const { return get(); }
    void   start() {}
    void   stop() {}

    bool derive(const wall_clock* wc, const cpu_clock* cc)
    {
        // DEBUG_PRINT_HERE("%s: %p %p", "successful derivation", wc, cc);
        if(wc && cc)
        {
            value = 100.0 * (cc->get() / wc->get());
            accum += value;
            return true;
        }
        return false;
    }

    bool derive(const wall_clock* wc, const user_clock* uc, const system_clock* sc)
    {
        // DEBUG_PRINT_HERE("%s: %p %p %p", "successful derivation", wc, uc, sc);
        if(wc && uc && sc)
        {
            value = 100.0 * ((uc->get() + sc->get()) / wc->get());
            accum += value;
            return true;
        }
        return false;
    }
};
}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
// bundle of tools
//
using pair_tuple_t = tim::auto_tuple<derived_cpu_util, wall_clock, cpu_clock>;
using pair_list_t  = tim::auto_list<derived_cpu_util, wall_clock, cpu_clock, peak_rss>;

using triplet_tuple_t =
    tim::auto_tuple<derived_cpu_util, wall_clock, user_clock, system_clock, peak_rss>;
using triplet_list_t =
    tim::auto_list<derived_cpu_util, wall_clock, user_clock, system_clock>;

//--------------------------------------------------------------------------------------//

long
fib(long n)
{
    return (n < 2) ? n : (fib(n - 1) + fib(n - 2));
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::timemory_init(argc, argv);
    tim::settings::destructor_report() = true;
    tim::settings::width()             = 12;

    long nfib = (argc > 1) ? atol(argv[1]) : 40;
    int  nitr = (argc > 2) ? atoi(argv[2]) : 10;

    // initially, provide all necessary info to compute
    pair_list_t::get_initializer() = [&](pair_list_t& pl) {
        pl.initialize<wall_clock, cpu_clock, derived_cpu_util>();
    };

    // initially, don't provide system_clock
    triplet_list_t::get_initializer() = [&](triplet_list_t& tl) {
        tl.initialize<wall_clock, user_clock, derived_cpu_util>();
    };

    for(int i = 0; i < nitr; ++i)
    {
        // should always report derived_cpu_util
        TIMEMORY_BLANK_MARKER(pair_tuple_t, "pair_tuple     ");
        long ans = fib(nfib);

        // should report for first half of iterations
        TIMEMORY_BLANK_MARKER(pair_list_t, "pair_list", i, "     ");
        ans += fib(nfib + (i % 3));

        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        {
            // should always report derived_cpu_util
            TIMEMORY_BLANK_MARKER(triplet_tuple_t, "triplet_tuple/", i);
            ans += fib(nfib - 1);
        }

        {
            // should report for second half of iterations
            TIMEMORY_BLANK_MARKER(triplet_list_t, "triplet_list/", i, " ");
            ans += fib(nfib - 1);

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        printf("\nAnswer = %li\n\n", ans);
        if(i == (nitr - 1) / 2)
        {
            // add system_clock to initialization
            triplet_list_t::get_initializer() = [&](triplet_list_t& tl) {
                tl.initialize<wall_clock, user_clock, system_clock, derived_cpu_util>();
            };

            // remove derived_cpu_util from initialization
            pair_list_t::get_initializer() = [&](pair_list_t& pl) {
                pl.initialize<wall_clock, cpu_clock>();
            };
        }
    }

    tim::timemory_finalize();
    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//
