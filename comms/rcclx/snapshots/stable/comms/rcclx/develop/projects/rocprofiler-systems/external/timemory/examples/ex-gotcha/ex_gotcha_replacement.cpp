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

#if defined(DEBUG) && !defined(VERBOSE)
#    define VERBOSE
#endif

#pragma GCC diagnostic ignored "-Wignored-attributes"

#include "ex_gotcha_lib.hpp"
#include "timemory/timemory.hpp"

#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace tim;
using namespace tim::component;
using std::cout;
using std::vector;

//======================================================================================//

static auto&
get_intercepts()
{
    static uint64_t _instance{ 0 };
    return _instance;
}

//======================================================================================//

namespace tim
{
namespace component
{
struct exp_replace : public base<exp_replace, void>
{
    double operator()(double val)
    {
        message(val);
        ++get_intercepts();
        return static_cast<double>(expf(static_cast<float>(val)));
    }

    static void message(double val)
    {
#if defined(VERBOSE)
        static bool _verbose = (tim::settings::verbose() > 1 || tim::settings::debug());
        if(_verbose) printf("\texecuting modified exp function : %20.3f...", val);
#else
        (void) val;
#endif
    }
};
}  // namespace component
}  // namespace tim

//======================================================================================//

using wc_t         = tim::component_bundle<TIMEMORY_API, wall_clock>;
using exp_time_t   = gotcha<2, wc_t>;
using exp_repl_t   = gotcha<2, std::tuple<>, exp_replace>;
using exp_bundle_t = tim::component_bundle<TIMEMORY_API, exp_time_t*, exp_repl_t*>;

using exp2expf_ot = typename exp_repl_t::operator_type;
static_assert(std::is_same<exp2expf_ot, exp_replace>::value,
              "exp_repl_t operator_type is not exp_replace");
static_assert(exp_repl_t::components_size == 0, "exp_repl_t should have no components");
static_assert(exp_repl_t::replaces, "exp_repl_t does not replace exp!");

using exptime_ot = typename exp_time_t::operator_type;
using exptime_ct = typename exp_time_t::bundle_type;
static_assert(std::is_same<exptime_ot, void>::value,
              "exp_time_t operator_type is not exp_replace");
static_assert(exp_time_t::components_size == 1, "exp_time_t should have no components");
static_assert(std::is_same<exptime_ct, wc_t>::value,
              "exp_time_t has incorrect components");
static_assert(!exp_time_t::replaces, "exp_time_t replaces exp!");

//======================================================================================//

namespace
{
auto gotcha_debug_lvl = tim::get_env<int>("GOTCHA_DEBUG", 0);
}  // namespace

//======================================================================================//

bool
init_gotcha(bool use_intercept, bool use_timers)
{
    printf("gotcha debug level: %i\n", gotcha_debug_lvl);

    //
    // configure the initializer for the gotcha component which replaces exp with expf
    //
    exp_repl_t::get_initializer() = []() {
        puts("Generating exp intercept...");
#if defined(TIMEMORY_HIPCC)
        exp_repl_t::template instrument<0, double, double>::generate("exp");
        exp_repl_t::template instrument<1, double, double>::generate("__exp_finite");
#else
        TIMEMORY_C_GOTCHA(exp_repl_t, 0, exp);
        TIMEMORY_DERIVED_GOTCHA(exp_repl_t, 1, exp, "__exp_finite");
#endif
        // exp might actually resolve to above symbol
    };

    //
    // configure the initializer for the gotcha components which places wall-clock
    // timers around exp and sum_exp
    //
    exp_time_t::get_initializer() = []() {
        puts("Generating exp timers...");
#if defined(TIMEMORY_HIPCC)
        exp_time_t::template instrument<0, double, double>::generate("exp");
#else
        TIMEMORY_C_GOTCHA(exp_time_t, 0, exp);
#endif
    };

    exp_bundle_t::get_initializer() = [=](exp_bundle_t& obj) {
        if(use_timers)
        {
            puts("initializing timers for exp...");
            obj.initialize<wall_clock, exp_time_t>();
        }
        if(use_intercept)
        {
            puts("initializing intercepts for exp...");
            obj.initialize<wall_clock, exp_repl_t>();
        }
    };

    return true;
}

//======================================================================================//

double
sum_exp(const std::vector<double>& data)
{
    static uint64_t cnt = 0;
    auto            ret = double{};
    auto            rng = std::mt19937{};
    rng.seed(std::random_device()());
    for(const auto& itr : data)
    {
        // without randomness, gcc will elide exp calculations
        ret += exp(itr + (0.1 * std::generate_canonical<double, 10>(rng)));
        ++cnt;
    }
    printf("Iterations: %lu, intercepts: %lu\n", cnt, get_intercepts());
    return ret;
}

//======================================================================================//

int
main(int argc, char** argv)
{
    puts("starting...");
    auto use_intercept = tim::get_env("EXP_REPLACE", true);
    auto use_timers    = tim::get_env("EXP_WRAPPER", !use_intercept);
    if(!init_gotcha(use_intercept, use_timers))
        throw std::runtime_error("Error! initialization failed!");

    tim::settings::file_output() = false;
    tim::timemory_init(argc, argv);

    uint64_t n = 1000;
    double   a = 2.0;
    double   b = 1.0;
    if(argc > 1) n = atoi(argv[1]);
    if(argc > 2) a = atof(argv[2]);
    if(argc > 3) b = atof(argv[3]);

    double       ret = 0.0;
    exp_bundle_t obj{ "example" };
    puts("starting iterations...");
    for(uint64_t i = 0; i < n; ++i)
    {
        obj.start();
        // this should be wrapped
        ret += sum_exp({ i + b, a * (i + b) });
        obj.stop();
        // this should not be wrapped
        ret += sum_exp({ i + b + 0.1, a * (i + b + 0.1) });
    }
    puts("iterations completed...");

    auto sz = tim::storage<wall_clock>::instance()->size();
    std::cout << "\nusing intercept    : " << std::boolalpha << use_intercept << '\n';
    std::cout << "exp -> expf        : " << get_intercepts() << "x" << std::endl;
    std::cout << "using timers       : " << std::boolalpha << use_timers << '\n';
    std::cout << "wall_clock records : " << sz << '\n' << std::endl;

    tim::timemory_finalize();

    auto rc_intercept =
        (use_intercept && get_intercepts() != 2 * n) ? EXIT_FAILURE : EXIT_SUCCESS;
    auto rc_timers = (use_timers && sz == 0) ? EXIT_FAILURE : EXIT_SUCCESS;

    printf("[%s] exit code: %i\n", argv[0], rc_intercept + rc_timers);
    return rc_intercept + rc_timers;
}

//======================================================================================//
