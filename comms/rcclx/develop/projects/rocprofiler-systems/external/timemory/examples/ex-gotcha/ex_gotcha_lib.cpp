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

#include "ex_gotcha_lib.hpp"

#include "timemory/utility/demangle.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>

namespace ext
{
//--------------------------------------------------------------------------------------//

template <typename Tp, typename FuncT, typename IncrT>
Tp
work(const std::string& fname, int nitr, FuncT&& func, IncrT&& incr)
{
#if !defined(VERBOSE)
    tim::consume_parameters(fname);
#endif

    Tp val = 2.0;
    Tp sum = 0.0;

    for(int i = 0; i < nitr; ++i)
    {
        sum += func(val);
#if defined(VERBOSE)
        printf("\t[itr: %2i]> %-6s %-4s(%7.3f) = %20.3f\n", i,
               tim::demangle<Tp>().c_str(), fname.c_str(), val, sum);
#endif
        val = incr(val, i + 1);
    }
    return sum;
}

//--------------------------------------------------------------------------------------//

tuple_t
do_exp_work(int nitr)
{
#if defined(VERBOSE)
    printf("\n");
    PRINT_HERE("%s", "");
    printf("\n");
#endif

    auto fsum = work<float>(
        "expf", nitr, [](float val) -> float { return expf(val); },
        [](float val, int i) -> float { return val + 0.25 * i; });

#if defined(VERBOSE)
    printf("\n");
#endif

    auto dsum = work<double>(
        "exp", nitr, [](double val) -> double { return exp(val); },
        [](double val, int i) -> double { return val + 0.25 * i; });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return tuple_t(fsum, dsum);
}

//--------------------------------------------------------------------------------------//

}  // namespace ext
