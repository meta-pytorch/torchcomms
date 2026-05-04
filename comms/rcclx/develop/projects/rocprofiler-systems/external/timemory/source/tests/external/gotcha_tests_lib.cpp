// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include "gotcha_tests_lib.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>

extern "C"
{
    double test_exp(double val) { return exp(val); }
}
//

namespace ext
{
//--------------------------------------------------------------------------------------//

void
do_puts(const char* msg)
{
    puts(msg);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Func, typename _Incr>
_Tp
work(int64_t nitr, _Func&& func, _Incr&& incr)
{
    _Tp val = 2.0;
    _Tp sum = 0.0;

    for(int64_t i = 0; i < nitr; ++i)
    {
        sum += func(val);
        val = incr(val, i + 1);
    }
    return sum;
}

//--------------------------------------------------------------------------------------//

std::tuple<float, double>
do_work(int64_t nitr, const std::pair<float, double>& p)
{
    auto fsum = work<float>(
        nitr, [](float val) -> float { return cosf(val); },
        [&](float val, int64_t i) -> float { return val + p.first * i; });

    auto dsum = work<double>(
        nitr, [](double val) -> double { return cos(val); },
        [&](double val, int64_t i) -> double { return val + p.second * i; });

    return std::tuple<float, double>(fsum, dsum);
}

//--------------------------------------------------------------------------------------//

}  // namespace ext

//--------------------------------------------------------------------------------------//

DoWork::DoWork(std::pair<float, double> pair)
: m_pair(std::move(pair))
, m_tuple{ 0.0f, 0.0 }
{}

//--------------------------------------------------------------------------------------//

void
DoWork::execute_fp4(int64_t nitr)
{
    std::get<0>(m_tuple) = ext::work<float>(
        nitr, [](float val) -> float { return cosf(val); },
        [&](float val, int64_t i) -> float { return val + m_pair.first * i; });
}

//--------------------------------------------------------------------------------------//

void
DoWork::execute_fp8(int64_t nitr)
{
    std::get<1>(m_tuple) = ext::work<double>(
        nitr, [](double val) -> double { return cos(val); },
        [&](double val, int64_t i) -> double { return val + m_pair.second * i; });
}

//--------------------------------------------------------------------------------------//

void
DoWork::execute_fp(int64_t nitr, const std::vector<float>& fvals,
                   const std::deque<double>& dvals)
{
    float fret = 0.0;
    for(const auto& itr : fvals)
    {
        fret += ext::work<float>(
            nitr, [](float val) -> float { return cosf(val); },
            [&](float val, int64_t i) -> float { return val + itr * i; });
    }
    std::get<0>(m_tuple) = fret;

    double dret = 0.0;
    for(const auto& itr : dvals)
    {
        dret += ext::work<double>(
            nitr, [](double val) -> double { return cos(val); },
            [&](double val, int64_t i) -> double { return val + itr * i; });
    }
    std::get<1>(m_tuple) = dret;
}

//--------------------------------------------------------------------------------------//

std::tuple<float, double>
DoWork::get() const
{
    return m_tuple;
}

//--------------------------------------------------------------------------------------//
