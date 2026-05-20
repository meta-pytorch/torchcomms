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

#pragma once

#include <cstdint>
#include <deque>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

extern "C"
{
    extern double test_exp(double);
}

namespace ext
{
std::tuple<float, double>
do_work(int64_t, const std::pair<float, double>&);

void
do_puts(const char*);
}  // namespace ext

class DoWork
{
public:
    DoWork(std::pair<float, double>);

    void execute_fp4(int64_t);
    void execute_fp8(int64_t);
    void execute_fp(int64_t, const std::vector<float>&, const std::deque<double>&);
    std::tuple<float, double> get() const;  // NOLINT

    friend std::ostream& operator<<(std::ostream& os, const DoWork& obj)
    {
        os << &obj;
        return os;
    }

private:
    std::pair<float, double>  m_pair;
    std::tuple<float, double> m_tuple;
};
