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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file "timemory/utility/functional.hpp"
 * This provides functions on types
 *
 */

#pragma once

//----------------------------------------------------------------------------//

#include "timemory/utility/macros.hpp"

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>

namespace tim
{
namespace func
{
template <bool _Condition, typename Tp = int>
using enable_if_t = typename std::enable_if<(_Condition), Tp>::type;

using std::max;
using std::min;

template <typename Lhs, typename Rhs>
struct add
{
    add(Lhs& lhs, const Rhs& rhs) { lhs += rhs; }
};

template <typename Lhs, typename Rhs>
struct subtract
{
    subtract(Lhs& lhs, const Rhs& rhs) { lhs -= rhs; }
};

template <typename Lhs, typename Rhs>
struct multiply
{
    multiply(Lhs& lhs, const Rhs& rhs) { lhs *= rhs; }
};

template <typename Lhs, typename Rhs>
struct divide
{
    divide(Lhs& lhs, const Rhs& rhs) { lhs /= rhs; }
};

}  // namespace func

}  // namespace tim
