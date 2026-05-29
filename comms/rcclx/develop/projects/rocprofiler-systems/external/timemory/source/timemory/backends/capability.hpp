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

#pragma once

#include "timemory/defines.h"

#include <sched.h>
#include <string>
#include <vector>

namespace timemory
{
namespace linux
{
namespace capability
{
using cap_value_t = int;

struct cap_info
{
    const char* name  = nullptr;
    cap_value_t value = -1;
};

struct cap_status
{
    unsigned long long inherited = 0;
    unsigned long long permitted = 0;
    unsigned long long effective = 0;
    unsigned long long bounding  = 0;
    unsigned long long ambient   = 0;
};

cap_status cap_read(pid_t);

std::string cap_name(cap_value_t);

std::vector<cap_value_t>
cap_decode(unsigned long long);

std::vector<cap_value_t>
cap_decode(const char*);
}  // namespace capability
}  // namespace linux
}  // namespace timemory
