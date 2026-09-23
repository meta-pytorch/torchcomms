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

/** \file timemory/timemory.h
 * \headerfile timemory/timemory.h "timemory/timemory.h"
 * Generic header for C and/or C++
 *
 */

#pragma once

#include "timemory/version.h"

#if defined(__cplusplus)
#    include "timemory/timemory.hpp"
#else
#    include "timemory/compat/timemory_c.h"

#    define TIMEMORY_SETTINGS_INIT TIMEMORY_C_SETTINGS_INIT
#    define TIMEMORY_INIT(...) TIMEMORY_C_INIT(__VA_ARGS__)
#    define TIMEMORY_FINALIZE(...) TIMEMORY_C_FINALIZE()

#    define TIMEMORY_AUTO_LABEL(...) TIMEMORY_C_AUTO_LABEL(__VA_ARGS__)

#    define TIMEMORY_BLANK_AUTO_TIMER(...) TIMEMORY_C_BLANK_AUTO_TIMER(__VA_ARGS__)
#    define TIMEMORY_BASIC_AUTO_TIMER(...) TIMEMORY_C_BASIC_AUTO_TIMER(__VA_ARGS__)
#    define TIMEMORY_AUTO_TIMER(...) TIMEMORY_C_AUTO_TIMER(__VA_ARGS__)
#    define FREE_TIMEMORY_AUTO_TIMER(...) FREE_TIMEMORY_C_AUTO_TIMER(__VA_ARGS__)

#    define TIMEMORY_BLANK_MARKER(...) TIMEMORY_C_BLANK_MARKER(__VA_ARGS__)
#    define TIMEMORY_BASIC_MARKER(...) TIMEMORY_C_BASIC_MARKER(__VA_ARGS__)
#    define TIMEMORY_MARKER(...) TIMEMORY_C_MARKER(__VA_ARGS__)
#    define FREE_TIMEMORY_MARKER(...) FREE_TIMEMORY_C_MARKER(__VA_ARGS__)

#endif
