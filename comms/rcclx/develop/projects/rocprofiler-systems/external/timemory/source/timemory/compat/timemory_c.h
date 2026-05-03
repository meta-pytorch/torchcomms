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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED)

#    define TIMEMORY_C_SETTINGS_INIT                                                     \
        {}
#    define TIMEMORY_C_INIT(...)
#    define TIMEMORY_C_AUTO_LABEL(...) ""
#    define TIMEMORY_C_BLANK_AUTO_TIMER(...) NULL
#    define TIMEMORY_C_BASIC_AUTO_TIMER(...) NULL
#    define TIMEMORY_C_AUTO_TIMER(...) NULL
#    define FREE_TIMEMORY_C_AUTO_TIMER(...)
#    define TIMEMORY_C_BASIC_AUTO_TUPLE(...) NULL
#    define TIMEMORY_C_BLANK_AUTO_TUPLE(...) NULL
#    define TIMEMORY_C_AUTO_TUPLE(...) NULL
#    define FREE_TIMEMORY_C_AUTO_TUPLE(...)

#else  // !defined(DISABLE_TIMEMORY)

#    include "timemory/compat/library.h"
#    include "timemory/compat/macros.h"
#    include "timemory/enum.h"

//======================================================================================//
//
//      C timemory macros
//
//======================================================================================//

#    define TIMEMORY_C_SETTINGS_INIT { 1, -1, -1, -1, -1, -1, -1, -1, -1 };
#    define TIMEMORY_C_INIT(argc, argv, settings) c_timemory_init(argc, argv, settings)
#    define TIMEMORY_C_FINALIZE() c_timemory_finalize()

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_C_BLANK_LABEL(c_str) c_str

#    define TIMEMORY_C_BASIC_LABEL(c_str) c_timemory_basic_label(__FUNCTION__, c_str)

#    define TIMEMORY_C_LABEL(c_str)                                                      \
        c_timemory_label(__FUNCTION__, __FILE__, __LINE__, c_str)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_C_BLANK_AUTO_TIMER(c_str)                                           \
        c_timemory_create_auto_timer(TIMEMORY_C_BLANK_LABEL(c_str))

#    define TIMEMORY_C_BASIC_AUTO_TIMER(c_str)                                           \
        c_timemory_create_auto_timer(TIMEMORY_C_BASIC_LABEL(c_str))

#    define TIMEMORY_C_AUTO_TIMER(c_str)                                                 \
        c_timemory_create_auto_timer(TIMEMORY_C_LABEL(c_str))

#    define FREE_TIMEMORY_C_AUTO_TIMER(ctimer)                                           \
        c_timemory_delete_auto_timer((void*) ctimer)

//--------------------------------------------------------------------------------------//

#    define TIMEMORY_C_BLANK_MARKER(c_str, ...)                                          \
        c_timemory_create_auto_tuple(TIMEMORY_C_BLANK_LABEL(c_str), __VA_ARGS__,         \
                                     TIMEMORY_COMPONENTS_END)

#    define TIMEMORY_C_BASIC_MARKER(c_str, ...)                                          \
        c_timemory_create_auto_tuple(TIMEMORY_C_BASIC_LABEL(c_str), __VA_ARGS__,         \
                                     TIMEMORY_COMPONENTS_END)

#    define TIMEMORY_C_MARKER(c_str, ...)                                                \
        c_timemory_create_auto_tuple(TIMEMORY_C_LABEL(c_str), __VA_ARGS__,               \
                                     TIMEMORY_COMPONENTS_END)

#    define FREE_TIMEMORY_C_MARKER(ctimer) c_timemory_delete_auto_tuple((void*) ctimer)

//--------------------------------------------------------------------------------------//

#endif  // !defined(DISABLE_TIMEMORY)
