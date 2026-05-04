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

#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif

#if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED) ||                           \
    (defined(TIMEMORY_ENABLED) && TIMEMORY_ENABLED == 0) ||                              \
    defined(TIMEMORY_DISABLE_MALLOCP)

    static inline void     timemory_mallocp_library_ctor() {}
    static inline uint64_t timemory_start_mallocp() { return 0; }
    static inline uint64_t timemory_stop_mallocp(uint64_t) { return 0; }
    static inline void     timemory_register_mallocp() {}
    static inline void     timemory_deregister_mallocp() {}

#else

// clang-format off
    extern void timemory_mallocp_library_ctor();
    extern uint64_t timemory_start_mallocp();
    extern uint64_t timemory_stop_mallocp(uint64_t);
    extern void timemory_register_mallocp();
    extern void timemory_deregister_mallocp();
// clang-format on

#endif

#if defined(__cplusplus)
}
#endif
