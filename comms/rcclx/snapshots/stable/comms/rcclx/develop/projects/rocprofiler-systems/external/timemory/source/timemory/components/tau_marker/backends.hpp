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

/**
 * \file timemory/components/tau_marker/backends.hpp
 * \brief Implementation of the tau_marker functions/utilities
 */

#pragma once

#if defined(TIMEMORY_USE_TAU)
#    if !defined(TAU_ENABLED)
#        define TAU_ENABLED
#    endif
#    if !defined(TAU_DOT_H_LESS_HEADERS)
#        define TAU_DOT_H_LESS_HEADERS
#    endif
#    include "TAU.h"
#    if !defined(TIMEMORY_TAU_INIT)
#        define TIMEMORY_TAU_INIT(...) Tau_init(__VA_ARGS__)
#    endif
#    if !defined(TIMEMORY_TAU_SET_NODE)
#        define TIMEMORY_TAU_SET_NODE(...) Tau_set_node(__VA_ARGS__)
#    endif
#    if !defined(TIMEMORY_TAU_START)
#        define TIMEMORY_TAU_START(...) Tau_start(__VA_ARGS__)
#    endif
#    if !defined(TIMEMORY_TAU_STOP)
#        define TIMEMORY_TAU_STOP(...) Tau_stop(__VA_ARGS__)
#    endif
#    if !defined(TIMEMORY_TAU_REGISTER_THREAD)
#        define TIMEMORY_TAU_REGISTER_THREAD TAU_REGISTER_THREAD
#    endif
#else
#    if !defined(TIMEMORY_TAU_INIT)
#        define TIMEMORY_TAU_INIT(...) ::tim::consume_parameters(__VA_ARGS__)
#    endif
#    if !defined(TIMEMORY_TAU_SET_NODE)
#        define TIMEMORY_TAU_SET_NODE(...) ::tim::consume_parameters(__VA_ARGS__)
#    endif
#    if !defined(TIMEMORY_TAU_START)
#        define TIMEMORY_TAU_START(...) ::tim::consume_parameters(__VA_ARGS__)
#    endif
#    if !defined(TIMEMORY_TAU_STOP)
#        define TIMEMORY_TAU_STOP(...) ::tim::consume_parameters(__VA_ARGS__)
#    endif
#    if !defined(TIMEMORY_TAU_REGISTER_THREAD)
#        define TIMEMORY_TAU_REGISTER_THREAD ::tim::consume_parameters
#    endif
#endif
