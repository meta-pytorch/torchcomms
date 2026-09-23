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

/**
 * \file timemory/hash/macros.hpp
 * \brief Include the macros for hash
 */

#pragma once

#include "timemory/dll.hpp"

#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_RUNTIME_EXTERN)
#    define TIMEMORY_USE_RUNTIME_EXTERN
#endif
//
#if !defined(TIMEMORY_RUNTIME_USER_BUNDLE_EXTERN_TEMPLATE)
#    if defined(TIMEMORY_RUNTIME_SOURCE)
#        define TIMEMORY_RUNTIME_USER_BUNDLE_EXTERN_TEMPLATE(TYPE, ...)                  \
            namespace tim                                                                \
            {                                                                            \
            namespace runtime                                                            \
            {                                                                            \
            template void insert(TYPE&, int);                                            \
            template void configure<TYPE>(int);                                          \
            template void configure(TYPE&, int);                                         \
            template void insert(TYPE&, int, __VA_ARGS__);                               \
            template void configure<TYPE>(int, __VA_ARGS__);                             \
            template void configure(TYPE&, int, __VA_ARGS__);                            \
            }                                                                            \
            }
#    elif !defined(TIMEMORY_USE_RUNTIME_EXTERN)
#        define TIMEMORY_RUNTIME_USER_BUNDLE_EXTERN_TEMPLATE(...)
#    else
#        define TIMEMORY_RUNTIME_USER_BUNDLE_EXTERN_TEMPLATE(TYPE, ...)                  \
            namespace tim                                                                \
            {                                                                            \
            namespace runtime                                                            \
            {                                                                            \
            extern template void insert(TYPE&, int);                                     \
            extern template void configure<TYPE>(int);                                   \
            extern template void configure(TYPE&, int);                                  \
            extern template void insert(TYPE&, int, __VA_ARGS__);                        \
            extern template void configure<TYPE>(int, __VA_ARGS__);                      \
            extern template void configure(TYPE&, int, __VA_ARGS__);                     \
            }                                                                            \
            }
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE)
#    if defined(TIMEMORY_RUNTIME_SOURCE)
#        define TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(...)                         \
            namespace tim                                                                \
            {                                                                            \
            namespace runtime                                                            \
            {                                                                            \
            template void initialize(__VA_ARGS__&, int);                                 \
            }                                                                            \
            }
#    elif !defined(TIMEMORY_USE_RUNTIME_EXTERN)
#        define TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(...)
#    else
#        define TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(...)                         \
            namespace tim                                                                \
            {                                                                            \
            namespace runtime                                                            \
            {                                                                            \
            extern template void initialize(__VA_ARGS__&, int);                          \
            }                                                                            \
            }
#    endif
#endif
