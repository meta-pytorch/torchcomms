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

/** \file timemory/ert/types.hpp
 * \headerfile timemory/ert/types.hpp "timemory/ert/types.hpp"
 * Provides declaration of types for ERT
 *
 */

#pragma once

#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_ERT_EXTERN)
#    define TIMEMORY_USE_ERT_EXTERN 1
#endif

#if defined(TIMEMORY_ERT_SOURCE)
#    if !defined(TIMEMORY_ERT_EXTERN_TEMPLATE)
#        define TIMEMORY_ERT_EXTERN_TEMPLATE(...) template __VA_ARGS__;
#    endif
#elif defined(TIMEMORY_USE_ERT_EXTERN)
#    if !defined(TIMEMORY_ERT_EXTERN_TEMPLATE)
#        define TIMEMORY_ERT_EXTERN_TEMPLATE(...) extern template __VA_ARGS__;
#    endif
#else
#    if !defined(TIMEMORY_ERT_EXTERN_TEMPLATE)
#        define TIMEMORY_ERT_EXTERN_TEMPLATE(...)
#    endif
#endif

#include "timemory/backends/cpu.hpp"
#include "timemory/macros/attributes.hpp"

// clang-format off
namespace tim { namespace component { struct ert_timer; } }
// clang-format on

namespace tim
{
namespace ert
{
class thread_barrier;
struct exec_params;

template <typename Tp = component::ert_timer>
class exec_data;

template <typename DeviceT, typename Tp, typename CounterT = component::ert_timer>
class counter;

template <typename DeviceT, typename Tp, typename CounterT>
struct TIMEMORY_VISIBLE configuration;

template <typename DeviceT, typename Tp, typename CounterT>
struct executor;

template <typename ExecutorT>
struct callback;

namespace cache_size = cpu::cache_size;
}  // namespace ert
}  // namespace tim
