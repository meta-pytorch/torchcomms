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
//

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/types.hpp"
#include "timemory/variadic/auto_bundle.hpp"
#include "timemory/variadic/component_bundle.hpp"
#include "timemory/variadic/macros.hpp"
#include "timemory/variadic/types.hpp"

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
using minimal_auto_tuple_t = auto_bundle<TIMEMORY_API, TIMEMORY_MINIMAL_TUPLE_TYPES>;
using minimal_auto_list_t  = auto_bundle<TIMEMORY_API, TIMEMORY_MINIMAL_LIST_TYPES>;
using minimal_auto_timer_t =
    auto_bundle<TIMEMORY_API, TIMEMORY_MINIMAL_TUPLE_TYPES, TIMEMORY_MINIMAL_LIST_TYPES>;
//
//--------------------------------------------------------------------------------------//
//
using full_auto_tuple_t = auto_bundle<TIMEMORY_API, TIMEMORY_FULL_TUPLE_TYPES>;
using full_auto_list_t  = auto_bundle<TIMEMORY_API, TIMEMORY_FULL_LIST_TYPES>;
using full_auto_timer_t =
    auto_bundle<TIMEMORY_API, TIMEMORY_FULL_TUPLE_TYPES, TIMEMORY_FULL_LIST_TYPES>;
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_FULL_AUTO_TIMER)
//
using auto_timer_tuple_t = full_auto_tuple_t;
using auto_timer_list_t  = full_auto_list_t;
using auto_timer         = full_auto_timer_t;
//
#else
//
using auto_timer_tuple_t = minimal_auto_tuple_t;
using auto_timer_list_t  = minimal_auto_list_t;
using auto_timer         = minimal_auto_timer_t;
//
#endif
//
//--------------------------------------------------------------------------------------//
//
using auto_timer_t = auto_timer;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

//
//--------------------------------------------------------------------------------------//
//

//
//--------------------------------------------------------------------------------------//
//
