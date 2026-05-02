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

#if defined(TIMEMORY_USE_HIP)
#    include <roctracer/roctx.h>
#endif

#include "timemory/backends/threading.hpp"
#include "timemory/macros.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace tim
{
//======================================================================================//
//
//                                  ROCTX
//
//======================================================================================//

namespace roctx
{
//--------------------------------------------------------------------------------------//

template <typename... ArgsT>
void
consume_parameters(ArgsT&&...)
{}

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_HIP)
using range_id_t = roctx_range_id_t;
#else
using range_id_t = int;
#endif

//--------------------------------------------------------------------------------------//

inline void
name_thread(const char*)
{}

//--------------------------------------------------------------------------------------//

inline void
name_thread(const std::string&)
{}

//--------------------------------------------------------------------------------------//

inline void name_thread(int32_t) {}

//--------------------------------------------------------------------------------------//

inline void
range_push(const char* _msg)
{
#if defined(TIMEMORY_USE_HIP)
    roctxRangePush(_msg);
#else
    consume_parameters(_msg);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
range_push(const std::string& _msg)
{
#if defined(TIMEMORY_USE_HIP)
    roctxRangePush(_msg.c_str());
#else
    consume_parameters(_msg);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
range_pop()
{
#if defined(TIMEMORY_USE_HIP)
    roctxRangePop();
#endif
}

//--------------------------------------------------------------------------------------//

inline range_id_t
range_start(const char* _msg)
{
#if defined(TIMEMORY_USE_HIP)
    return roctxRangeStart(_msg);
#else
    consume_parameters(_msg);
    return 0;
#endif
}

//--------------------------------------------------------------------------------------//

inline range_id_t
range_start(const std::string& _msg)
{
    return range_start(_msg.c_str());
}

//--------------------------------------------------------------------------------------//

inline void
range_stop(const range_id_t& _id)
{
#if defined(TIMEMORY_USE_HIP)
    roctxRangeStop(_id);
#else
    consume_parameters(_id);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
mark(const char* _msg)
{
#if defined(TIMEMORY_USE_HIP)
    roctxMarkA(_msg);
#else
    consume_parameters(_msg);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
mark(const std::string& _msg)
{
#if defined(TIMEMORY_USE_HIP)
    roctxMarkA(_msg.c_str());
#else
    consume_parameters(_msg);
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace roctx
}  // namespace tim
