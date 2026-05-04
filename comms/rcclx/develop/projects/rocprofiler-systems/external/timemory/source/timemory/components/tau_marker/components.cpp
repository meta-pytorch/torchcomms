//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#ifndef TIMEMORY_COMPONENTS_TAU_MARKER_COMPONENTS_CPP_
#define TIMEMORY_COMPONENTS_TAU_MARKER_COMPONENTS_CPP_

#include "timemory/components/tau_marker/macros.hpp"

#if !defined(TIMEMORY_TAU_MARKER_COMPONENT_HEADER_MODE) ||                               \
    (defined(TIMEMORY_TAU_MARKER_COMPONENT_HEADER_MODE) &&                               \
     TIMEMORY_TAU_MARKER_COMPONENT_HEADER_MODE < 1)
#    include "timemory/components/tau_marker/components.hpp"
#endif

#include "timemory/components/base.hpp"
#include "timemory/components/tau_marker/backends.hpp"
#include "timemory/components/tau_marker/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
TIMEMORY_TAU_MARKER_INLINE
void
tau_marker::global_init()
{
    TIMEMORY_TAU_SET_NODE(dmp::rank());
}

TIMEMORY_TAU_MARKER_INLINE
void
tau_marker::thread_init()
{
    TIMEMORY_TAU_REGISTER_THREAD();
}

TIMEMORY_TAU_MARKER_INLINE
void
tau_marker::start(const char* _prefix)
{
    TIMEMORY_TAU_START(_prefix);
}

TIMEMORY_TAU_MARKER_INLINE
void
tau_marker::stop(const char* _prefix)
{
    TIMEMORY_TAU_STOP(_prefix);
}
}  // namespace component
}  // namespace tim

#endif
