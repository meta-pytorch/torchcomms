//
// MIT License
// Copyright (c) 2019 Jonathan R. Madsen
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ----------------------------------------------------------------------
// TiMemory
//
// Provides empty macros when Tasking is compiled with TiMemory disabled
// ----------------------------------------------------------------------

#pragma once

#include "Globals.hh"

#ifdef PTL_USE_TIMEMORY
#    include <timemory/timemory.hpp>
#endif

namespace PTL
{
//--------------------------------------------------------------------------------------//
#ifdef PTL_USE_TIMEMORY

typedef tim::auto_timer AutoTimer;

inline void
InitializeTiMemory()
{
    tim::manager* instance = tim::manager::instance();
    instance->enable(true);
}

#else

#    define TIMEMORY_AUTO_TIMER(str)
#    define TIMEMORY_AUTO_TIMER_OBJ(str)                                                 \
        {}

#    define TIMEMORY_BASIC_AUTO_TIMER(str)
#    define TIMEMORY_BASIC_AUTO_TIMER_OBJ(str)                                           \
        {}

#    define TIMEMORY_DEBUG_BASIC_AUTO_TIMER(str)
#    define TIMEMORY_DEBUG_AUTO_TIMER(str)

inline void
InitializeTiMemory()
{}

#endif
//--------------------------------------------------------------------------------------//

}  // namespace PTL
