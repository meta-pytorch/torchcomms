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

#include "timemory/components.hpp"
#include "timemory/components/opaque/definition.hpp"
#include "timemory/ert/definition.hpp"
#include "timemory/operations/definition.hpp"
#include "timemory/storage/definition.hpp"
#include "timemory/variadic/definition.hpp"

#if !defined(TIMEMORY_USE_CONTAINERS) ||                                                 \
    (defined(TIMEMORY_USE_CONTAINERS) && TIMEMORY_USE_CONTAINERS > 0)
#    include "timemory/containers/definition.hpp"
#endif

#if !defined(TIMEMORY_USE_EXTERN)
//
#    if !defined(TIMEMORY_USE_MANAGER_EXTERN)
#        include "timemory/manager.hpp"
#    endif

#    if !defined(TIMEMORY_USE_CONFIG_EXTERN)
#        include "timemory/config.hpp"
#    endif

#    if !defined(TIMEMORY_USE_CORE_EXTERN)
#        include "timemory/environment.hpp"
#        include "timemory/hash.hpp"
#        include "timemory/plotting.hpp"
#        include "timemory/settings.hpp"
#    endif
//
#endif
