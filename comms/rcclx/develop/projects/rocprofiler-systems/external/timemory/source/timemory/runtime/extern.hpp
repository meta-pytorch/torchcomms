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
 * \file timemory/runtime/extern.hpp
 * \brief Include the extern declarations for runtime
 */

#pragma once

#if defined(TIMEMORY_USER_BUNDLE_SOURCE)
#    error "Error! Extern templates cannot be defined as part of user_bundle source"
#endif

#include "timemory/backends/extern.hpp"
#include "timemory/backends/types/mpi/extern.hpp"
#include "timemory/components/extern.hpp"
#include "timemory/components/user_bundle/extern.hpp"
#include "timemory/hash/extern.hpp"
#include "timemory/manager/extern.hpp"
#include "timemory/operations/extern.hpp"
#include "timemory/plotting/extern.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/insert.hpp"
#include "timemory/runtime/invoker.hpp"
#include "timemory/runtime/macros.hpp"
#include "timemory/runtime/properties.hpp"
#include "timemory/runtime/types.hpp"
#include "timemory/storage/extern.hpp"
#include "timemory/types.hpp"

TIMEMORY_RUNTIME_USER_BUNDLE_EXTERN_TEMPLATE(component::user_global_bundle, scope::config)
TIMEMORY_RUNTIME_USER_BUNDLE_EXTERN_TEMPLATE(component::user_ompt_bundle, scope::config)
TIMEMORY_RUNTIME_USER_BUNDLE_EXTERN_TEMPLATE(component::user_mpip_bundle, scope::config)
TIMEMORY_RUNTIME_USER_BUNDLE_EXTERN_TEMPLATE(component::user_ncclp_bundle, scope::config)
//
#if defined(TIMEMORY_USE_CONTAINERS_EXTERN) && TIMEMORY_USE_CONTAINERS > 0
#    include "timemory/containers/auto_timer.hpp"
#    include "timemory/containers/auto_user_bundle.hpp"

TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(full_auto_timer_t)
TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(minimal_auto_timer_t)
TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(auto_user_bundle_t)
//
TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(complete_component_list_t)
TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(available_component_list_t)
TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(complete_auto_list_t)
TIMEMORY_RUNTIME_INITIALIZE_EXTERN_TEMPLATE(available_auto_list_t)
#endif
