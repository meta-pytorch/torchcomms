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
 * \file timemory/components/user_bundle/extern.hpp
 * \brief Include the extern declarations for user_bundle components
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/extern/common.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/opaque/declaration.hpp"
#include "timemory/components/types.hpp"
#include "timemory/components/user_bundle/components.hpp"
#include "timemory/components/user_bundle/types.hpp"
#include "timemory/environment/types.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/manager/manager.hpp"
#include "timemory/operations/definition.hpp"
#include "timemory/plotting/types.hpp"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/storage/declaration.hpp"

#if defined(TIMEMORY_USER_BUNDLE_SOURCE)
#    include "timemory/components/extern.hpp"
#    include "timemory/storage/definition.hpp"
#endif

#if defined(TIMEMORY_USER_BUNDLE_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS)
#        define TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(NAME, VAL)                        \
            TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(NAME, VAL)
#    endif
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_STORAGE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(...)                                 \
            TIMEMORY_INSTANTIATE_EXTERN_STORAGE(__VA_ARGS__)
#    endif
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(...) template __VA_ARGS__;
#    endif
//
//--------------------------------------------------------------------------------------//
//
#elif defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS)
#        define TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(NAME, VAL)                        \
            TIMEMORY_DECLARE_EXTERN_OPERATIONS(NAME, VAL)
#    endif
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_STORAGE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(...)                                 \
            TIMEMORY_DECLARE_EXTERN_STORAGE(__VA_ARGS__)
#    endif
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(...) extern template __VA_ARGS__;
#    endif
//
//--------------------------------------------------------------------------------------//
//
#else
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS)
#        define TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(...)
#    endif
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_STORAGE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(...)
#    endif
//
#    if !defined(TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE)
#        define TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(...)
#    endif
#endif

#if !defined(TIMEMORY_EXTERN_USER_BUNDLE)
#    define TIMEMORY_EXTERN_USER_BUNDLE(NAME)                                            \
        TIMEMORY_EXTERN_USER_BUNDLE_TEMPLATE(                                            \
            struct tim::component::base<TIMEMORY_ESC(tim::component::NAME), void>)       \
        TIMEMORY_EXTERN_USER_BUNDLE_OPERATIONS(TIMEMORY_ESC(component::NAME), false)     \
        TIMEMORY_EXTERN_USER_BUNDLE_STORAGE(TIMEMORY_ESC(component::NAME))
#endif

TIMEMORY_EXTERN_USER_BUNDLE(user_global_bundle)
TIMEMORY_EXTERN_USER_BUNDLE(user_ompt_bundle)
TIMEMORY_EXTERN_USER_BUNDLE(user_mpip_bundle)
TIMEMORY_EXTERN_USER_BUNDLE(user_ncclp_bundle)
TIMEMORY_EXTERN_USER_BUNDLE(user_trace_bundle)
TIMEMORY_EXTERN_USER_BUNDLE(user_profiler_bundle)
TIMEMORY_EXTERN_USER_BUNDLE(user_kokkosp_bundle)

#if defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
namespace tim
{
namespace component
{
extern template struct user_bundle<global_bundle_idx, project::timemory>;
extern template struct user_bundle<ompt_bundle_idx, project::timemory>;
extern template struct user_bundle<mpip_bundle_idx, project::timemory>;
extern template struct user_bundle<ncclp_bundle_idx, project::timemory>;
extern template struct user_bundle<trace_bundle_idx, project::timemory>;
extern template struct user_bundle<profiler_bundle_idx, project::timemory>;
extern template struct user_bundle<kokkosp_bundle_idx, project::kokkosp>;
}  // namespace component
}  // namespace tim
#endif
