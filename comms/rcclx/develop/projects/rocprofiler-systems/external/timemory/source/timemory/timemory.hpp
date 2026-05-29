//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file timemory/timemory.hpp
 * \headerfile timemory/timemory.hpp "timemory/timemory.hpp"
 * All-inclusive timemory header
 *
 */

#pragma once

#if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED) ||                           \
    (defined(TIMEMORY_ENABLED) && TIMEMORY_ENABLED == 0)

#    include <ostream>
#    include <string>

namespace tim
{
template <typename... ArgsT>
void
timemory_init(ArgsT...)
{}
inline void
timemory_finalize()
{}
inline void
print_env()
{}

/// this provides "functionality" for *_HANDLE macros
/// and can be omitted if these macros are not utilized
struct dummy
{
    template <typename... Types, typename... ArgsT>
    static void configure(ArgsT&&...)
    {}

    template <typename... ArgsT>
    dummy(ArgsT&&...)
    {}
    ~dummy()            = default;
    dummy(const dummy&) = default;
    dummy(dummy&&)      = default;
    dummy& operator=(const dummy&) = default;
    dummy& operator=(dummy&&) = default;

    void start() {}
    void stop() {}
    void report_at_exit(bool) {}
    template <typename... ArgsT>
    void mark_begin(ArgsT&&...)
    {}
    template <typename... ArgsT>
    void mark_end(ArgsT&&...)
    {}
    friend std::ostream& operator<<(std::ostream& os, const dummy&) { return os; }
};
}  // namespace tim

#    if !defined(TIMEMORY_MACROS)
#        define TIMEMORY_MACROS
#    endif

// startup/shutdown/configure
#    if !defined(TIMEMORY_INIT)
#        define TIMEMORY_INIT(...)
#    endif

#    if !defined(TIMEMORY_FINALIZE)
#        define TIMEMORY_FINALIZE()
#    endif

#    if !defined(TIMEMORY_CONFIGURE)
#        define TIMEMORY_CONFIGURE(...)
#    endif

// label creation
#    if !defined(TIMEMORY_BASIC_LABEL)
#        define TIMEMORY_BASIC_LABEL(...) std::string("")
#    endif

#    if !defined(TIMEMORY_LABEL)
#        define TIMEMORY_LABEL(...) std::string("")
#    endif

#    if !defined(TIMEMORY_JOIN)
#        define TIMEMORY_JOIN(...) std::string("")
#    endif

// define an object
#    if !defined(TIMEMORY_BLANK_MARKER)
#        define TIMEMORY_BLANK_MARKER(...)
#    endif

#    if !defined(TIMEMORY_BASIC_MARKER)
#        define TIMEMORY_BASIC_MARKER(...)
#    endif

#    if !defined(TIMEMORY_MARKER)
#        define TIMEMORY_MARKER(...)
#    endif

// define an unique pointer object
#    if !defined(TIMEMORY_BLANK_POINTER)
#        define TIMEMORY_BLANK_POINTER(...)
#    endif

#    if !defined(TIMEMORY_BASIC_POINTER)
#        define TIMEMORY_BASIC_POINTER(...)
#    endif

#    if !defined(TIMEMORY_POINTER)
#        define TIMEMORY_POINTER(...)
#    endif

// define an object with a caliper reference
#    if !defined(TIMEMORY_BLANK_CALIPER)
#        define TIMEMORY_BLANK_CALIPER(...)
#    endif

#    if !defined(TIMEMORY_BASIC_CALIPER)
#        define TIMEMORY_BASIC_CALIPER(...)
#    endif

#    if !defined(TIMEMORY_CALIPER)
#        define TIMEMORY_CALIPER(...)
#    endif

// define a static object with a caliper reference
#    if !defined(TIMEMORY_STATIC_BLANK_CALIPER)
#        define TIMEMORY_STATIC_BLANK_CALIPER(...)
#    endif

#    if !defined(TIMEMORY_STATIC_BASIC_CALIPER)
#        define TIMEMORY_STATIC_BASIC_CALIPER(...)
#    endif

#    if !defined(TIMEMORY_STATIC_CALIPER)
#        define TIMEMORY_STATIC_CALIPER(...)
#    endif

// invoke member function on caliper reference or type within reference
#    if !defined(TIMEMORY_CALIPER_APPLY)
#        define TIMEMORY_CALIPER_APPLY(...)
#    endif

#    if !defined(TIMEMORY_CALIPER_TYPE_APPLY)
#        define TIMEMORY_CALIPER_TYPE_APPLY(...)
#    endif

#    if !defined(TIMEMORY_CALIPER_APPLY0)
#        define TIMEMORY_CALIPER_APPLY0(...)
#    endif

#    if !defined(TIMEMORY_CALIPER_TYPE_APPLY0)
#        define TIMEMORY_CALIPER_TYPE_APPLY0(...)
#    endif

#    if !defined(TIMEMORY_CALIPER_LAMBDA)
#        define TIMEMORY_CALIPER_LAMBDA(...)
#    endif

#    if !defined(TIMEMORY_CALIPER_TYPE_LAMBDA)
#        define TIMEMORY_CALIPER_TYPE_LAMBDA(...)
#    endif

// get an object
#    if !defined(TIMEMORY_BLANK_HANDLE)
#        define TIMEMORY_BLANK_HANDLE(...) tim::dummy()
#    endif

#    if !defined(TIMEMORY_BASIC_HANDLE)
#        define TIMEMORY_BASIC_HANDLE(...) tim::dummy()
#    endif

#    if !defined(TIMEMORY_HANDLE)
#        define TIMEMORY_HANDLE(...) tim::dummy()
#    endif

// get a pointer to an object
#    if !defined(TIMEMORY_BLANK_RAW_POINTER)
#        define TIMEMORY_BLANK_RAW_POINTER(...) nullptr
#    endif

#    if !defined(TIMEMORY_BASIC_RAW_POINTER)
#        define TIMEMORY_BASIC_RAW_POINTER(...) nullptr
#    endif

#    if !defined(TIMEMORY_RAW_POINTER)
#        define TIMEMORY_RAW_POINTER(...) nullptr
#    endif

// debug only
#    if !defined(TIMEMORY_DEBUG_BLANK_MARKER)
#        define TIMEMORY_DEBUG_BLANK_MARKER(...)
#    endif

#    if !defined(TIMEMORY_DEBUG_BASIC_MARKER)
#        define TIMEMORY_DEBUG_BASIC_MARKER(...)
#    endif

#    if !defined(TIMEMORY_DEBUG_MARKER)
#        define TIMEMORY_DEBUG_MARKER(...)
#    endif

// auto-timers
#    if !defined(TIMEMORY_BLANK_AUTO_TIMER)
#        define TIMEMORY_BLANK_AUTO_TIMER(...)
#    endif

#    if !defined(TIMEMORY_BASIC_AUTO_TIMER)
#        define TIMEMORY_BASIC_AUTO_TIMER(...)
#    endif

#    if !defined(TIMEMORY_AUTO_TIMER)
#        define TIMEMORY_AUTO_TIMER(...)
#    endif

#    if !defined(TIMEMORY_BLANK_AUTO_TIMER_HANDLE)
#        define TIMEMORY_BLANK_AUTO_TIMER_HANDLE(...)
#    endif

#    if !defined(TIMEMORY_BASIC_AUTO_TIMER_HANDLE)
#        define TIMEMORY_BASIC_AUTO_TIMER_HANDLE(...)
#    endif

#    if !defined(TIMEMORY_AUTO_TIMER_HANDLE)
#        define TIMEMORY_AUTO_TIMER_HANDLE(...)
#    endif

#    if !defined(TIMEMORY_DEBUG_BASIC_AUTO_TIMER)
#        define TIMEMORY_DEBUG_BASIC_AUTO_TIMER(...)
#    endif

#    if !defined(TIMEMORY_DEBUG_AUTO_TIMER)
#        define TIMEMORY_DEBUG_AUTO_TIMER(...)
#    endif

// auto-bundle (user-bundles)
#    if !defined(TIMEMORY_BLANK_AUTO_BUNDLE)
#        define TIMEMORY_BLANK_AUTO_BUNDLE(...)
#    endif

#    if !defined(TIMEMORY_BASIC_AUTO_BUNDLE)
#        define TIMEMORY_BASIC_AUTO_BUNDLE(...)
#    endif

#    if !defined(TIMEMORY_AUTO_BUNDLE)
#        define TIMEMORY_AUTO_BUNDLE(...)
#    endif

#    if !defined(TIMEMORY_BLANK_AUTO_BUNDLE_HANDLE)
#        define TIMEMORY_BLANK_AUTO_BUNDLE_HANDLE(...)
#    endif

#    if !defined(TIMEMORY_BASIC_AUTO_BUNDLE_HANDLE)
#        define TIMEMORY_BASIC_AUTO_BUNDLE_HANDLE(...)
#    endif

#    if !defined(TIMEMORY_AUTO_BUNDLE_HANDLE)
#        define TIMEMORY_AUTO_BUNDLE_HANDLE(...)
#    endif

#    if !defined(TIMEMORY_DEBUG_BASIC_AUTO_BUNDLE)
#        define TIMEMORY_DEBUG_BASIC_AUTO_BUNDLE(...)
#    endif

#    if !defined(TIMEMORY_DEBUG_AUTO_BUNDLE)
#        define TIMEMORY_DEBUG_AUTO_BUNDLE(...)
#    endif

#    if !defined(TIMEMORY_TOOLSET_ALIAS)
#        define TIMEMORY_TOOLSET_ALIAS(...)
#    endif

#    if !defined(TIMEMORY_DECLARE_COMPONENT)
#        define TIMEMORY_DECLARE_COMPONENT(...)
#    endif

#    if !defined(TIMEMORY_STATISTICS_TYPE)
#        define TIMEMORY_STATISTICS_TYPE(...)
#    endif

#    if !defined(TIMEMORY_TEMPLATE_STATISTICS_TYPE)
#        define TIMEMORY_TEMPLATE_STATISTICS_TYPE(...)
#    endif

#    if !defined(TIMEMORY_VARIADIC_STATISTICS_TYPE)
#        define TIMEMORY_VARIADIC_STATISTICS_TYPE(...)
#    endif

#    if !defined(TIMEMORY_DEFINE_CONCRETE_TRAIT)
#        define TIMEMORY_DEFINE_CONCRETE_TRAIT(...)
#    endif

#    if !defined(TIMEMORY_DEFINE_TEMPLATE_TRAIT)
#        define TIMEMORY_DEFINE_TEMPLATE_TRAIT(...)
#    endif

#    if !defined(TIMEMORY_DEFINE_VARIADIC_TRAIT)
#        define TIMEMORY_DEFINE_VARIADIC_TRAIT(...)
#    endif

#    include "timemory/general/serialization.hpp"

#else

#    if !defined(TIMEMORY_MASTER_HEADER)
#        define TIMEMORY_MASTER_HEADER
#    endif

#    if !defined(TIMEMORY_ENABLED)
#        define TIMEMORY_ENABLED 1
#    endif

//
//   versioning header
//
#    include "timemory/version.h"
//
#    include "timemory/extern.hpp"
//
#    include "timemory/api.hpp"
#    include "timemory/config.hpp"
#    include "timemory/enum.h"
#    include "timemory/general.hpp"
#    include "timemory/plotting.hpp"
#    include "timemory/types.hpp"
#    include "timemory/units.hpp"
//
#    include "timemory/components.hpp"
#    include "timemory/containers.hpp"
#    include "timemory/ert.hpp"
#    include "timemory/settings.hpp"
#    include "timemory/utility.hpp"
#    include "timemory/variadic.hpp"
//
#    include "timemory/definition.hpp"
#    include "timemory/runtime.hpp"
//
//======================================================================================//
//

#    include "timemory/variadic/definition.hpp"
#endif  // ! defined(DISABLE_TIMEMORY)
