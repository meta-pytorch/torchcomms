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
//

#pragma once

#include "timemory/log/color.hpp"
#include "timemory/log/macros.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/macros/os.hpp"

#include <cstdint>
#include <cstdio>
#include <iosfwd>
#include <string>
#include <utility>

#if defined(TIMEMORY_CORE_SOURCE)
#    define TIMEMORY_UTILITY_SOURCE 1
#    define TIMEMORY_UTILITY_INLINE
#    define TIMEMORY_UTILITY_LINKAGE(...) __VA_ARGS__
#elif defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_UTILITY_EXTERN 1
#    define TIMEMORY_UTILITY_INLINE
#    define TIMEMORY_UTILITY_LINKAGE(...) __VA_ARGS__
#else
#    define TIMEMORY_UTILITY_INLINE inline
#    define TIMEMORY_UTILITY_HEADER_MODE 1
#    define TIMEMORY_UTILITY_LINKAGE(...) inline __VA_ARGS__
#endif
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_UTILITY_EXTERN)
#    define TIMEMORY_USE_UTILITY_EXTERN 1
#endif

#if !defined(TIMEMORY_DEFAULT_UMASK)
#    define TIMEMORY_DEFAULT_UMASK 0777
#endif

#if defined(TIMEMORY_WINDOWS)
namespace tim
{
using pid_t = DWORD;
}
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DECLARE_EXTERN_TEMPLATE)
#    define TIMEMORY_DECLARE_EXTERN_TEMPLATE(...) extern template __VA_ARGS__;
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_INSTANTIATE_EXTERN_TEMPLATE)
#    define TIMEMORY_INSTANTIATE_EXTERN_TEMPLATE(...) template __VA_ARGS__;
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_ESC)
#    define TIMEMORY_ESC(...) __VA_ARGS__
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DELETED_OBJECT)
#    define TIMEMORY_DELETED_OBJECT(NAME)                                                \
        NAME()            = delete;                                                      \
        NAME(const NAME&) = delete;                                                      \
        NAME(NAME&&)      = delete;                                                      \
        NAME& operator=(const NAME&) = delete;                                           \
        NAME& operator=(NAME&&) = delete;
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DELETE_COPY_MOVE_OBJECT)
#    define TIMEMORY_DELETE_COPY_MOVE_OBJECT(NAME)                                       \
        NAME(const NAME&) = delete;                                                      \
        NAME(NAME&&)      = delete;                                                      \
        NAME& operator=(const NAME&) = delete;                                           \
        NAME& operator=(NAME&&) = delete;
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DEFAULT_MOVE_ONLY_OBJECT)
#    define TIMEMORY_DEFAULT_MOVE_ONLY_OBJECT(NAME)                                      \
        NAME(const NAME&)     = delete;                                                  \
        NAME(NAME&&) noexcept = default;                                                 \
        NAME& operator=(const NAME&) = delete;                                           \
        NAME& operator=(NAME&&) noexcept = default;
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DEFAULT_OBJECT)
#    define TIMEMORY_DEFAULT_OBJECT(NAME)                                                \
        TIMEMORY_HOST_DEVICE_FUNCTION NAME() = default;                                  \
        NAME(const NAME&)                    = default;                                  \
        NAME(NAME&&) noexcept                = default;                                  \
        NAME& operator=(const NAME&) = default;                                          \
        NAME& operator=(NAME&&) noexcept = default;
#endif

//======================================================================================//
//
//      Quick way to create a globally accessible setting
//
//======================================================================================//

#if !defined(TIMEMORY_CREATE_STATIC_VARIABLE_ACCESSOR)
#    define TIMEMORY_CREATE_STATIC_VARIABLE_ACCESSOR(TYPE, FUNC_NAME, VARIABLE)          \
        static TYPE& FUNC_NAME()                                                         \
        {                                                                                \
            static TYPE _instance = Type::VARIABLE;                                      \
            return _instance;                                                            \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_CREATE_STATIC_FUNCTION_ACCESSOR)
#    define TIMEMORY_CREATE_STATIC_FUNCTION_ACCESSOR(TYPE, FUNC_NAME, VARIABLE)          \
        static TYPE& FUNC_NAME()                                                         \
        {                                                                                \
            static TYPE _instance = Type::VARIABLE();                                    \
            return _instance;                                                            \
        }
#endif
