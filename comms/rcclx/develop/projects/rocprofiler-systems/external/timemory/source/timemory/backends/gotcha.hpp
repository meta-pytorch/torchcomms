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

/** \file backends/gotcha.hpp
 * \headerfile backends/gotcha.hpp "timemory/backends/gotcha.hpp"
 * Defines GOTCHA backend
 *
 */

#pragma once

#include "timemory/backends/defines.hpp"
#include "timemory/defines.h"
#include "timemory/mpl/concepts.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

//======================================================================================//

#if defined(TIMEMORY_USE_GOTCHA)

#    include <gotcha/gotcha.h>
#    include <gotcha/gotcha_types.h>

#else

///
/// The representation of a Gotcha action as it passes through the pipeline
///
typedef struct __gotcha_binding_timemory
{
    // NOLINTNEXTLINE
    const char* name = nullptr;  //!< The name of the function being wrapped
    // NOLINTNEXTLINE
    void* wrapper_pointer = nullptr;  //!< A pointer to the wrapper function
    // NOLINTNEXTLINE
    void* function_handle = nullptr;  //!< A pointer to the function being wrapped

    __gotcha_binding_timemory(const char* _name, void* _wrap, void* _handle)
    : name(_name)
    , wrapper_pointer(_wrap)
    , function_handle(_handle)
    {}

    __gotcha_binding_timemory()                                 = default;
    ~__gotcha_binding_timemory()                                = default;
    __gotcha_binding_timemory(const __gotcha_binding_timemory&) = default;
    __gotcha_binding_timemory(__gotcha_binding_timemory&&)      = default;
    __gotcha_binding_timemory& operator=(const __gotcha_binding_timemory&) = default;
    __gotcha_binding_timemory& operator=(__gotcha_binding_timemory&&) = default;
} _gotcha_binding_timemory;

//======================================================================================//

///
/// The representation of an error (or success) of a Gotcha action
///
typedef enum __gotcha_error_timemory
{
    GOTCHA_SUCCESS = 0,         //!< The call succeeded
    GOTCHA_FUNCTION_NOT_FOUND,  //!< The call looked up a function which could not be
                                //!< found
    GOTCHA_INTERNAL,            //!< Internal gotcha error
    GOTCHA_INVALID_TOOL         //!< Invalid tool name
} _gotcha_error_timemory;

#endif

//======================================================================================//

namespace tim
{
namespace backend
{
namespace gotcha
{
using string_t = std::string;
using size_t   = std::size_t;

#if defined(TIMEMORY_USE_GOTCHA)

using binding_t = struct gotcha_binding_t;
using wrappee_t = gotcha_wrappee_handle_t;
using error_t   = gotcha_error_t;

#else

using binding_t = _gotcha_binding_timemory;
using wrappee_t = void*;
using error_t   = _gotcha_error_timemory;

#endif

//--------------------------------------------------------------------------------------//

void
set_verbose(int);

int
get_verbose();

bool
initialize();

const char*
get_error(error_t err);

error_t
set_priority(const char* _tool, int _priority = 0);

error_t
get_priority(const char* _tool, int& _priority);

error_t
wrap(binding_t& _bind, const char* _label);

inline auto
set_priority(const std::string& _tool, int _priority = 0)
{
    return set_priority(_tool.c_str(), _priority);
}

inline auto
get_priority(const std::string& _tool, int& _priority)
{
    return get_priority(_tool.c_str(), _priority);
}

inline auto
wrap(binding_t& _bind, std::string& _label)
{
    return wrap(_bind, _label.c_str());
}

template <size_t N, typename StringT>
std::array<error_t, N>
wrap(std::array<binding_t, N>& _arr, const std::array<bool, N>& _filled,
     std::array<StringT, N>& _labels)
{
    static_assert(concepts::is_string_type<StringT>::value,
                  "Error! labels must be string type");
    initialize();
    std::array<error_t, N> _ret;
    for(size_t i = 0; i < N; ++i)
    {
        if(_filled[i])
        {
            _ret[i] = wrap(_arr[i], _labels[i]);
        }
        else
        {
            _ret[i] = GOTCHA_SUCCESS;
        }
    }
    return _ret;
}
}  // namespace gotcha
}  // namespace backend
}  // namespace tim

#if defined(TIMEMORY_BACKENDS_HEADER_MODE) && TIMEMORY_BACKENDS_HEADER_MODE > 0
#    include "timemory/backends/gotcha.cpp"
#endif
