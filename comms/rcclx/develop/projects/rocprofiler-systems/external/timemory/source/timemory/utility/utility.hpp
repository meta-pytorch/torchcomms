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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file utility/utility.hpp
 * \headerfile utility/utility.hpp "timemory/utility/utility.hpp"
 * General utility functions
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

//--------------------------------------------------------------------------------------//

namespace tim
{
// alias here for common string type
// there is also a string_view_t alias in macros/language.hpp which is std::string in
// c++14 and std::string_view in c++17 and newer
using string_t = std::string;

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline bool
isfinite(const Tp& arg)
{
#if defined(TIMEMORY_WINDOWS)
    // Windows seems to be missing std::isfinite
    return (arg == arg && arg != std::numeric_limits<Tp>::infinity() &&
            arg != -std::numeric_limits<Tp>::infinity())
               ? true
               : false;
#else
    return std::isfinite(arg);
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE std::string
                        dirname(std::string _fname);

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE int
makedir(std::string _dir, int umask = TIMEMORY_DEFAULT_UMASK);

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE bool
get_bool(const std::string& strbool, bool _default = false) noexcept;

//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_UTILITY_INLINE std::vector<std::string>
                        read_command_line(pid_t _pid);

}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/utility.cpp"
#endif
