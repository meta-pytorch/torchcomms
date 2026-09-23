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

#include "timemory/utility/macros.hpp"

#include <string>

namespace tim
{
namespace utility
{
class path : public std::string
{
public:
    using size_type = std::string::size_type;

public:
    TIMEMORY_UTILITY_INLINE path(const std::string& _path);
    TIMEMORY_UTILITY_INLINE path(char* _path);
    TIMEMORY_UTILITY_INLINE path(const path& rhs);
    TIMEMORY_UTILITY_INLINE path(const char* _path);

    TIMEMORY_UTILITY_INLINE path& operator=(const std::string& rhs);
    TIMEMORY_UTILITY_INLINE path& operator=(const path& rhs);
    TIMEMORY_UTILITY_INLINE path& insert(size_type __pos, const std::string& __s);
    TIMEMORY_UTILITY_INLINE path& insert(size_type __pos, const path& __s);

    // OS-dependent representation
    static TIMEMORY_UTILITY_INLINE std::string osrepr(std::string _path);
    static TIMEMORY_UTILITY_INLINE std::string os();
    static TIMEMORY_UTILITY_INLINE std::string inverse();
    static TIMEMORY_UTILITY_INLINE std::string canonical(std::string _path);
};
}  // namespace utility
}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/path.cpp"
#endif
