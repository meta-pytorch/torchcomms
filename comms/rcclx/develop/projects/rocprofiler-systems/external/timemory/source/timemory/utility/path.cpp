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

#ifndef TIMEMORY_UTILITY_PATH_CPP_
#define TIMEMORY_UTILITY_PATH_CPP_

#include "timemory/utility/macros.hpp"

#if !defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/path.hpp"
#endif

#include "timemory/utility/filepath.hpp"

namespace tim
{
//
namespace utility
{
//
path::path(const std::string& _path)
: std::string(osrepr(_path))
{}

path::path(char* _path)
: std::string(osrepr(std::string(_path)))
{}

path::path(const path& rhs)
: std::string(osrepr(rhs))
{}

path::path(const char* _path)
: std::string(osrepr(std::string(const_cast<char*>(_path))))
{}

path&
path::operator=(const std::string& rhs)
{
    std::string::operator=(osrepr(rhs));
    return *this;
}

path&
path::operator=(const path& rhs)
{
    if(this != &rhs)
        std::string::operator=(osrepr(rhs));
    return *this;
}

path&
path::insert(size_type __pos, const std::string& __s)
{
    std::string::operator=(osrepr(std::string::insert(__pos, __s)));
    return *this;
}

path&
path::insert(size_type __pos, const path& __s)
{
    std::string::operator=(osrepr(std::string::insert(__pos, __s)));
    return *this;
}

std::string
path::os()
{
#if defined(TIMEMORY_WINDOWS)
    return "\\";
#elif defined(TIMEMORY_UNIX)
    return "/";
#endif
}

std::string
path::inverse()
{
#if defined(TIMEMORY_WINDOWS)
    return "/";
#elif defined(TIMEMORY_UNIX)
    return "\\";
#endif
}

//
// OS-dependent representation
std::string
path::osrepr(std::string _path)
{
#if defined(TIMEMORY_WINDOWS)
    filepath::replace(_path, '/', "\\");
    filepath::replace(_path, "\\\\", "\\");
#elif defined(TIMEMORY_UNIX)
    filepath::replace(_path, '\\', "/");
    filepath::replace(_path, "//", "/");
#endif
    return _path;
}

// common representation
std::string
path::canonical(std::string _path)
{
    return filepath::canonical(std::move(_path));
}
//
}  // namespace utility
}  // namespace tim

#endif
