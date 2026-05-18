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

#include "timemory/macros/os.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>

#if defined(TIMEMORY_UNIX)
#    include <sys/ioctl.h>
#    include <unistd.h>
#elif defined(TIMEMORY_WINDOWS)
#    include <windows.h>  // GetConsoleScreenBufferInfo
// should already be included by os.hpp
#endif

namespace tim
{
namespace utility
{
namespace console
{
inline std::tuple<int32_t, std::string>
get_columns()
{
    using return_type = std::tuple<int32_t, std::string>;

#if defined(TIMEMORY_UNIX)
    struct winsize size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
    return return_type{ size.ws_col - 1, "ioctl" };
#elif defined(TIMEMORY_WINDOWS)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return return_type{ csbi.srWindow.Right - csbi.srWindow.Left,
                        "GetConsoleScreenBufferInfo" };
#else
    return return_type{ 0, "none" };
#endif
}
}  // namespace console
}  // namespace utility
}  // namespace tim
