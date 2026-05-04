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
#include "timemory/process/defines.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#if defined(TIMEMORY_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(TIMEMORY_MACOS)
#        include <libproc.h>
#        include <mach/mach.h>
#    endif
#elif defined(TIMEMORY_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#    if !defined(WIN32_LEAN_AND_MEAN)
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#endif

namespace tim
{
namespace process
{
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_UNIX)
using id_t = pid_t;
#elif defined(TIMEMORY_WINDOWS)
using id_t = DWORD;
#else
using id_t = int;
#endif
//
//--------------------------------------------------------------------------------------//
//
/// \fn pid_t tim::process::get_id()
/// \brief get the process id of this process
///
inline id_t
get_id()
{
#if defined(TIMEMORY_WINDOWS)
    return GetCurrentProcessId();
#elif defined(TIMEMORY_UNIX)
    return getpid();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
/// \fn pid_t& tim::process::get_target_id()
/// \brief get the target process id of this process (may differ from process::get_id)
///
inline id_t&
get_target_id()
{
    static auto instance = get_id();
    return instance;
}
//
//--------------------------------------------------------------------------------------//
//
inline id_t
get_parent_id()
{
#if defined(TIMEMORY_UNIX)
    return getppid();
#else
    return get_id();
#endif
}
//
inline id_t
get_group_id(id_t _id = get_id())
{
#if defined(TIMEMORY_UNIX)
    return getpgid(_id);
#else
    return get_id();
    (void) _id;
#endif
}
//
inline id_t
get_session_id(id_t _id = get_id())
{
#if defined(TIMEMORY_UNIX)
    return getsid(_id);
#else
    return get_id();
    (void) _id;
#endif
}
//
inline std::vector<id_t>
get_siblings(id_t _id = get_parent_id())
{
    auto _data = std::vector<id_t>{};

#if defined(TIMEMORY_UNIX)
    std::ifstream _ifs{ "/proc/" + std::to_string(_id) + "/task/" + std::to_string(_id) +
                        "/children" };
    while(_ifs)
    {
        id_t _n = 0;
        _ifs >> _n;
        if(!_ifs || _n <= 0)
            break;
        _data.emplace_back(_n);
    }
#endif
    return _data;
}
//
inline auto
get_num_siblings(id_t _id = get_parent_id())
{
    return get_siblings(_id).size();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace process
}  // namespace tim
