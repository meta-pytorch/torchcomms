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

/** \file utility/filepath.hpp
 * \headerfile utility/filepath.hpp "timemory/utility/filepath.hpp"
 * Functions for converting OS filepaths
 *
 */

#pragma once

#ifndef TIMEMORY_UTILITY_FILEPATH_HPP_
#    define TIMEMORY_UTILITY_FILEPATH_HPP_
#endif

#include "timemory/log/macros.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/launch_process.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#if defined(TIMEMORY_UNIX)
#    include <execinfo.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#elif defined(TIMEMORY_WINDOWS)
#    include <direct.h>
#endif

#if defined(TIMEMORY_LINUX)
#    include <linux/limits.h>
#endif

#if defined(TIMEMORY_WINDOWS) && !defined(PATH_MAX)
#    if defined(MAX_PATH)
#        define PATH_MAX MAX_PATH
#    else
#        define PATH_MAX 260
#    endif
#endif

#if !defined(TIMEMORY_PATH_MAX)
#    if defined(PATH_MAX)
#        define TIMEMORY_PATH_MAX PATH_MAX
#    else
#        define TIMEMORY_PATH_MAX 4096
#    endif
#endif

namespace tim
{
namespace filepath
{
int
makedir(std::string _dir, int umask = TIMEMORY_DEFAULT_UMASK);

std::string
get_cwd();

std::string&
replace(std::string& _path, char _c, const char* _v);

std::string&
replace(std::string& _path, const char* _c, const char* _v);

std::string
os();

std::string
inverse();

std::string
osrepr(std::string _path);

std::string
canonical(std::string _path);

std::string dirname(std::string);

bool
exists(std::string _fname);

bool
direxists(std::string _fname);

const char*
basename(std::string_view _fname);

//--------------------------------------------------------------------------------------//

template <typename... Args>
inline bool
open(std::ofstream& _ofs, std::string _fpath, Args&&... _args)
{
    auto _path = canonical(_fpath);
    auto _base = canonical(_fpath);
    auto _pos  = _path.find_last_of('/');
    if(_pos != std::string::npos)
    {
        _path = _path.substr(0, _pos);
        _base = _base.substr(_pos + 1);
    }
    else
    {
        _path  = {};
        _fpath = std::string{ "./" } + _base;
    }

    if(!_path.empty())
    {
        auto ret = makedir(osrepr(_path));
        if(ret != 0)
            _fpath = std::string{ "./" } + _base;
    }

    _ofs.open(osrepr(_fpath), std::forward<Args>(_args)...);
    return (_ofs && _ofs.is_open() && _ofs.good());
}

//--------------------------------------------------------------------------------------//

template <typename... Args>
inline bool
open(std::ifstream& _ofs, std::string _fpath, Args&&... _args)
{
    auto _path = canonical(_fpath);
    auto _base = canonical(_fpath);
    auto _pos  = _path.find_last_of('/');
    if(_pos != std::string::npos)
    {
        _path = _path.substr(0, _pos);
        _base = _base.substr(_pos + 1);
    }
    else
    {
        _path  = {};
        _fpath = std::string{ "./" } + _base;
    }

    _ofs.open(osrepr(_fpath), std::forward<Args>(_args)...);
    return (_ofs && _ofs.is_open() && _ofs.good());
}

//--------------------------------------------------------------------------------------//

template <typename... Args>
inline auto*
fopen(std::string _fpath, const char* _mode)
{
    auto _path = canonical(_fpath);
    auto _base = canonical(_fpath);
    auto _pos  = _path.find_last_of('/');
    if(_pos != std::string::npos)
    {
        _path = _path.substr(0, _pos);
        _base = _base.substr(_pos + 1);
    }
    else
    {
        _path  = {};
        _fpath = std::string{ "./" } + _base;
    }

    if(!_path.empty())
    {
        auto ret = makedir(osrepr(_path));
        if(ret != 0)
            _fpath = std::string{ "./" } + _base;
    }

    return ::fopen(osrepr(_fpath).c_str(), _mode);
}

//--------------------------------------------------------------------------------------//

template <size_t MaxLen = TIMEMORY_PATH_MAX>
inline std::string
realpath(const std::string& _relpath, std::string* _resolved = nullptr,
         bool _warn_on_fail = true)
{
    auto _len = std::min<size_t>(_relpath.length(), TIMEMORY_PATH_MAX);

    char        _buffer[TIMEMORY_PATH_MAX] = { '\0' };
    const char* _result                    = _buffer;

#if defined(TIMEMORY_UNIX)
    if(::realpath(_relpath.c_str(), _buffer) == nullptr)
    {
        auto _err = errno;
        errno     = 0;
        if(_warn_on_fail)
        {
            TIMEMORY_PRINTF_WARNING(stderr, "realpath failed for '%s' :: %s\n", _relpath,
                                    strerror(_err));
        }
        _result = _relpath.data();
    }
#else
    if(_fullpath(_buffer, _relpath.c_str(), _len) == nullptr)
    {
        if(_warn_on_fail)
        {
            TIMEMORY_PRINTF_WARNING(stderr, "fullpath failed for '%s'\n");
        }
        _result = _relpath.data();
    }
#endif

    if(_resolved)
    {
        _resolved->clear();
        _len = strnlen(_result, TIMEMORY_PATH_MAX);
        _resolved->resize(_len);
        for(size_t i = 0; i < _len; ++i)
            (*_resolved)[i] = _result[i];
    }

    return (_resolved) ? *_resolved : std::string{ _result };
}

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_UNIX)
inline bool
is_link(const std::string& _path)
{
    struct stat _buffer;
    if(lstat(_path.c_str(), &_buffer) == 0)
        return (S_ISLNK(_buffer.st_mode) != 0);
    return false;
}

template <size_t MaxLen = TIMEMORY_PATH_MAX>
inline std::string
readlink(const std::string& _path)
{
    // if not a symbolic link, just return the path
    if(!is_link(_path))
        return _path;

    char    _buffer[MaxLen];
    ssize_t _buffer_len = MaxLen;
    _buffer_len         = ::readlink(_path.c_str(), _buffer, _buffer_len);
    if(_buffer_len < 0 || _buffer_len == (MaxLen))
    {
        auto* _path_rp = ::realpath(_path.c_str(), nullptr);
        if(_path_rp)
        {
            auto _ret = std::string{ _path_rp };
            free(_path_rp);
            return _ret;
        }
    }
    else
    {
        _buffer[_buffer_len] = '\0';
        return _buffer;
    }
    return _path;
}
#endif
}  // namespace filepath
}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE) && TIMEMORY_UTILITY_HEADER_MODE > 0
#    include "timemory/utility/filepath.cpp"
#endif
