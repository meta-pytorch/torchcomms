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

#ifndef TIMEMORY_UTILITY_FILEPATH_HPP_
#    include "timemory/utility/filepath.hpp"
#endif

#include "timemory/macros/os.hpp"
#include "timemory/utility/delimit.hpp"

#if defined(TIMEMORY_LINUX)
#    include <unistd.h>
#endif

#if defined(TIMEMORY_MACOS)
#    include <libgen.h>
#endif

#if defined(TIMEMORY_USE_SHLWAPI)
#    include <shlwapi.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace tim
{
namespace filepath
{
TIMEMORY_UTILITY_INLINE
std::string
get_cwd()
{
    char _default_buffer[PATH_MAX];
    _default_buffer[0] = '\0';

    size_t _sz     = PATH_MAX;
    char*  _buffer = _default_buffer;

    auto _alloc_buffer = [&]() {
        if(_buffer != _default_buffer)
            delete[] _buffer;
        _sz *= 2;
        _buffer    = new char[_sz];
        _buffer[0] = '\0';
    };

#if defined(TIMEMORY_WINDOWS)
    while(_getcwd(_buffer, _sz) == nullptr)
#else
    while(getcwd(_buffer, _sz) == nullptr)
#endif
    {
        auto _err = errno;
        if(_err == ERANGE)
        {
            _alloc_buffer();
            continue;
        }
        else
        {
            TIMEMORY_PRINTF_WARNING(stderr, "getcwd failed :: %s\n", strerror(_err));
            return std::string{};
        }
    }

    auto _v = std::string{ _buffer };
    if(_buffer != _default_buffer)
        delete[] _buffer;
    return _v;
}

TIMEMORY_UTILITY_INLINE int
makedir(std::string _dir, int umask)
{
#if defined(TIMEMORY_UNIX)
    _dir = osrepr(_dir);
    if(_dir.empty())
        return 0;

    if(direxists(_dir))
        return 0;

    auto _make_dir = [umask](std::string_view _v) {
        int _err = 0;
        int _ret = ::mkdir(_v.data(), umask);
        if(_ret != 0)
            _err = errno;
        return _err;
    };

    if(_make_dir(_dir) != 0)
    {
        std::string _base = (_dir.find('/') == 0) ? "" : get_cwd();
        for(const auto& itr : delimit(_dir, "/"))
        {
            if(itr == ".")
            {
                continue;
            }
            else if(itr == "..")
            {
                _base = dirname(_base);
            }
            else
            {
                _base += "/" + itr;
                if(!direxists(_base))
                {
                    auto _err = _make_dir(_base);
                    if(_err != 0)
                    {
                        TIMEMORY_PRINTF_WARNING(stderr, "mkdir(\"%s\", %i) failed: %s\n",
                                                _base.c_str(), umask, strerror(_err));
                        return -1;
                    }
                }
            }
        }
    }
#elif defined(TIMEMORY_WINDOWS)
    _dir = osrepr(_dir);
    if(_dir.empty())
        return 0;

    int ret = _mkdir(_dir.c_str());
    if(ret != 0)
    {
        int err = errno;
        if(err != EEXIST)
        {
            std::stringstream _mdir{};
            _mdir << "_mkdir(" << _dir.c_str() << ") returned: " << ret << "\n";
            std::stringstream _sdir{};
            _sdir << "mkdir " << _dir;
            ret = (launch_process(_sdir.str().c_str())) ? 0 : 1;
            if(ret != 0)
            {
                std::cerr << _mdir.str() << _sdir.str() << " returned: " << ret
                          << std::endl;
            }
            return ret;
        }
    }
#endif
    return 0;
}

TIMEMORY_UTILITY_INLINE std::string&
                        replace(std::string& _path, char _c, const char* _v)
{
    auto _pos = std::string::npos;
    while((_pos = _path.find(_c)) != std::string::npos)
        _path.replace(_pos, 1, _v);
    return _path;
}

TIMEMORY_UTILITY_INLINE std::string&
                        replace(std::string& _path, const char* _c, const char* _v)
{
    auto _pos = std::string::npos;
    while((_pos = _path.find(_c)) != std::string::npos)
        _path.replace(_pos, strlen(_c), _v);
    return _path;
}

#if defined(TIMEMORY_WINDOWS)

TIMEMORY_UTILITY_INLINE std::string
                        os()
{
    return "\\";
}

TIMEMORY_UTILITY_INLINE std::string
                        inverse()
{
    return "/";
}

TIMEMORY_UTILITY_INLINE std::string
                        osrepr(std::string _path)
{
    // OS-dependent representation
    while(_path.find('/') != std::string::npos)
        _path.replace(_path.find('/'), 1, "\\");
    return _path;
}

#elif defined(TIMEMORY_UNIX)

TIMEMORY_UTILITY_INLINE std::string
os()
{
    return "/";
}

TIMEMORY_UTILITY_INLINE std::string
inverse()
{
    return "\\";
}

TIMEMORY_UTILITY_INLINE std::string
osrepr(std::string _path)
{
    // OS-dependent representation
    while(_path.find("\\\\") != std::string::npos)
        _path.replace(_path.find("\\\\"), 2, "/");
    while(_path.find('\\') != std::string::npos)
        _path.replace(_path.find('\\'), 1, "/");
    return _path;
}

#endif

TIMEMORY_UTILITY_INLINE std::string
                        canonical(std::string _path)
{
    replace(_path, '\\', "/");
    replace(_path, "//", "/");
    return _path;
}

TIMEMORY_UTILITY_INLINE std::string
                        canonicalize(std::string _path)
{
    replace(_path, '\\', "/");
    replace(_path, "//", "/");

    if(_path.find("./") == 0)
        _path = _path.replace(0, 1, get_cwd());
    else if(_path.find("../") == 0)
        _path = _path.insert(0, get_cwd() + "/");

    auto _leading_dash = (_path.find('/') == 0);
    auto _pieces       = delimit(_path, "/");
    std::reverse(_pieces.begin(), _pieces.end());
    auto _tree = std::vector<std::string>{};
    for(size_t i = 0; i < _pieces.size(); ++i)
    {
        const auto& itr = _pieces.at(i);
        if(itr == ".")
        {
            continue;
        }
        else if(itr == "..")
            ++i;
        else
            _tree.emplace_back(itr);
    }
    std::reverse(_tree.begin(), _tree.end());
    auto _cpath = std::string{ (_leading_dash) ? "/" : "" };
    for(size_t i = 0; i < _tree.size() - 1; ++i)
        _cpath += _tree.at(i) + "/";
    _cpath += _tree.back();
    return _cpath;
}

TIMEMORY_UTILITY_INLINE std::string
                        dirname(std::string _fname)
{
#if defined(TIMEMORY_UNIX)
    // _fname = realpath(_fname, nullptr, false);
    while(_fname.find("\\\\") != std::string::npos)
        _fname.replace(_fname.find("\\\\"), 2, "/");
    while(_fname.find('\\') != std::string::npos)
        _fname.replace(_fname.find('\\'), 1, "/");

    auto _pos = _fname.find_last_of('/');
    return (_pos != std::string::npos) ? _fname.substr(0, _pos) : _fname;
#elif defined(TIMEMORY_WINDOWS)
    while(_fname.find('/') != std::string::npos)
        _fname.replace(_fname.find('/'), 1, "\\");

    _fname = _fname.substr(0, _fname.find_last_of('\\'));
    return (_fname.at(_fname.length() - 1) == '\\')
               ? _fname.substr(0, _fname.length() - 1)
               : _fname;
#endif
}

TIMEMORY_UTILITY_INLINE bool
exists(std::string _fname)
{
    _fname = osrepr(_fname);
#if defined(TIMEMORY_UNIX)
    struct stat _buffer;
    if(stat(_fname.c_str(), &_buffer) == 0)
        return (S_ISREG(_buffer.st_mode) != 0 || S_ISLNK(_buffer.st_mode) != 0);
    return false;
#elif defined(TIMEMORY_USE_SHLWAPI)
    return PathFileExists(_fname.c_str());
#else
    return false;
#endif
}

TIMEMORY_UTILITY_INLINE bool
direxists(std::string _fname)
{
    _fname = osrepr(_fname);
#if defined(TIMEMORY_UNIX)
    struct stat _buffer;
    if(stat(_fname.c_str(), &_buffer) == 0)
        return (S_ISDIR(_buffer.st_mode) != 0);
    return false;
#elif defined(TIMEMORY_USE_SHLWAPI)
    return PathIsDirectory(_fname.c_str());
#else
    return false;
#endif
}

TIMEMORY_UTILITY_INLINE
const char*
basename(std::string_view _fname)
{
#if defined(TIMEMORY_LINUX)
    return ::basename(_fname.data());
#else
#    if defined(TIMEMORY_WINDOWS)
    auto _pos = _fname.find_last_of("/\\");
#    else
    auto _pos = _fname.find_last_of('/');
#    endif
    if(_pos < _fname.length())
        return _fname.data() + _pos + 1;
    return _fname.data();
#endif
}
}  // namespace filepath
}  // namespace tim

//--------------------------------------------------------------------------------------//
