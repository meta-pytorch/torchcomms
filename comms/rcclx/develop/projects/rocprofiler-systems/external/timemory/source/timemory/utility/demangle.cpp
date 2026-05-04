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

#ifndef TIMEMORY_UTILITY_DEMANGLE_CPP_
#define TIMEMORY_UTILITY_DEMANGLE_CPP_

#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#if !defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/demangle.hpp"
#endif

#include "timemory/utility/delimit.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <utility>

#if defined(TIMEMORY_UNIX)
#    include <cxxabi.h>
#endif

namespace tim
{
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle(const char* _mangled_name, int* _status)
{
#if defined(TIMEMORY_ENABLE_DEMANGLE)
    // return the mangled since there is no buffer
    if(!_mangled_name)
        return std::string{};

    int         _ret = 0;
    std::string _demangled_name{ _mangled_name };
    if(!_status)
        _status = &_ret;

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into which the
    //      demangled name is stored. If output_buffer is not long enough, it is expanded
    //      using realloc. output_buffer may instead be NULL; in that case, the demangled
    //      name is placed in a region of memory allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the demangled name
    //      is placed in *length.
    //  status:
    //      *status is set to one of the following values
    char* _demang = abi::__cxa_demangle(_mangled_name, nullptr, nullptr, _status);
    switch(*_status)
    {
        //  0 : The demangling operation succeeded.
        // -1 : A memory allocation failiure occurred.
        // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
        // -3 : One of the arguments is invalid.
        case 0:
        {
            if(_demang)
                _demangled_name = std::string{ _demang };
            break;
        }
        case -1:
        {
            char _msg[1024];
            ::memset(_msg, '\0', 1024 * sizeof(char));
            ::snprintf(_msg, 1024, "memory allocation failure occurred demangling %s",
                       _mangled_name);
            ::perror(_msg);
            break;
        }
        case -2: break;
        case -3:
        {
            char _msg[1024];
            ::memset(_msg, '\0', 1024 * sizeof(char));
            ::snprintf(_msg, 1024, "Invalid argument in: (\"%s\", nullptr, nullptr, %p)",
                       _mangled_name, (void*) _status);
            ::perror(_msg);
            break;
        }
        default: break;
    };

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
#else
    (void) _status;
    return _mangled_name;
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE std::string
                        demangle(const std::string& _str, int* _status)
{
    return demangle(_str.c_str(), _status);
}

#if defined(TIMEMORY_UNIX)
//
TIMEMORY_UTILITY_INLINE
std::string
demangle_native_backtrace(const char* cstr)
{
    auto _trim = [](std::string& _sub, size_t& _len) {
        size_t _pos = 0;
        while(!_sub.empty() && (_pos = _sub.find_first_of(' ')) == 0)
        {
            _sub = _sub.erase(_pos, 1);
            --_len;
        }
        while(!_sub.empty() && (_pos = _sub.find_last_of(' ')) == _sub.length() - 1)
        {
            _sub = _sub.substr(0, _sub.length() - 1);
            --_len;
        }
        return _sub;
    };

    auto _save = std::string{ cstr };
    if(_save.length() != strlen(cstr))
        TIMEMORY_EXCEPTION("length != strlen: " << _save.length() << " vs. "
                                                << strlen(cstr) << " :: " << cstr);
    auto str = demangle(std::string{ cstr });
    if(str.empty())
        return str;
    auto beg = str.find('(');
    if(beg == std::string::npos)
    {
        beg = str.find("_Z");
        if(beg != std::string::npos)
            beg -= 1;
    }
    auto end = str.find('+', beg);
    if(beg != std::string::npos && end != std::string::npos)
    {
        auto len = end - (beg + 1);
        auto sub = str.substr(beg + 1, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg + 1, len, dem);
    }
    else if(beg != std::string::npos)
    {
        auto len = str.length() - (beg + 1);
        auto sub = str.substr(beg + 1, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg + 1, len, dem);
    }
    else if(end != std::string::npos)
    {
        auto len = end;
        auto sub = str.substr(beg, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg, len, dem);
    }

    if(str.empty())
    {
        TIMEMORY_TESTING_EXCEPTION(__FUNCTION__ << " resulted in empty string. input: "
                                                << _save);
        return _save;
    }

    // cleanup std::_1::basic_<...> types since demangled versions are for diagnostics
    using pair_t = std::pair<std::string, std::string>;
    for(auto&& itr : { pair_t{ demangle<std::string>(), "std::string" },
                       pair_t{ demangle<std::istream>(), "std::istream" },
                       pair_t{ demangle<std::ostream>(), "std::ostream" },
                       pair_t{ demangle<std::stringstream>(), "std::stringstream" },
                       pair_t{ demangle<std::istringstream>(), "std::istringstream" },
                       pair_t{ demangle<std::ostringstream>(), "std::ostringstream" } })
        str = str_transform(str, itr.first, "", [itr](const std::string&) -> std::string {
            return itr.second;
        });
    return str;
}
//
TIMEMORY_UTILITY_INLINE
std::string
demangle_native_backtrace(const std::string& str)
{
    return demangle_native_backtrace(str.c_str());
}
//
TIMEMORY_UTILITY_INLINE
std::string
demangle_unw_backtrace(const char* cstr)
{
    auto _save = std::string{ cstr };
    if(_save.length() != strlen(cstr))
        TIMEMORY_EXCEPTION("length != strlen: " << _save.length() << " vs. "
                                                << strlen(cstr) << " :: " << cstr);
    auto _str = demangle(cstr);
    if(_str.empty())
        return _str;
    auto _beg = _str.find("_Z");
    auto _end = _str.find(' ', _beg);
    if(_beg != std::string::npos && _end != std::string::npos)
    {
        auto _len = _end - _beg;
        auto _sub = _str.substr(_beg, _len);
        auto _dem = demangle(_sub);
        _str      = _str.replace(_beg, _len, _dem);
    }

    if(_str.empty())
    {
        TIMEMORY_TESTING_EXCEPTION(__FUNCTION__ << " resulted in empty string. input: "
                                                << _save);
        return _save;
    }

    // cleanup std::_1::basic_<...> types since demangled versions are for diagnostics
    using pair_t = std::pair<std::string, std::string>;
    for(auto&& itr : { pair_t{ demangle<std::string>(), "std::string" },
                       pair_t{ demangle<std::istream>(), "std::istream" },
                       pair_t{ demangle<std::ostream>(), "std::ostream" },
                       pair_t{ demangle<std::stringstream>(), "std::stringstream" },
                       pair_t{ demangle<std::istringstream>(), "std::istringstream" },
                       pair_t{ demangle<std::ostringstream>(), "std::ostringstream" } })
        _str =
            str_transform(_str, itr.first, "", [itr](const std::string&) -> std::string {
                return itr.second;
            });

    return _str;
}
//
TIMEMORY_UTILITY_INLINE
std::string
demangle_unw_backtrace(const std::string& _str)
{
    return demangle_unw_backtrace(_str.c_str());
}
//
#else
//
TIMEMORY_UTILITY_INLINE
std::string
demangle_backtrace(const char* cstr)
{
    return std::string{ cstr };
}
//
TIMEMORY_UTILITY_INLINE
std::string
demangle_backtrace(const std::string& str)
{
    return str;
}
//
TIMEMORY_UTILITY_INLINE
std::string
demangle_native_backtrace(const char* cstr)
{
    return std::string{ cstr };
}
//
TIMEMORY_UTILITY_INLINE
std::string
demangle_native_backtrace(const std::string& str)
{
    return str;
}
//
#endif  // defined(TIMEMORY_UNIX)
}  // namespace tim

#endif
