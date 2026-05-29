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

#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <cstring>
#include <string>
#include <typeinfo>

namespace tim
{
std::string
demangle(const char* _mangled_name, int* _status = nullptr);

std::string
demangle(const std::string& _str, int* _status = nullptr);

inline std::string
demangle(std::string_view _str, int* _status = nullptr)
{
    return (_str.length() == strnlen(_str.data(), _str.length()))
               ? demangle(_str.data(), _status)
               : demangle(std::string{ _str }, _status);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline auto
demangle()
{
    // static because a type demangle will always be the same
    static auto _val = demangle(typeid(Tp).name());
    return _val;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline auto
try_demangle()
{
    // static because a type demangle will always be the same
    static auto _val = []() {
        // wrap the type in type_list and then extract ... from tim::type_list<...>
        auto _tmp = ::tim::demangle(typeid(type_list<Tp>).name());
        auto _key = std::string{ "type_list" };
        auto _idx = _tmp.find(_key);
        if(_idx == std::string::npos)
            return _tmp;
        _idx = _tmp.find('<', _idx);
        if(_idx == std::string::npos)
            return _tmp;
        _tmp = _tmp.substr(_idx + 1);
        _idx = _tmp.find_last_of('>');
        if(_idx == std::string::npos)
            return _tmp;
        _tmp = _tmp.substr(0, _idx);
        // strip trailing whitespaces
        while(!_tmp.empty() && (_idx = _tmp.find_last_of(' ')) == _tmp.length() - 1)
        {
            _tmp = _tmp.substr(0, _idx);
        }
        return _tmp;
    }();
    return _val;
}

//--------------------------------------------------------------------------------------//

std::string
demangle_native_backtrace(const char* cstr);

std::string
demangle_native_backtrace(const std::string& str);

#if defined(TIMEMORY_UNIX)
//
std::string
demangle_unw_backtrace(const char* cstr);

std::string
demangle_unw_backtrace(const std::string& str);
//
template <typename Tp>
std::string
demangle_backtrace(Tp&& _v)
{
    return demangle_unw_backtrace(demangle_native_backtrace(std::forward<Tp>(_v)));
}
//
#else
//
template <typename Tp>
std::string
demangle_backtrace(Tp&& _v)
{
    return demangle_native_backtrace(std::forward<Tp>(_v));
}
//
#endif

//--------------------------------------------------------------------------------------//

}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/demangle.cpp"
#endif
