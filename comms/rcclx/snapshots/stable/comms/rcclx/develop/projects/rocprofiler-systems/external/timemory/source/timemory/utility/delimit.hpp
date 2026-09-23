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

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
inline T
from_string(const std::string& str)
{
    std::stringstream ss{};
    ss << str;
    T val{};
    ss >> val;
    return val;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
inline T
from_string(const char* cstr)
{
    std::stringstream ss{};
    ss << cstr;
    T val{};
    ss >> val;
    return val;
}
//
//--------------------------------------------------------------------------------------//
//  delimit a string into a set
//
namespace utility
{
template <typename ContainerT, typename... Args>
inline auto
emplace_impl(ContainerT& _c, int, Args&&... _args)
    -> decltype(_c.emplace_back(std::forward<Args>(_args)...))
{
    return _c.emplace_back(std::forward<Args>(_args)...);
}

template <typename ContainerT, typename... Args>
inline auto
emplace_impl(ContainerT& _c, long, Args&&... _args)
    -> decltype(_c.emplace(std::forward<Args>(_args)...))
{
    return _c.emplace(std::forward<Args>(_args)...);
}

template <typename ContainerT, typename... Args>
inline auto
emplace(ContainerT& _c, Args&&... _args)
{
    return emplace_impl(_c, 0, std::forward<Args>(_args)...);
}

template <typename ContainerT, typename ArgT>
inline auto
reserve_impl(ContainerT& _c, int, ArgT _arg) -> decltype(_c.reserve(_arg), bool())
{
    _c.reserve(_arg);
    return true;
}

template <typename ContainerT, typename ArgT>
inline auto
reserve_impl(ContainerT&, long, ArgT)
{
    return false;
}

template <typename ContainerT, typename ArgT>
inline auto
reserve(ContainerT& _c, ArgT _arg)
{
    return reserve_impl(_c, 0, _arg);
}
}  // namespace utility
//
//--------------------------------------------------------------------------------------//
//  delimit a string into a set
//
template <typename ContainerT = std::vector<std::string>,
          typename PredicateT = std::function<std::string(const std::string&)>>
inline ContainerT
delimit(
    const std::string& line, std::string_view delimiters = "\"',;: ",
    PredicateT&& predicate = [](const std::string& s) -> std::string { return s; })
{
    size_t     _beginp = 0;  // position that is the beginning of the new string
    size_t     _delimp = 0;  // position of the delimiter in the string
    ContainerT _result = {};
    if(utility::reserve(_result, 0))
    {
        size_t _nmax = 0;
        for(char itr : line)
        {
            if(delimiters.find(itr) != std::string::npos)
                ++_nmax;
        }
        utility::reserve(_result, _nmax);
    }
    while(_beginp < line.length() && _delimp < line.length())
    {
        // find the first character (starting at _delimp) that is not a delimiter
        _beginp = line.find_first_not_of(delimiters, _delimp);
        // if no a character after or at _end that is not a delimiter is not found
        // then we are done
        if(_beginp == std::string::npos)
            break;
        // starting at the position of the new string, find the next delimiter
        _delimp = line.find_first_of(delimiters, _beginp);
        std::string _tmp{};
        try
        {
            // starting at the position of the new string, get the characters
            // between this position and the next delimiter
            if(_beginp < line.length())
                _tmp = line.substr(_beginp, _delimp - _beginp);
        } catch(std::exception& e)
        {
            // print the exception but don't fail, unless maybe it should?
            fprintf(stderr, "%s\n", e.what());
        }
        // don't add empty strings
        if(!_tmp.empty())
        {
            utility::emplace(_result, std::forward<PredicateT>(predicate)(_tmp));
        }
    }
    return _result;
}
//
//--------------------------------------------------------------------------------------//
///  \brief apply a string transformation to substring inbetween a common delimiter.
///  e.g.
//
template <typename PredicateT = std::function<std::string(const std::string&)>>
inline std::string
str_transform(const std::string& input, const std::string& _begin,
              const std::string& _end, PredicateT&& predicate)
{
    size_t      _beg_pos = 0;  // position that is the beginning of the new string
    size_t      _end_pos = 0;  // position of the delimiter in the string
    std::string _result  = input;
    while(_beg_pos < _result.length() && _end_pos < _result.length())
    {
        // find the first sequence of characters after the end-position
        _beg_pos = _result.find(_begin, _end_pos);

        // if sequence wasn't found, we are done
        if(_beg_pos == std::string::npos)
            break;

        // starting after the position of the first delimiter, find the end sequence
        if(!_end.empty())
            _end_pos = _result.find(_end, _beg_pos + 1);
        else
            _end_pos = _beg_pos + _begin.length();

        // break if not found
        if(_end_pos == std::string::npos)
            break;

        // length of the substr being operated on
        auto _len = _end_pos - _beg_pos;

        // get the substring between the two delimiters (including first delimiter)
        auto _sub = _result.substr(_beg_pos, _len);

        // apply the transform
        auto _transformed = predicate(_sub);

        // only replace if necessary
        if(_sub != _transformed)
        {
            _result = _result.replace(_beg_pos, _len, _transformed);
            // move end to the end of transformed string
            _end_pos = _beg_pos + _transformed.length();
        }
    }
    return _result;
}
}  // namespace tim
