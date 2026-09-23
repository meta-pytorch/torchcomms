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

#include "timemory/api.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/defines.h"
#include "timemory/log/logger.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wformat-security"
#endif

namespace tim
{
namespace operation
{
template <typename Tp, typename ApiT = TIMEMORY_API>
struct file_output_message;
//
template <typename Tp>
struct file_output_message<Tp, void>
{
    template <typename... Args>
    auto operator()(const std::string& _name, const char* const _format = nullptr,
                    Args... _args)
    {
        generic({ _name }, { std::string{} }, _format, _args...);
    }

    template <typename... Args>
    auto operator()(const std::string& _name, std::string&& _preamble,
                    const char* const _format = nullptr, Args... _args)
    {
        generic({ _name }, { _preamble }, _format, _args...);
    }

    template <typename... Args>
    auto operator()(const std::vector<std::string>& _name,
                    const std::vector<std::string>& _preamble,
                    const char*                     _format = nullptr, Args... _args)
    {
        generic(_name, { _preamble }, _format, _args...);
    }

    ~file_output_message()
    {
        if(m_used)
        {
            fprintf(stderr, "%s\n", tim::log::color::end());
            fflush(stderr);
        }
    }

    template <typename Arg, typename... Args>
    auto append(Arg&& _arg, Args... _args) const
    {
        if(!m_used)
        {
            fflush(stderr);
            int _id = process::get_id();
            if(dmp::is_initialized())
                _id = dmp::rank();
            fprintf(stderr, "%s[%s][%i]> ", tim::log::color::source(),
                    TIMEMORY_PROJECT_NAME, _id);
            m_used = true;
        }
        fprintf(stderr, log::color::source());
        fprintf(stderr, std::forward<Arg>(_arg), timemory_proxy_value(_args, 0)...);
    }

protected:
    template <typename... Args>
    void generic(const std::vector<std::string>& _filenames,
                 const std::vector<std::string>& _preambles, const char* const _format,
                 Args... _args) const
    {
        auto _patch_preamble = [](std::string _v) {
            if(_v.front() != '[')
                _v = std::string{ "[" } + _v;
            if(_v.back() != ']')
                _v += std::string{ "]" };
            return _v;
        };

        std::string _preamble{};
        for(const auto& itr : _preambles)
            _preamble += _patch_preamble(itr);

        std::string _filename{};
        for(const auto& itr : _filenames)
        {
            if(!_filename.empty())
                _filename += " and ";
            _filename += std::string{ "'" } + itr + std::string{ "'" };
        }

        fflush(stderr);
        if(!m_used)
        {
            int _id = process::get_id();
            if(dmp::is_initialized())
                _id = dmp::rank();
            fprintf(stderr, "%s[%s][%i]%s> ", tim::log::color::source(),
                    TIMEMORY_PROJECT_NAME, _id, _preamble.c_str());
        }

        fprintf(stderr, "Outputting %s", _filename.c_str());

        if(_format)
            fprintf(stderr, _format, timemory_proxy_value(_args, 0)...);

        m_used = true;
    }

private:
    mutable bool m_used = false;
};
//
template <typename Tp, typename ApiT>
struct file_output_message : file_output_message<Tp, void>
{
    using base_type = file_output_message<Tp, void>;

    using base_type::operator();
    using base_type::append;
};
}  // namespace operation
}  // namespace tim

#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
