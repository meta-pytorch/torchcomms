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

/**
 * \file timemory/config/config.hpp
 * \brief The declaration for the types for config without definitions
 */

#pragma once

#include "timemory/config/macros.hpp"
#include "timemory/config/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/settings/declaration.hpp"

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              config
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types, typename... Args,
          enable_if_t<(sizeof...(Types) > 0 && sizeof...(Args) >= 2), int>>
inline void
timemory_init(Args&&... _args)
{
    using types_type = tim::convert_t<tuple_concat_t<Types...>, tim::type_list<>>;
    manager::get_storage<types_type>::initialize();
    timemory_init(std::forward<Args>(_args)...);
}
//
//--------------------------------------------------------------------------------------//
//
namespace config
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Func>
void
read_command_line(Func&& _func)
{
#if defined(TIMEMORY_LINUX)
    auto _cmdline = tim::read_command_line(tim::process::get_target_id());
    if(tim::settings::verbose() > 1 || tim::settings::debug())
    {
        for(auto& itr : _cmdline)
            std::cout << itr << " ";
        std::cout << std::endl;
    }

    int    _argc = _cmdline.size();
    char** _argv = new char*[_argc];
    for(int i = 0; i < _argc; ++i)
        _argv[i] = (char*) _cmdline.at(i).c_str();
    _func(_argc, _argv);
    delete[] _argv;
#else
    consume_parameters(_func);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace config
}  // namespace tim
