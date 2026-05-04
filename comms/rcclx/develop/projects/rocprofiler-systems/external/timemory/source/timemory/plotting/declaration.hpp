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

#ifndef TIMEMORY_PLOTTING_DECLARATION_HPP_
#    define TIMEMORY_PLOTTING_DECLARATION_HPP_
#endif

#include "timemory/plotting/macros.hpp"
#include "timemory/plotting/types.hpp"
#include "timemory/settings/declaration.hpp"

#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tim
{
namespace plotting
{
namespace operation
{
template <typename Arg, typename... Args>
auto
join(const char* sep, Arg&& arg, Args&&... args)
{
    std::stringstream ss;
    ss << std::forward<Arg>(arg);
    tim::consume_parameters(
        ::std::initializer_list<int>{ (ss << sep << std::forward<Args>(args), 0)... });
    return ss.str();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
plot(std::string _prefix, std::optional<std::string> _dir, std::optional<bool> _echo_dart,
     std::string _json_file)
{
    if(!_dir)
        _dir = settings::output_path();
    if(!_echo_dart)
        _echo_dart = settings::dart_output();

    if constexpr(std::is_same<typename Tp::value_type, void>::value)
    {
        if(settings::debug() || settings::verbose() > 2)
            TIMEMORY_PRINT_HERE("%s", "");
        return;
    }

    if(!settings::json_output() && !trait::requires_json<Tp>::value)
    {
        if(settings::debug() || settings::verbose() > 2)
            TIMEMORY_PRINT_HERE("%s", "");
        return;
    }

    if(_prefix.empty())
        _prefix = Tp::get_description();

    if(_json_file.empty())
        _json_file = settings::compose_output_filename(Tp::get_label(), ".json");

    auto _label = demangle<Tp>();

    plot(_label, _prefix, *_dir, *_echo_dart, _json_file);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
template <typename... Types>
struct plot
{
    static void generate(std::string _prefix, const std::string& _dir, bool _echo,
                         std::string _json)
    {
        TIMEMORY_FOLD_EXPRESSION(operation::plot<Types>(_prefix, _dir, _echo, _json));
    }
};
//
template <typename... Types>
struct plot<std::tuple<Types...>> : plot<Types...>
{};
//
template <typename... Types>
struct plot<type_list<Types...>> : plot<Types...>
{};
}  // namespace impl
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
std::enable_if_t<(sizeof...(Types) > 0), void>
plot(const std::string& _prefix, const std::string& _dir, bool _echo_dart,
     const std::string& _json_file)
{
    impl::plot<Types...>::generate(_prefix, _dir, _echo_dart, _json_file);
}
}  // namespace plotting
}  // namespace tim
