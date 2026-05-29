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

#include "timemory/macros/language.hpp"
#include "timemory/utility/types.hpp"

namespace tim
{
namespace runtime
{
//
template <typename Tools, typename Func, typename... Args,
          typename Ret = std::result_of_t<Func(Args...)>,
          std::enable_if_t<!std::is_same<Ret, void>::value, int> = 0>
Ret
invoke(string_view_t&& label, Func&& func, Args&&... args)
{
    using tool_t = typename Tools::component_type;
    tool_t _obj{ std::forward<string_view_t>(label) };
    _obj.construct(std::forward<Args>(args)...);
    _obj.start();
    Ret ret = func(std::forward<Args>(args)...);
    _obj.stop();
    return ret;
}
//
template <typename Tools, typename Func, typename... Args,
          typename Ret = std::result_of_t<Func(Args...)>,
          std::enable_if_t<std::is_same<Ret, void>::value, int> = 0>
Ret
invoke(string_view_t&& label, Func&& func, Args&&... args)
{
    using tool_t = typename Tools::component_type;
    tool_t _obj{ std::forward<string_view_t>(label) };
    _obj.construct(std::forward<Args>(args)...);
    _obj.start();
    func(std::forward<Args>(args)...);
    _obj.stop();
}
//
}  // namespace runtime
}  // namespace tim
