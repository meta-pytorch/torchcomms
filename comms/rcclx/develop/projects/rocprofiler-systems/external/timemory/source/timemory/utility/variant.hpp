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
#include "timemory/mpl/concepts.hpp"

#include <utility>

#if defined(CXX17)
#    include <variant>

namespace tim
{
namespace utility
{
struct variant_assign_if_diff_index : std::true_type
{};

struct variant_ignore_if_diff_index : std::false_type
{};

template <typename VarT, typename FuncT, typename ArgT, typename AssignT,
          typename AssignFuncT>
inline decltype(auto)
variant_apply(VarT& _var, FuncT&& _func, ArgT&& _arg, AssignT, AssignFuncT _assign)
{
    if constexpr(concepts::is_unqualified_same<decltype(_var), decltype(_arg)>::value)
    {
        if(_var.index() != _arg.index())
        {
            if constexpr(AssignT::value)
                _assign(_var, _arg);
        }
        else
        {
            std::visit(std::forward<FuncT>(_func), _var, std::forward<ArgT>(_arg));
        }
    }
    else
    {
        std::visit(
            [&_func, &_arg](auto& _v) {
                std::forward<FuncT>(_func)(_v, std::forward<ArgT>(_arg));
            },
            _var);
    }
    return _var;
    (void) _assign;
}

template <typename VarT, typename FuncT, typename ArgT, typename AssignT>
inline decltype(auto)
variant_apply(VarT& _var, FuncT&& _func, ArgT&& _arg, AssignT _assign)
{
    return variant_apply(_var, std::forward<FuncT>(_func), std::forward<ArgT>(_arg),
                         _assign, [](VarT& _out, const VarT& _inp) { _out = _inp; });
}

template <typename VarT, typename FuncT, typename ArgT>
inline decltype(auto)
variant_apply(VarT& _var, FuncT&& _func, ArgT&& _arg)
{
    return variant_apply(_var, std::forward<FuncT>(_func), std::forward<ArgT>(_arg),
                         variant_assign_if_diff_index{});
}

template <typename VarT, typename FuncT>
inline decltype(auto)
variant_apply(VarT& _var, FuncT&& _func)
{
    std::visit(std::forward<FuncT>(_func), _var);
    return _var;
}
}  // namespace utility
}  // namespace tim
#endif
