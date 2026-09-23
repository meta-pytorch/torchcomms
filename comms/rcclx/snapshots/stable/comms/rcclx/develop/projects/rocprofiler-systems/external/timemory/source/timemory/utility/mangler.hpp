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

/** \file utility/mangler.hpp
 * \headerfile utility/mangler.hpp "timemory/utility/mangler.hpp"
 * Provides mangling for C++ functions
 *
 */

#pragma once

#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/type_id.hpp"

#include <string>
#include <tuple>

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace impl
{
template <typename... Args>
struct mangler
{
    static std::string mangle(std::string func, bool is_memfun, bool is_const, bool nsp)
    {
        auto nargs = sizeof...(Args);

        // non-namespaced functions must add "E";
        auto addE = (func.find("::") < func.find('('));

        std::string ret = "_Z";
        if(func.length() > 0 && func[0] == '&')
            func = func.substr(1);
        auto delim = delimit(func, ":()<>");
        if(delim.size() > 1)
            ret += "N";
        if(is_memfun && is_const)
            ret += "K";
        for(const auto& itr : delim)
        {
            ret += std::to_string(itr.length());
            ret += itr;
        }

        if(addE || nsp)
            ret += "E";
        if(nargs == 0)
        {
            ret += "v";
        }
        else
        {
            ret += mpl::apply<std::string>::join("", utility::type_id<Args>::name()...);
        }

        return ret;
    }
};

//--------------------------------------------------------------------------------------//

template <typename... Args>
struct mangler<type_list<Args...>>
{
    static std::string mangle(const std::string& func, bool is_memfun, bool is_const,
                              bool nsp = false)
    {
        return mangler<Args...>::mangle(func, is_memfun, is_const, nsp);
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//--------------------------------------------------------------------------------------//

template <typename FuncT, typename TraitsT = mpl::function_traits<FuncT>>
std::string
mangle(const std::string& func)
{
    using args_type          = typename TraitsT::args_type;
    constexpr bool is_memfun = TraitsT::is_memfun;
    constexpr bool is_const  = TraitsT::is_const;
    return impl::mangler<args_type>::mangle(func, is_memfun, is_const);
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
