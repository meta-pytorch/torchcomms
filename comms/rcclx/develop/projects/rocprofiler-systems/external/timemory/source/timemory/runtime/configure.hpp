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

/** \file runtime/configure.hpp
 * \headerfile runtime/configure.hpp "timemory/runtime/configure.hpp"
 * Provides implementation for initialize, enumerate_components which regularly
 * change as more features are added
 *
 */

#pragma once

#include "timemory/enum.h"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/types.hpp"

#include <initializer_list>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tim
{
//======================================================================================//

template <typename Bundle, typename EnumT, typename... Args>
void
configure(std::initializer_list<EnumT> components, Args&&... args)
{
    configure<Bundle>(std::vector<EnumT>(components), std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

template <typename Bundle, typename... Args>
void
configure(const std::initializer_list<std::string>& components, Args&&... args)
{
    configure<Bundle>(enumerate_components(components), std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a container of string
//
template <typename Bundle, typename... ExtraArgs,
          template <typename, typename...> class Container, typename... Args>
void
configure(const Container<std::string, ExtraArgs...>& components, Args&&... args)
{
    configure<Bundle>(enumerate_components(components), std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a string
//
template <typename Bundle, typename... Args>
void
configure(const std::string& components, Args&&... args)
{
    configure<Bundle>(enumerate_components(tim::delimit(components)),
                      std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

template <
    typename Bundle, template <typename, typename...> class Container, typename Intp,
    typename... ExtraArgs, typename... Args,
    typename std::enable_if<std::is_integral<Intp>::value ||
                                std::is_same<Intp, TIMEMORY_NATIVE_COMPONENT>::value,
                            int>::type>
void
configure(const Container<Intp, ExtraArgs...>& components, Args&&... args)
{
    for(auto itr : components)
        runtime::configure<Bundle>(itr, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

template <typename Bundle, template <typename, typename...> class Container,
          typename... ExtraArgs, typename... Args>
void
configure(const Container<const char*, ExtraArgs...>& components, Args&&... args)
{
    std::unordered_set<std::string> _components;
    for(auto itr : components)
        _components.insert(std::string(itr));
    configure<Bundle>(_components, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

template <typename Bundle, typename... Args>
void
configure(const int ncomponents, const int* components, Args&&... args)
{
    for(int i = 0; i < ncomponents; ++i)
        runtime::configure<Bundle>(components[i], std::forward<Args>(args)...);
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
