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

/** \file runtime/enumerate.hpp
 * \headerfile runtime/enumerate.hpp "timemory/runtime/enumerate.hpp"
 * Provides implementation for initialize, enumerate_components which regularly
 * change as more features are added
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/environment/declaration.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/runtime/types.hpp"

#include <initializer_list>
#include <set>
#include <vector>

namespace tim
{
//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to generate an array of enumerations from a list of string
///      that can be subsequently used to initialize an auto_list or a component_list
///
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_list<wall_clock, cpu_clock, cpu_util, cuda_event>;
///
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::initialize(*obj, tim::enumerate_components({ "cpu_clock", "cpu_util"}));
///
template <typename StringT, typename... ExtraArgs,
          template <typename, typename...> class Container>
std::vector<TIMEMORY_COMPONENT>
enumerate_components(const Container<StringT, ExtraArgs...>& component_names)
{
    std::set<TIMEMORY_COMPONENT> _set;
    for(const auto& itr : component_names)
        _set.insert(runtime::enumerate(std::string(itr)));
    std::vector<TIMEMORY_COMPONENT> _vec;
    _vec.reserve(_set.size());
    for(auto&& itr : _set)
        _vec.emplace_back(itr);
    return _vec;
}

//--------------------------------------------------------------------------------------//

inline std::set<TIMEMORY_COMPONENT>
enumerate_components(const std::initializer_list<std::string>& component_names)
{
    return enumerate_components(std::set<std::string>(component_names));
}

//--------------------------------------------------------------------------------------//

template <typename StringT = std::string>
std::vector<TIMEMORY_COMPONENT>
enumerate_components(const std::string& names, const StringT& env_id = "")
{
    if(std::string(env_id).length() > 0)
    {
        return enumerate_components(tim::delimit(get_env<std::string>(env_id, names)));
    }
    {
        return enumerate_components(tim::delimit(names));
    }
}

//======================================================================================//

template <typename... ExtraArgs>
std::set<TIMEMORY_COMPONENT>
enumerate_components(const std::set<std::string, ExtraArgs...>& component_names)
{
    std::set<TIMEMORY_COMPONENT> vec;
    for(const auto& itr : component_names)
        vec.insert(runtime::enumerate(itr));
    return vec;
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
