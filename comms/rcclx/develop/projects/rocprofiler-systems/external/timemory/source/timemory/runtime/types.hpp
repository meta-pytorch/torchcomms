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

/** \file runtime/types.hpp
 * \headerfile runtime/types.hpp "timemory/runtime/types.hpp"
 * Declarations
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/environment/declaration.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/runtime/macros.hpp"
#include "timemory/utility/delimit.hpp"

#include <initializer_list>
#include <string>

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
enumerate_components(const Container<StringT, ExtraArgs...>& component_names);

template <typename... ExtraArgs>
std::set<TIMEMORY_COMPONENT>
enumerate_components(const std::set<std::string, ExtraArgs...>& component_names);

std::set<TIMEMORY_COMPONENT>
enumerate_components(const std::initializer_list<std::string>& component_names);

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to initialize a auto_list or component_list from a list
///      of enumerations
//
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_list<wall_clock, cpu_clock, cpu_util, cuda_event>;
//
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::initialize(*obj, { CPU_CLOCK, CPU_UTIL });
//
///  typename... ExtraArgs
///      required because of extra "hidden" template parameters in STL containers
//
template <
    template <typename...> class CompList, typename... CompTypes,
    template <typename, typename...> class Container, typename Intp,
    typename... ExtraArgs,
    typename std::enable_if<std::is_integral<Intp>::value ||
                                std::is_same<Intp, TIMEMORY_NATIVE_COMPONENT>::value,
                            int>::type = 0>
void
initialize(CompList<CompTypes...>& obj, const Container<Intp, ExtraArgs...>& components);

template <typename T, typename... Args>
void
initialize(T* obj, Args&&... args)
{
    if(obj)
        initialize(*obj, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to insert tools into a bundle
//
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_tuple<user_global_bundle>;
//
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::insert(obj.get<user_global_bundle>(), { CPU_CLOCK, CPU_UTIL });
//
///  typename... ExtraArgs
///      required because of extra "hidden" template parameters in STL containers
//
template <
    size_t Idx, typename Type, template <size_t, typename> class Bundle,
    template <typename, typename...> class Container, typename Intp,
    typename... ExtraArgs,
    typename std::enable_if<std::is_integral<Intp>::value ||
                                std::is_same<Intp, TIMEMORY_NATIVE_COMPONENT>::value,
                            int>::type = 0>
void
insert(Bundle<Idx, Type>& obj, const Container<Intp, ExtraArgs...>& components);

template <typename T, typename... Args>
void
insert(T* obj, Args&&... args)
{
    if(obj)
        insert(*obj, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to insert tools into a bundle
//
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_tuple<user_global_bundle>;
///
///      tim::configure<user_global_bundle>({ CPU_CLOCK, CPU_UTIL });
///
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
//
///  typename... ExtraArgs
///      required because of extra "hidden" template parameters in STL containers
//
template <
    typename Bundle_t, template <typename, typename...> class Container, typename Intp,
    typename... ExtraArgs, typename... Args,
    typename std::enable_if<std::is_integral<Intp>::value ||
                                std::is_same<Intp, TIMEMORY_NATIVE_COMPONENT>::value,
                            int>::type = 0>
void
configure(const Container<Intp, ExtraArgs...>& components, Args&&...);

template <typename Bundle, typename EnumT = int, typename... Args>
void
configure(std::initializer_list<EnumT> components, Args&&... args);

template <typename Bundle, typename... Args>
void
configure(const std::initializer_list<std::string>& components, Args&&... args);

template <typename Bundle, typename... ExtraArgs,
          template <typename, typename...> class Container, typename... Args>
void
configure(const Container<std::string, ExtraArgs...>& components, Args&&... args);

template <typename Bundle, typename... Args>
void
configure(const std::string& components, Args&&... args);

template <typename Bundle, template <typename, typename...> class Container,
          typename... ExtraArgs, typename... Args>
void
configure(const Container<const char*, ExtraArgs...>& components, Args&&... args);

template <typename Bundle, typename... Args>
void
configure(int ncomponents, const int* components, Args&&... args);

//======================================================================================//

namespace runtime
{
//
template <typename Tp, typename Arg, typename... Args>
void
initialize(Tp& obj, int idx, Arg&&, Args&&...);
//
template <typename Tp, typename Arg, typename... Args>
void
insert(Tp& obj, int idx, Arg&&, Args&&...);
//
template <typename Tp, typename Arg, typename... Args>
void
configure(int idx, Arg&&, Args&&... args);
//
template <typename Tp, typename Arg, typename... Args>
void
configure(Tp& obj, int idx, Arg&&, Args&&... args);
//
int
enumerate(string_view_cref_t key);
//
component::info
get_info(int);
//
component::info
get_info(string_view_cref_t key);
//
template <typename Tp>
void
initialize(Tp& obj, int idx);
//
template <typename Tp>
void
insert(Tp& obj, int idx);
//
template <typename Tp>
void
configure(int idx);
//
template <typename Tp>
void
configure(Tp& obj, int idx);
//
template <typename Tp>
void
insert(Tp& obj, int idx, scope::config _scope);
//
template <typename Tp>
void
configure(int idx, scope::config _scope);
//
template <typename Tp>
void
configure(Tp& obj, int idx, scope::config _scope);
//
}  // namespace runtime

//======================================================================================//

namespace env
{
//--------------------------------------------------------------------------------------//

template <template <typename...> class CompList, typename... CompTypes,
          typename std::enable_if<sizeof...(CompTypes) != 0, int>::type = 0>
void
initialize(CompList<CompTypes...>& obj, const std::string& env_var,
           const std::string& default_env)
{
    auto env_result = get_env(env_var, default_env);
    tim::initialize(obj, enumerate_components(delimit(env_result)));
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class CompList, typename... CompTypes,
          typename std::enable_if<sizeof...(CompTypes) == 0, int>::type = 0>
void
initialize(CompList<CompTypes...>&, const std::string&, const std::string&)
{}

//--------------------------------------------------------------------------------------//

template <typename T, typename... Args>
void
initialize(T* obj, Args&&... args)
{
    if(obj)
        initialize(*obj, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

template <size_t Idx, typename Type, template <size_t, typename> class Bundle>
void
insert(Bundle<Idx, Type>& obj, const std::string& env_var, const std::string& default_env)
{
    auto env_result = get_env(env_var, default_env);
    tim::insert(obj, enumerate_components(delimit(env_result)));
}

//--------------------------------------------------------------------------------------//

template <typename T, typename... Args>
void
insert(T* obj, Args&&... args)
{
    if(obj)
        insert(*obj, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

template <typename Bundle, typename... Args>
void
configure(const std::string& env_var, const std::string& default_env, Args&&... args)
{
    auto env_result = get_env(env_var, default_env);
    tim::configure<Bundle>(enumerate_components(delimit(env_result)),
                           std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//

}  // namespace env

}  // namespace tim
