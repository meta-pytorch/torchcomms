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

#include "timemory/components.hpp"
#include "timemory/components/opaque/declaration.hpp"
#include "timemory/components/placeholder.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/macros/language.hpp"
#include "timemory/runtime/info.hpp"
#include "timemory/runtime/macros.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/variadic/definition.hpp"

#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace runtime
{
//
//--------------------------------------------------------------------------------------//
//
template <int I>
using make_int_sequence = std::make_integer_sequence<int, I>;
template <int... Ints>
using int_sequence             = std::integer_sequence<int, Ints...>;
using component_match_set_t    = std::set<std::string>;
using component_match_vector_t = std::vector<bool (*)(const char*)>;
using component_match_index_t  = std::vector<TIMEMORY_COMPONENT>;
using opaque_pair_t            = std::pair<component::opaque, std::set<size_t>>;
//
//--------------------------------------------------------------------------------------//
//
template <int I, int V, typename... Args>
enable_if_t<component::enumerator<I>::value && I != V, void>
do_enumerator_generate(std::vector<opaque_pair_t>& opaque_array, int idx, Args&&... args)
{
    using type = component::enumerator_t<I>;
    IF_CONSTEXPR(!concepts::is_placeholder<type>::value)
    {
        if(idx == I)
        {
            opaque_array.push_back(
                { component::factory::get_opaque<type>(std::forward<Args>(args)...),
                  component::factory::get_typeids<type>() });
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <int I, int V, typename... Args>
enable_if_t<!component::enumerator<I>::value || I == V, void>
do_enumerator_generate(std::vector<opaque_pair_t>&, int, Args&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
//                  The actual implementation of the function calls
//
//--------------------------------------------------------------------------------------//
//
template <int I, typename Tp, typename... Args>
enable_if_t<component::enumerator<I>::value, void>
do_enumerator_init(Tp& obj, int idx, Args&&... args)
{
    using type = component::enumerator_t<I>;
    IF_CONSTEXPR(!concepts::is_placeholder<type>::value &&
                 !std::is_same<decay_t<Tp>, type>::value)
    {
        if(idx == I)
            obj.template initialize<type>(std::forward<Args>(args)...);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <int I, typename Tp, typename... Args>
enable_if_t<!component::enumerator<I>::value, void>
do_enumerator_init(Tp&, int, Args&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
template <int I>
inline void
do_enumerator_enumerate()
{
    using type            = component::enumerator_t<I>;
    constexpr auto _is_ph = concepts::is_placeholder<type>::value;
    IF_CONSTEXPR(!_is_ph)
    {
        std::string _id = component::properties<type>::id();
        if(_id != "TIMEMORY_COMPONENTS_END")
        {
            using match_func_t = bool (*)(const char*);
            match_func_t _func = &component::properties<type>::matches;
            auto         itr   = runtime::info::find(I);
            if(itr != runtime::info::end())
            {
                // append the static info to identifier
                itr->second(_id, component::properties<type>::ids(), _func);
            }
            else
            {
                std::set<std::string> _ids{};
                for(auto&& iitr : component::properties<type>::ids())
                    _ids.insert(iitr);
                runtime::info::emplace(I, component::info{ I, _id, _ids, _func });
            }
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
//                  The fold expressions to call the actual functions
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, int... Ints, typename... Args>
void
enumerator_init(Tp& obj, int idx, int_sequence<Ints...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(
        do_enumerator_init<Ints>(obj, idx, std::forward<Args>(args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, int... Ints, typename... Args>
void
enumerator_insert(Tp& obj, int idx, int_sequence<Ints...>, Args&&... args)
{
    constexpr int              TpV = component::properties<Tp>::value;
    std::vector<opaque_pair_t> opaque_array{};
    TIMEMORY_FOLD_EXPRESSION(do_enumerator_generate<Ints, TpV>(
        opaque_array, idx, std::forward<Args>(args)...));
    for(auto&& itr : opaque_array)
        obj.insert(std::move(itr.first), std::move(itr.second));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, int... Ints, typename... Args>
void
enumerator_configure(int idx, int_sequence<Ints...>, Args&&... args)
{
    constexpr int              TpV = component::properties<Tp>::value;
    std::vector<opaque_pair_t> opaque_array{};
    TIMEMORY_FOLD_EXPRESSION(do_enumerator_generate<Ints, TpV>(
        opaque_array, idx, std::forward<Args>(args)...));
    for(auto&& itr : opaque_array)
        Tp::configure(std::move(itr.first), std::move(itr.second));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, int... Ints, typename... Args>
void
enumerator_configure(Tp& obj, int idx, int_sequence<Ints...>, Args&&... args)
{
    constexpr int              TpV = component::properties<Tp>::value;
    std::vector<opaque_pair_t> opaque_array{};
    TIMEMORY_FOLD_EXPRESSION(do_enumerator_generate<Ints, TpV>(
        opaque_array, idx, std::forward<Args>(args)...));
    for(auto&& itr : opaque_array)
        obj.configure(std::move(itr.first), std::move(itr.second));
}
//
//--------------------------------------------------------------------------------------//
//
template <int... Ints>
void enumerator_enumerate(int_sequence<Ints...>)
{
    TIMEMORY_FOLD_EXPRESSION(do_enumerator_enumerate<Ints>());
}
//
//--------------------------------------------------------------------------------------//
//
//                      The forward declared functions
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Arg, typename... Args>
void
initialize(Tp& obj, int idx, Arg&& arg, Args&&... args)
{
    enumerator_init(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{},
                    std::forward<Arg>(arg), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Arg, typename... Args>
void
insert(Tp& obj, int idx, Arg&& arg, Args&&... args)
{
    enumerator_insert(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{},
                      std::forward<Arg>(arg), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Arg, typename... Args>
void
configure(int idx, Arg&& arg, Args&&... args)
{
    enumerator_configure<Tp>(idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{},
                             std::forward<Arg>(arg), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Arg, typename... Args>
void
configure(Tp& obj, int idx, Arg&& arg, Args&&... args)
{
    enumerator_configure(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{},
                         std::forward<Arg>(arg), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
enumerate_info()
{
    static bool _once =
        (enumerator_enumerate(make_int_sequence<TIMEMORY_COMPONENTS_END>{}), true);
    (void) _once;
}
//
//--------------------------------------------------------------------------------------//
//
inline int
enumerate(tim::string_view_cref_t key)
{
    enumerate_info();

    auto&& itr = runtime::info::find(key);

    if(!itr.second && settings::verbose() >= 0)
        runtime::info::error_message(key);

    return itr.first.index;
}
//
//--------------------------------------------------------------------------------------//
//
inline component::info
get_info(tim::string_view_cref_t _v)
{
    enumerate_info();
    return runtime::info::find(_v).first;
}
//
//--------------------------------------------------------------------------------------//
//
inline component::info
get_info(int _idx)
{
    enumerate_info();
    auto&& itr = runtime::info::find(_idx);
    if(itr == runtime::info::end())
        return component::info{};
    return itr->second;
}
//
//--------------------------------------------------------------------------------------//
//
//                      Non-variadic specializations
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
initialize(Tp& obj, int idx)
{
    enumerator_init(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{});
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
insert(Tp& obj, int idx)
{
    enumerator_insert(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{});
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
configure(int idx)
{
    enumerator_configure<Tp>(idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{});
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
configure(Tp& obj, int idx)
{
    enumerator_configure(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{});
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
insert(Tp& obj, int idx, scope::config _scope)
{
    enumerator_insert(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{}, _scope);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
configure(int idx, scope::config _scope)
{
    enumerator_configure<Tp>(idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{}, _scope);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
configure(Tp& obj, int idx, scope::config _scope)
{
    enumerator_configure(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{}, _scope);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace runtime
}  // namespace tim
