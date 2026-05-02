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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "timemory/components/opaque/declaration.hpp"
#include "timemory/components/opaque/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/sample.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/variadic/functional.hpp"

#include <cassert>
#include <cstdint>
#include <functional>
#include <set>
#include <string>

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
namespace factory
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename... Args,
          enable_if_t<concepts::is_wrapper<Tp>::value, int> = 0>
static auto
create_heap_variadic(Args... args)
{
    return new Tp{ std::move(args)... };
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Label, typename... Args,
          enable_if_t<!concepts::is_wrapper<Tp>::value, int> = 0>
static auto
create_heap_variadic(Label&& _label, scope::config, Args&&... args)
{
    return new Tp{ std::forward<Label>(_label), std::forward<Args>(args)... };
}
//
//--------------------------------------------------------------------------------------//
//
namespace hidden
{
//
//--------------------------------------------------------------------------------------//
//
//  Configure the tool for a specific component
//
template <typename Toolset>
enable_if_t<!concepts::is_wrapper<Toolset>::value && trait::is_available<Toolset>::value,
            opaque>
get_opaque(scope::config _scope)
{
    opaque _obj = Toolset::get_opaque(_scope);

    // this is not in base-class impl
    _obj.m_init = []() { operation::init_storage<Toolset>{}; };

    return _obj;
}
//
//--------------------------------------------------------------------------------------//
//
//  Configure the tool for a specific set of tools
//
template <typename Toolset, typename... Args>
enable_if_t<concepts::is_wrapper<Toolset>::value, opaque>
get_opaque(scope::config _scope, Args... args)
{
    using Toolset_t = Toolset;

    if(Toolset::size() == 0)
    {
        TIMEMORY_DEBUG_PRINT_HERE("returning! %s is empty", demangle<Toolset>().c_str());
        return opaque{};
    }

    opaque _obj{};

    _obj.m_valid = true;

    _obj.m_typeid = typeid_hash<Toolset>();

    _obj.m_init = []() {};

    auto _setup = [=](void* v_result, string_view_cref_t _prefix,
                      scope::config _arg_scope) {
        TIMEMORY_DEBUG_PRINT_HERE("Setting up %s", demangle<Toolset>().c_str());
        Toolset_t* _result = static_cast<Toolset_t*>(v_result);
        if(_result == nullptr)
        {
            _result =
                create_heap_variadic<Toolset_t>(_prefix, _scope + _arg_scope, args...);
        }
        else
        {
            _result->rekey(_prefix);
        }
        return static_cast<void*>(_result);
    };

    _obj.m_setup = _setup;

    _obj.m_push = [_setup](void*& v_result, string_view_cref_t _prefix,
                           scope::config arg_scope) {
        v_result = _setup(v_result, _prefix, arg_scope);
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Pushing %s", demangle<Toolset>().c_str());
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->push();
        }
    };

    _obj.m_sample = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Sampling %s", demangle<Toolset_t>().c_str());
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->sample();
        }
    };

    _obj.m_start = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Starting %s", demangle<Toolset>().c_str());
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->start();
        }
    };

    _obj.m_stop = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Stopping %s", demangle<Toolset>().c_str());
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->stop();
        }
    };

    _obj.m_pop = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Popping %s", demangle<Toolset>().c_str());
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->pop();
        }
    };

    _obj.m_get = [](void* v_result, void*& ptr, size_t _hash) {
        if(v_result && !ptr)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Getting %s", demangle<Toolset>().c_str());
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->get(ptr, _hash);
        }
    };

    _obj.m_del = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Deleting %s", demangle<Toolset>().c_str());
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            delete _result;
        }
    };

    return _obj;
}
//
//--------------------------------------------------------------------------------------//
//
//  If a tool is not avail or has no contents return empty opaque
//
template <typename Toolset, typename... Args>
enable_if_t<!trait::is_available<Toolset>::value, opaque>
get_opaque(scope::config, Args&&...)
{
    return opaque{};
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, bool IsWrapper = concepts::is_wrapper<T>::value>
struct opaque_typeids;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct opaque_typeids<T, false>
{
    using result_type = std::set<size_t>;
    static_assert(!concepts::is_wrapper<T>::value,
                  "Internal timemory error! Type should not be a variadic wrapper");

    template <typename U = T>
    static auto get(enable_if_t<trait::is_available<U>::value, int> = 0)
    {
        return result_type({ typeid_hash<U>() });
    }

    template <typename U = T>
    static auto hash(enable_if_t<trait::is_available<U>::value, int> = 0)
    {
        return typeid_hash<U>();
    }

    template <typename U = T>
    static result_type get(enable_if_t<!trait::is_available<U>::value, long> = 0)
    {
        return result_type({ 0 });
    }

    template <typename U = T>
    static size_t hash(enable_if_t<!trait::is_available<U>::value, long> = 0)
    {
        return 0;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename... T>
struct opaque_typeids<TupleT<T...>, false>
{
    using result_type = std::set<size_t>;
    using this_type   = opaque_typeids<TupleT<T...>, false>;
    static_assert(!concepts::is_wrapper<TupleT<T...>>::value,
                  "Internal timemory error! Type should not be a variadic wrapper");

    template <typename U = TupleT<T...>>
    static result_type get(enable_if_t<trait::is_available<U>::value, int> = 0)
    {
        return result_type({ typeid_hash<U>() });
    }

    template <typename U = TupleT<T...>>
    static result_type get(enable_if_t<!trait::is_available<U>::value, long> = 0)
    {
        return result_type({ 0 });
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename... T>
struct opaque_typeids<TupleT<T...>, true>
{
    using result_type = std::set<size_t>;
    using this_type   = opaque_typeids<TupleT<T...>>;
    static_assert(concepts::is_wrapper<TupleT<T...>>::value,
                  "Internal timemory error! Type should be a variadic wrapper");

    template <typename U = TupleT<T...>>
    static result_type get(enable_if_t<trait::is_available<U>::value, int> = 0)
    {
        auto ret = result_type{ typeid_hash<U>() };
        TIMEMORY_FOLD_EXPRESSION(get<T>(ret));
        return ret;
    }

    template <typename U = TupleT<T...>>
    static result_type get(enable_if_t<!trait::is_available<U>::value, long> = 0)
    {
        return result_type({ 0 });
    }

private:
    template <typename U>
    static void get(result_type& ret, enable_if_t<trait::is_available<U>::value, int> = 0)
    {
        ret.insert(typeid_hash<U>());
    }

    template <typename U>
    static void get(result_type&, enable_if_t<!trait::is_available<U>::value, int> = 0)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace hidden
//
//--------------------------------------------------------------------------------------//
//
//  Generic handler
//
template <typename Toolset, typename Arg, typename... Args>
opaque
get_opaque(Arg&& arg, Args&&... args)
{
    return hidden::get_opaque<Toolset>(std::forward<Arg>(arg),
                                       std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
//  Default with no args
//
template <typename Toolset>
opaque
get_opaque()
{
    return hidden::get_opaque<Toolset>(tim::scope::get_default());
}
//
//--------------------------------------------------------------------------------------//
//
//  With scope arguments
//
template <typename Toolset>
opaque
get_opaque(scope::config _scope)
{
    return hidden::get_opaque<Toolset>(_scope);
}
//
//--------------------------------------------------------------------------------------//
//
//  Default with no args
//
template <typename Toolset>
std::set<size_t>
get_typeids()
{
    return hidden::opaque_typeids<Toolset>::get();
}
//
}  // namespace factory
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
