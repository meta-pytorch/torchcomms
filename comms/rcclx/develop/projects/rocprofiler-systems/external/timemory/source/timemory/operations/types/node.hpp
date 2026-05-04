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
 * \file timemory/operations/types/node.hpp
 * \brief Definition for various functions for pushing and popping storage nodes
 */

#pragma once

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/add_secondary.hpp"
#include "timemory/operations/types/add_statistics.hpp"
#include "timemory/operations/types/math.hpp"

#if !defined(NDEBUG)
#    include "timemory/settings/settings.hpp"
#endif

#include <type_traits>

namespace tim
{
namespace operation
{
//
template <typename Up>
static constexpr bool
storage_is_nullptr_t()
{
    return std::is_same<
        decay_t<decltype(operation::get_storage<Up>{}(std::declval<Up&>()))>,
        std::nullptr_t>::value;
}
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename ValT>
inline auto
get_node_hash(
    ValT&& _v,
    enable_if_t<std::is_integral<std::remove_cv_t<decay_t<ValT>>>::value, int> = 0)
{
    return _v;
}

template <typename ValT>
inline auto
get_node_hash(
    ValT&& _v,
    enable_if_t<concepts::is_string_type<std::remove_cv_t<decay_t<ValT>>>::value, int> =
        0)
{
    return get_hash_id(std::forward<ValT>(_v));
}
//
template <typename Tp>
struct push_node
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(push_node)

    TIMEMORY_INLINE push_node(type& obj, scope::config _scope, hash_value_t _hash,
                              int64_t _tid = threading::get_id())
    {
        (*this)(obj, _scope, _hash, _tid);
    }

    TIMEMORY_INLINE push_node(type& obj, scope::config _scope, string_view_cref_t _key,
                              int64_t _tid = threading::get_id())
    {
        (*this)(obj, _scope, get_hash_id(_key), _tid);
    }

    TIMEMORY_INLINE auto operator()(type& obj, scope::config _scope, hash_value_t _hash,
                                    int64_t _tid = threading::get_id()) const
    {
        init_storage<Tp>::init();
        return sfinae(obj, 0, 0, 0, _scope, _hash, _tid);
    }

    TIMEMORY_INLINE auto operator()(type& obj, scope::config _scope,
                                    string_view_cref_t _key,
                                    int64_t            _tid = threading::get_id()) const
    {
        return (*this)(obj, _scope, get_hash_id(_key), _tid);
    }

    //----------------------------------------------------------------------------------//
    //      explicitly provided storage
    //----------------------------------------------------------------------------------//
    template <typename HashT, typename StorageT>
    TIMEMORY_INLINE push_node(
        type& obj, scope::config _scope, HashT&& _hash, StorageT* _storage,
        int64_t _tid = threading::get_id(),
        enable_if_t<std::is_base_of<base::storage, StorageT>::value, int> = 0)
    {
        (*this)(obj, _scope, get_node_hash(std::forward<HashT>(_hash)), _storage, _tid);
    }

    template <typename HashT, typename StorageT = void>
    TIMEMORY_INLINE auto operator()(type& obj, scope::config _scope, HashT&& _hash,
                                    StorageT* _storage,
                                    int64_t   _tid = threading::get_id()) const
    {
        init_storage<Tp>::init();
        return sfinae(obj, 0, 0, 0, _scope, get_node_hash(std::forward<HashT>(_hash)),
                      _storage, _tid);
    }

private:
    //  typical resolution: variadic bundle of components, component
    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto sfinae(Up& obj, int, int, int, Args&&... args) const
        -> decltype(obj.push(std::forward<Args>(args)...))
    {
        return obj.push(std::forward<Args>(args)...);
    }

    //  typical resolution: variadic bundle of components, component
    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto sfinae(Up& obj, int, int, long, Args&&...) const
        -> decltype(obj.push())
    {
        return obj.push();
    }

    //  typical resolution: component inheriting from component::base<T, V>
    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    TIMEMORY_INLINE auto sfinae(Up& _obj, int, long, long, scope::config _scope,
                                hash_value_t _hash, StorageT* _storage, int64_t _tid,
                                enable_if_t<!storage_is_nullptr_t<Up>(), int> = 0) const
        -> decltype(_obj.get_iterator())
    {
        using storage_type          = StorageT;
        constexpr bool force_flat_v = trait::flat_storage<Tp>::value;
        constexpr bool force_time_v = trait::timeline_storage<Tp>::value;
        if(!operation::get_is_on_stack<type, false>{}(_obj))
        {
            // reset some state
            operation::set_is_on_stack<type>{}(_obj, true);
            operation::set_is_flat<type>{}(_obj, _scope.is_flat() || force_flat_v);

            if(_storage == nullptr)
            {
                _storage = operation::get_storage<type>{}(_obj);

                // set the storage pointer
                if(threading::get_id() == _tid)
                    operation::set_storage<type>{}(_storage, _tid);
            }

            // if storage is a nullptr, iterator is stale
            if(_storage == nullptr)
                return nullptr;

            // get the current depth
            auto _beg_depth = operation::get_depth<storage_type>{}(*_storage);
            // check against max depth when not flat
            if(!operation::get_is_flat<type, false>{}(_obj))
            {
                auto* _settings = settings::instance();
                if(_settings && _beg_depth + 1 > _settings->get_max_depth())
                {
                    operation::set_is_on_stack<type>{}(_obj, false);
                    return nullptr;
                }
            }
            // assign iterator to the insertion point in storage
            operation::set_iterator<type>{}(
                _obj,
                operation::insert<storage_type>{}(*_storage, _scope, _obj, _hash, _tid));
            // get the new depth
            auto _end_depth = operation::get_depth<storage_type>{}(*_storage);
            // configure the depth change state
            operation::set_depth_change<type>{}(
                _obj,
                (_beg_depth < _end_depth) || (_scope.is_timeline() || force_time_v));
            // add to stack
            operation::stack_push<storage_type>{}(*_storage, &_obj);
        }
        return _obj.get_iterator();
    }

    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    TIMEMORY_INLINE auto sfinae(Up& _obj, int, long, long, scope::config _scope,
                                hash_value_t _hash, int64_t _tid,
                                enable_if_t<!storage_is_nullptr_t<Up>(), int> = 0) const
        -> decltype(sfinae(_obj, 0, 0, 0, _scope, _hash, std::declval<StorageT*>(), _tid))
    {
        StorageT* _storage = nullptr;
        return sfinae(_obj, 0, 0, 0, _scope, _hash, _storage, _tid);
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    TIMEMORY_INLINE void sfinae(Up&, long, long, long, Args&&...) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct pop_node
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(pop_node)

    TIMEMORY_INLINE explicit pop_node(type& obj, int64_t _tid = threading::get_id())
    {
        (*this)(obj, _tid);
    }

    template <typename Arg, typename... Args, enable_if_t<(sizeof...(Args) > 0), int> = 0>
    TIMEMORY_INLINE explicit pop_node(type& obj, Arg&& arg, Args&&... args)
    {
        (*this)(obj, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    TIMEMORY_INLINE auto operator()(type& obj, int64_t _tid = threading::get_id()) const
    {
        return sfinae(obj, 0, 0, 0, _tid);
    }

    // call this overload if tid + extra args
    template <typename Arg, typename... Args,
              enable_if_t<(sizeof...(Args) > 0 && std::is_integral<decay_t<Arg>>::value),
                          int> = 0>
    TIMEMORY_INLINE auto operator()(type& obj, Arg&& arg, Args&&... args) const
    {
        return sfinae(obj, 0, 0, 0, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    // call this overload if tid is not provided
    template <typename Arg, typename... Args,
              enable_if_t<!std::is_integral<decay_t<Arg>>::value, int> = 0,
              enable_if_t<!std::is_base_of<base::storage,
                                           std::remove_pointer_t<decay_t<Arg>>>::value,
                          int>                                         = 0>
    TIMEMORY_INLINE auto operator()(type& obj, Arg&& arg, Args&&... args) const
    {
        return sfinae(obj, 0, 0, 0, threading::get_id(), std::forward<Arg>(arg),
                      std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //      explicitly provided storage
    //----------------------------------------------------------------------------------//
    template <typename StorageT, typename... Args,
              enable_if_t<std::is_base_of<base::storage, StorageT>::value, int> = 0>
    TIMEMORY_INLINE pop_node(type& obj, StorageT* _storage,
                             int64_t _tid = threading::get_id(), Args&&... args)
    {
        (*this)(obj, _storage, _tid, std::forward<Args>(args)...);
    }

    template <typename StorageT = void, typename... Args,
              enable_if_t<std::is_base_of<base::storage, StorageT>::value, int> = 0>
    TIMEMORY_INLINE auto operator()(type& obj, StorageT* _storage,
                                    int64_t _tid = threading::get_id(),
                                    Args&&... args) const
    {
        return sfinae(obj, 0, 0, 0, _storage, _tid, std::forward<Args>(args)...);
    }

private:
    //  typical resolution: variadic bundle of components, component
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, int, Args&&... args) const
        -> decltype(obj.pop(std::forward<Args>(args)...))
    {
        return obj.pop(std::forward<Args>(args)...);
    }

    //  typical resolution: variadic bundle of components, component
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, long, Args&&...) const -> decltype(obj.pop())
    {
        return obj.pop();
    }

    //  typical resolution: component inheriting from component::base<T, V>
    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    TIMEMORY_HOT auto sfinae(Up& _obj, int, long, long, StorageT* _storage, int64_t _tid,
                             enable_if_t<!storage_is_nullptr_t<Up>(), int> = 0) const
        -> decltype(_obj.get_iterator())
    {
        using storage_type = StorageT;

        // return
        if(operation::get_is_invalid<Up, false>{}(_obj))
            return nullptr;

        if(operation::get_is_on_stack<type, true>{}(_obj) && _obj.get_iterator())
        {
            if(_storage == nullptr)
                _storage = operation::get_storage<type>{}(_obj, _tid);

            // if storage is null, iterator is stale which means targ and stats are too
            if(_storage == nullptr)
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE(
                    settings::debug() && settings::verbose() >= 1,
                    "storage for thread %li was deleted for component of type %s while "
                    "it was still on the stack",
                    _tid, demangle<Tp>().c_str());
                return nullptr;
            }

            operation::set_is_on_stack<type>{}(_obj, false);
            auto&& itr   = _obj.get_iterator();
            type&  targ  = itr->data();
            auto&  stats = itr->stats();

            if(settings::debug() && settings::verbose() >= 5)
            {
                TIMEMORY_PRINTF(stderr, "\n");
                TIMEMORY_PRINTF(stderr, "[START][TARG][%i][%li]> %s\n", process::get_id(),
                                threading::get_id(), TIMEMORY_JOIN("", targ).data());
                TIMEMORY_PRINTF(stderr, "[START][DATA][%i][%li]> %s\n", process::get_id(),
                                threading::get_id(), TIMEMORY_JOIN("", _obj).data());
            }

            // reset depth change and set valid state on target
            operation::set_depth_change<type>{}(_obj, false);
            operation::set_is_invalid<type>{}(targ, false);
            // add measurement to target in storage
            operation::plus<type>(targ, _obj);
            //
            if(settings::debug() && settings::verbose() >= 5)
            {
                TIMEMORY_PRINTF(stderr, "[AFTER][TARG][%i][%li]> %s\n", process::get_id(),
                                threading::get_id(), TIMEMORY_JOIN("", targ).data());
            }
            // add the secondary data
            operation::add_secondary<type>(_storage, itr, _obj);
            // update the statistics
            operation::add_statistics<type>(_obj, stats);
            // if not finalizing manipulate call-stack hierarchy
            if(!storage_type::is_finalizing())
            {
                if(operation::get_is_flat<type, false>{}(_obj))
                {
                    // just pop if flat
                    operation::stack_pop<storage_type>{}(*_storage, &_obj);
                }
                else
                {
                    // get depth and then pop
                    auto _beg_depth = operation::get_depth<storage_type>{}(_storage);
                    operation::pop<storage_type>{}(*_storage);
                    operation::stack_pop<storage_type>{}(*_storage, &_obj);
                    // get depth and determine if depth change occurred
                    auto _end_depth = operation::get_depth<storage_type>{}(_storage);
                    operation::set_depth_change<type>{}(_obj, _beg_depth > _end_depth);
                }
            }
            operation::set_is_running<type>{}(targ, false);
        }
        return _obj.get_iterator();
    }

    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    TIMEMORY_HOT auto sfinae(Up& _obj, int, long, long, int64_t _tid,
                             enable_if_t<!storage_is_nullptr_t<Up>(), int> = 0) const
        -> decltype(sfinae(_obj, 0, 0, 0, std::declval<StorageT*>(), _tid))
    {
        StorageT* _storage = nullptr;
        return sfinae(_obj, 0, 0, 0, _storage, _tid);
    }

    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    TIMEMORY_HOT auto sfinae(Up& _obj, int, long, long, int64_t _tid, StorageT* _storage,
                             enable_if_t<!storage_is_nullptr_t<Up>(), int> = 0) const
        -> decltype(sfinae(_obj, 0, 0, 0, _storage, _tid))
    {
        return sfinae(_obj, 0, 0, 0, _storage, _tid);
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, long, Args&&...) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
