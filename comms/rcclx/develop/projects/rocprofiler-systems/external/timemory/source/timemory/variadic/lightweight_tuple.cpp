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

#ifndef TIMEMORY_VARIADIC_LIGHTWEIGHT_TUPLE_CPP_
#define TIMEMORY_VARIADIC_LIGHTWEIGHT_TUPLE_CPP_

#include "timemory/variadic/lightweight_tuple.hpp"

#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types/set.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/types.hpp"

//======================================================================================//
//
//      tim::get functions
//
namespace tim
{
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>::lightweight_tuple()
{
    bundle_type::init(type_list_type{}, *this, m_data, get_initializer());
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename FuncT, enable_if_t<std::is_invocable<FuncT>::value>>
lightweight_tuple<Types...>::lightweight_tuple(FuncT&& _func)
{
    bundle_type::init(type_list_type{}, *this, m_data, std::move(_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... T>
lightweight_tuple<Types...>::lightweight_tuple(const string_t&     _key,
                                               quirk::config<T...> _config,
                                               transient_func_t    _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, false_type{}, _config))
, m_data(invoke::construct<data_type>(_key, _config))
{
    bundle_type::init(type_list_type{}, *this, m_data, std::move(_init_func), _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... T>
lightweight_tuple<Types...>::lightweight_tuple(const captured_location_t& _loc,
                                               quirk::config<T...>        _config,
                                               transient_func_t           _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _loc, false_type{}, _config))
, m_data(invoke::construct<data_type>(_loc, _config))
{
    bundle_type::init(type_list_type{}, *this, m_data, std::move(_init_func), _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... T>
lightweight_tuple<Types...>::lightweight_tuple(size_t _hash, quirk::config<T...> _config,
                                               transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _hash, false_type{}, _config))
, m_data(invoke::construct<data_type>(_hash, _config))
{
    bundle_type::init(type_list_type{}, *this, m_data, std::move(_init_func), _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>::lightweight_tuple(size_t _hash, scope::config _scope,
                                               transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _hash, false_type{}, _scope))
, m_data(invoke::construct<data_type>(_hash, m_scope))
{
    bundle_type::init(type_list_type{}, *this, m_data, std::move(_init_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>::lightweight_tuple(const string_t& _key, scope::config _scope,
                                               transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, false_type{}, _scope))
{
    bundle_type::init(type_list_type{}, *this, m_data, std::move(_init_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>::lightweight_tuple(const captured_location_t& loc,
                                               scope::config              _scope,
                                               transient_func_t           _init_func)
: lightweight_tuple(loc.get_hash(), _scope, std::move(_init_func))
{}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>::~lightweight_tuple()
{
    IF_CONSTEXPR(!quirk_config<quirk::explicit_stop>::value)
    {
        if(m_is_active())
            stop();
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>
lightweight_tuple<Types...>::clone(bool _store, scope::config _scope)
{
    lightweight_tuple tmp(*this);
    tmp.m_store(_store);
    tmp.m_scope = _scope;
    return tmp;
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename... Types>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::push(int64_t _tid)
{
    if(!m_is_pushed())
    {
        // reset the data
        invoke::reset(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(true);
        // insert node or find existing node
        invoke::push(m_data, m_scope, m_hash, _tid);
    }
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename... Types>
template <typename... Tp>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::push(mpl::piecewise_select<Tp...>, int64_t _tid)
{
    using pw_type = mpl::implemented_t<Tp...>;
    // reset the data
    invoke_piecewise<operation::reset>(pw_type{});
    // insert node or find existing node
    invoke_piecewise<operation::push_node>(pw_type{}, m_scope, m_hash, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename... Types>
template <typename... Tp>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::push(mpl::piecewise_ignore<Tp...>, int64_t _tid)
{
    using pw_type = mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>;
    // reset the data
    invoke_piecewise<operation::reset>(pw_type{});
    // insert node or find existing node
    invoke_piecewise<operation::push_node>(pw_type{}, m_scope, m_hash, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename... Types>
template <typename... Tp>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::push(mpl::piecewise_select<Tp...>, scope::config _scope,
                                  int64_t _tid)
{
    using pw_type = mpl::implemented_t<Tp...>;
    // reset the data
    invoke_piecewise<operation::reset>(pw_type{});
    // insert node or find existing node
    invoke_piecewise<operation::push_node>(pw_type{}, _scope, m_hash, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename... Types>
template <typename... Tp>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::push(mpl::piecewise_ignore<Tp...>, scope::config _scope,
                                  int64_t _tid)
{
    using pw_type = mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>;
    // reset the data
    invoke_piecewise<operation::reset>(pw_type{});
    // insert node or find existing node
    invoke_piecewise<operation::push_node>(pw_type{}, _scope, m_hash, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// pop out of graph
//
template <typename... Types>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::pop(int64_t _tid)
{
    if(m_is_pushed())
    {
        // set the current node to the parent node
        invoke::pop(m_data, _tid);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(false);
    }
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// pop out of graph
//
template <typename... Types>
template <typename... Tp>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::pop(mpl::piecewise_select<Tp...>, int64_t _tid)
{
    using pw_type = mpl::implemented_t<Tp...>;
    // set the current node to the parent node
    invoke_piecewise<operation::pop_node>(pw_type{}, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// pop out of graph
//
template <typename... Types>
template <typename... Tp>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::pop(mpl::piecewise_ignore<Tp...>, int64_t _tid)
{
    using pw_type = mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>;
    // set the current node to the parent node
    invoke_piecewise<operation::pop_node>(pw_type{}, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// measure functions
//
template <typename... Types>
template <typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::measure(Args&&... args)
{
    invoke::measure(m_data, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// sample functions
//
template <typename... Types>
template <typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::sample(Args&&... args)
{
    invoke::invoke<operation::sample, TIMEMORY_API>(m_data, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename... Types>
template <typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::start(Args&&... args)
{
    invoke::start(m_data, std::forward<Args>(args)...);
    m_is_active(true);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename... Types>
template <typename... Tp, typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::start(mpl::piecewise_select<Tp...>, Args&&... args)
{
    auto&& _data = mpl::get_reference_tuple<mpl::available_t<std::tuple<Tp...>>>(m_data);
    invoke::start(_data, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename... Types>
template <typename... Tp, typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::start(mpl::piecewise_ignore<Tp...>, Args&&... args)
{
    using selected_t =
        convert_t<mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>,
                  std::tuple<>>;
    auto&& _data = mpl::get_reference_tuple<selected_t>(m_data);
    invoke::start(_data, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::stop(Args&&... args)
{
    invoke::stop(m_data, std::forward<Args>(args)...);
    ++m_laps;
    m_is_active(false);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename... Types>
template <typename... Tp, typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::stop(mpl::piecewise_select<Tp...>, Args&&... args)
{
    auto&& _data = mpl::get_reference_tuple<mpl::available_t<std::tuple<Tp...>>>(m_data);
    invoke::stop(_data, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename... Types>
template <typename... Tp, typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::stop(mpl::piecewise_ignore<Tp...>, Args&&... args)
{
    using selected_t =
        convert_t<mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>,
                  std::tuple<>>;
    auto&& _data = mpl::get_reference_tuple<selected_t>(m_data);
    invoke::stop(_data, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// recording
//
template <typename... Types>
template <typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::record(Args&&... args)
{
    ++m_laps;
    invoke::record(m_data, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// reset data
//
template <typename... Types>
template <typename... Args>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::reset(Args&&... args)
{
    invoke::reset(m_data, std::forward<Args>(args)...);
    m_laps = 0;
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// get data
//
template <typename... Types>
template <typename... Args>
auto
lightweight_tuple<Types...>::get(Args&&... args) const
{
    return invoke::get(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// reset data
//
template <typename... Types>
template <typename... Args>
auto
lightweight_tuple<Types...>::get_labeled(Args&&... args) const
{
    return invoke::get_labeled(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// this_type operators
//
template <typename... Types>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::operator-=(const this_type& rhs)
{
    invoke::invoke_impl::invoke_data<operation::minus, TIMEMORY_API>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::operator-=(this_type& rhs)
{
    invoke::invoke_impl::invoke_data<operation::minus, TIMEMORY_API>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::operator+=(const this_type& rhs)
{
    invoke::invoke_impl::invoke_data<operation::plus, TIMEMORY_API>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::operator+=(this_type& rhs)
{
    invoke::invoke_impl::invoke_data<operation::plus, TIMEMORY_API>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
typename lightweight_tuple<Types...>::data_type&
lightweight_tuple<Types...>::data()
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
const typename lightweight_tuple<Types...>::data_type&
lightweight_tuple<Types...>::data() const
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
lightweight_tuple<Types...>::set_prefix(const string_t& _key) const
{
    invoke::set_prefix(m_data, m_hash, _key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
lightweight_tuple<Types...>::set_prefix(size_t _hash) const
{
    auto itr = get_hash_ids()->find(_hash);
    if(itr != get_hash_ids()->end())
        invoke::set_prefix(m_data, _hash, itr->second);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
lightweight_tuple<Types...>&
lightweight_tuple<Types...>::set_scope(scope::config val)
{
    m_scope = val;
    invoke::set_scope(m_data, m_scope);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
lightweight_tuple<Types...>::init_storage()
{
    static thread_local bool _once = []() {
        apply_v::type_access<operation::init_storage, data_type>();
        return true;
    }();
    consume_parameters(_once);
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

#endif
