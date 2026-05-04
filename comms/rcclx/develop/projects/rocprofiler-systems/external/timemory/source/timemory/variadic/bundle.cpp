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

#ifndef TIMEMORY_VARIADIC_BUNDLE_CPP_
#define TIMEMORY_VARIADIC_BUNDLE_CPP_

#if !defined(TIMEMORY_VARIADIC_BUNDLE_HPP_)
#    include "timemory/variadic/bundle.hpp"
#endif

#include "timemory/backends/dmp.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types/set.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/bundle_execute.hpp"
#include "timemory/variadic/functional.hpp"
#include "timemory/variadic/types.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::initializer_type&
bundle<Tag, BundleT, TupleT>::get_initializer()
{
    static initializer_type _instance = [](this_type&) {};
    return _instance;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::bundle()
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, get_initializer());
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... T>
bundle<Tag, BundleT, TupleT>::bundle(const string_t& _key, quirk::config<T...> _config,
                                     transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, true_type{}, _config))
, m_data(invoke::construct<data_type, Tag>(_key, _config))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle, T...>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func),
                      _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... T>
bundle<Tag, BundleT, TupleT>::bundle(hash_value_t _hash, quirk::config<T...> _config,
                                     transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _hash, true_type{}, _config))
, m_data(invoke::construct<data_type, Tag>(_hash, _config))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle, T...>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func),
                      _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... T>
bundle<Tag, BundleT, TupleT>::bundle(const captured_location_t& _loc,
                                     quirk::config<T...>        _config,
                                     transient_func_t           _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _loc, true_type{}, _config))
, m_data(invoke::construct<data_type, Tag>(_loc, _config))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle, T...>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func),
                      _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... T>
bundle<Tag, BundleT, TupleT>::bundle(const string_t& _key, bool _store,
                                     quirk::config<T...> _config,
                                     transient_func_t    _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, _store, _config))
, m_data(invoke::construct<data_type, Tag>(_key, _config))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle, T...>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func),
                      _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... T>
bundle<Tag, BundleT, TupleT>::bundle(const captured_location_t& _loc, bool _store,
                                     quirk::config<T...> _config,
                                     transient_func_t    _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _loc, _store, _config))
, m_data(invoke::construct<data_type, Tag>(_loc, _config))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle, T...>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func),
                      _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::bundle(hash_value_t _hash, bool _store,
                                     scope::config _scope, transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _hash, _store, _scope))
, m_data(invoke::construct<data_type, Tag>(_hash, m_scope))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::bundle(const string_t& _key, bool _store,
                                     scope::config _scope, transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, _store, _scope))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::bundle(const captured_location_t& _loc, bool _store,
                                     scope::config _scope, transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _loc, _store, _scope))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::bundle(hash_value_t _hash, scope::config _scope,
                                     transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _hash, true_type{}, _scope))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::bundle(const string_t& _key, scope::config _scope,
                                     transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, true_type{}, _scope))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::bundle(const captured_location_t& _loc,
                                     scope::config _scope, transient_func_t _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _loc, true_type{}, _scope))
{
    update_last_instance(&get_this_type(), get_last_instance(),
                         quirk_config<quirk::stop_last_bundle>::value);
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    bundle_type::init(type_list_type{}, get_this_type(), m_data, std::move(_init_func));
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::~bundle()
{
    if(get_last_instance() == &get_this_type())
        update_last_instance(nullptr, get_last_instance(), false);

    IF_CONSTEXPR(!quirk_config<quirk::explicit_stop>::value)
    {
        if(m_is_active())
            stop();
    }

    IF_CONSTEXPR(optional_count() > 0)
    {
#if defined(DEBUG) && !defined(NDEBUG)
        if(tim::settings::debug() && tim::settings::verbose() > 4)
        {
            TIMEMORY_PRINT_HERE("%s", "deleting components");
        }
#endif
        invoke::destroy<Tag>(m_data);
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>::bundle(const bundle& rhs)
: bundle_type(rhs)
{
    using copy_oper_t = convert_each_t<operation::copy, remove_pointers_t<data_type>>;
    IF_CONSTEXPR(optional_count() > 0) { apply_v::set_value(m_data, nullptr); }
    apply_v::access2<copy_oper_t>(m_data, rhs.m_data);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>&
bundle<Tag, BundleT, TupleT>::operator=(const bundle& rhs)
{
    if(this != &rhs)
    {
        bundle_type::operator=(rhs);
        invoke::destroy<Tag>(m_data);
        invoke::invoke_impl::invoke_data<operation::copy, Tag>(m_data, rhs.m_data);
        // apply_v::access<operation_t<operation::copy>>(m_data);
    }
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// this_type operators
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::operator-=(const this_type& rhs)
{
    bundle_type::operator-=(static_cast<const bundle_type&>(rhs));
    invoke::invoke_impl::invoke_data<operation::minus, Tag>(m_data, rhs.m_data);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::operator+=(const this_type& rhs)
{
    bundle_type::operator+=(static_cast<const bundle_type&>(rhs));
    invoke::invoke_impl::invoke_data<operation::plus, Tag>(m_data, rhs.m_data);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
bundle<Tag, BundleT, TupleT>
bundle<Tag, BundleT, TupleT>::clone(bool _store, scope::config _scope)
{
    bundle tmp(*this);
    tmp.m_store(_store);
    tmp.m_scope = _scope;
    return tmp;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
void
bundle<Tag, BundleT, TupleT>::init_storage()
{
    static thread_local bool _once = []() {
        apply_v::type_access<operation::init_storage, mpl::non_quirk_t<reference_type>>();
        return true;
    }();
    consume_parameters(_once);
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::push(int64_t _tid)
{
    if(!m_enabled())
        return get_this_type();

    if(!m_is_pushed())
    {
        // reset the data
        invoke::reset<Tag>(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(true);
        // insert node or find existing node
        invoke::push<Tag>(m_data, m_scope, m_hash, _tid);
    }
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::push(mpl::piecewise_select<Tp...>, int64_t _tid)
{
    if(!m_enabled())
        return get_this_type();

    using pw_type = convert_t<mpl::implemented_t<Tp...>, mpl::piecewise_select<>>;
    // reset the data
    invoke::invoke<operation::reset, Tag>(pw_type{}, m_data);
    // insert node or find existing node
    invoke::invoke<operation::push_node, Tag>(pw_type{}, m_data, m_scope, m_hash, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::push(mpl::piecewise_ignore<Tp...>, int64_t _tid)
{
    if(!m_enabled())
        return get_this_type();

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
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::push(mpl::piecewise_select<Tp...>, scope::config _scope,
                                   int64_t _tid)
{
    if(!m_enabled())
        return get_this_type();

    using pw_type = convert_t<mpl::implemented_t<Tp...>, mpl::piecewise_select<>>;
    // reset the data
    invoke::invoke<operation::reset, Tag>(pw_type{}, m_data);
    // insert node or find existing node
    invoke::invoke<operation::push_node, Tag>(pw_type{}, m_data, _scope, m_hash, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::push(mpl::piecewise_ignore<Tp...>, scope::config _scope,
                                   int64_t _tid)
{
    if(!m_enabled())
        return get_this_type();

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
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::pop(int64_t _tid)
{
    if(!m_enabled())
        return get_this_type();

    if(m_is_pushed())
    {
        // set the current node to the parent node
        invoke::pop<Tag>(m_data, _tid);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(false);
    }
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// pop out of graph
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::pop(mpl::piecewise_select<Tp...>, int64_t _tid)
{
    if(!m_enabled())
        return get_this_type();

    using pw_type = convert_t<mpl::implemented_t<Tp...>, mpl::piecewise_select<>>;
    // set the current node to the parent node
    invoke::invoke<operation::pop_node, Tag>(pw_type{}, m_data, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// pop out of graph
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::pop(mpl::piecewise_ignore<Tp...>, int64_t _tid)
{
    if(!m_enabled())
        return get_this_type();

    using pw_type = mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>;
    // set the current node to the parent node
    invoke_piecewise<operation::pop_node>(pw_type{}, _tid);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// measure functions
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::measure(Args&&... args)
{
    return invoke<operation::measure>(std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// sample functions
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::sample(Args&&... args)
{
    return invoke<operation::sample>(std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::start(mpl::lightweight, Args&&... args)
{
    if(!m_enabled())
        return get_this_type();

    assemble(*this);
    invoke::start<Tag>(m_data, std::forward<Args>(args)...);
    m_is_active(true);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::stop(mpl::lightweight, Args&&... args)
{
    if(!m_enabled())
        return get_this_type();

    invoke::stop<Tag>(m_data, std::forward<Args>(args)...);
    if(m_is_active())
        ++m_laps;
    derive(*this);
    m_is_active(false);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::start(mpl::piecewise_select<Tp...>, Args&&... args)
{
    if(!m_enabled())
        return get_this_type();

    using select_tuple_t = mpl::sort<trait::start_priority, std::tuple<Tp...>>;

    invoke_piecewise<operation::reset>(type_list<Tp...>{});
    IF_CONSTEXPR(!quirk_config<quirk::explicit_push>::value &&
                 !quirk_config<quirk::no_store>::value)
    {
        if(m_store() && !bundle_type::m_explicit_push())
        {
            invoke_piecewise<operation::push_node>(type_list<Tp...>{}, m_scope, m_hash);
        }
    }

    // start components
    auto&& _data = mpl::get_reference_tuple<select_tuple_t>(m_data);
    invoke::invoke<operation::standard_start, Tag>(_data, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::start(mpl::piecewise_ignore<Tp...>, Args&&... args)
{
    using selected_t =
        mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>;
    return start(convert_t<selected_t, mpl::piecewise_select<>>{},
                 std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::stop(mpl::piecewise_select<Tp...>, Args&&... args)
{
    if(!m_enabled())
        return get_this_type();

    using select_tuple_t = mpl::sort<trait::stop_priority, std::tuple<Tp...>>;

    // stop components
    auto&& _data = mpl::get_reference_tuple<select_tuple_t>(m_data);
    invoke::invoke<operation::standard_stop, Tag>(_data, std::forward<Args>(args)...);

    IF_CONSTEXPR(!quirk_config<quirk::explicit_pop>::value &&
                 !quirk_config<quirk::no_store>::value)
    {
        if(m_store() && !bundle_type::m_explicit_pop())
        {
            invoke_piecewise<operation::pop_node>(type_list<Tp...>{});
        }
    }
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tp, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::stop(mpl::piecewise_ignore<Tp...>, Args&&... args)
{
    using selected_t =
        mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>;
    return stop(convert_t<selected_t, mpl::piecewise_select<>>{},
                std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// start/stop functions
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::start(Args&&... args)
{
    if(!m_enabled())
        return get_this_type();

    // push components into the call-stack
    IF_CONSTEXPR(!quirk_config<quirk::explicit_push>::value &&
                 !quirk_config<quirk::no_store>::value)
    {
        if(m_store() && !bundle_type::m_explicit_push())
            push(tid());
    }

    // start components
    start(mpl::lightweight{}, std::forward<Args>(args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::stop(Args&&... args)
{
    if(!m_enabled())
        return get_this_type();

    // stop components
    stop(mpl::lightweight{}, std::forward<Args>(args)...);

    // pop components off of the call-stack stack
    IF_CONSTEXPR(!quirk_config<quirk::explicit_pop>::value &&
                 !quirk_config<quirk::no_store>::value)
    {
        if(m_store() && !bundle_type::m_explicit_pop())
            pop(tid());
    }
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
// recording
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::record(Args&&... args)
{
    if(!m_enabled())
        return get_this_type();

    ++m_laps;
    return invoke<operation::record>(std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// reset data
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::reset(Args&&... args)
{
    m_laps = 0;
    return invoke<operation::reset>(std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
uint64_t
bundle<Tag, BundleT, TupleT>::count()
{
    uint64_t _count = 0;
    invoke::invoke<operation::generic_counter>(m_data, std::ref(_count));
    return _count;
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::construct(Args&&... _args)
{
    // using construct_t = operation_t<operation::construct>;
    // apply_v::access<construct_t>(m_data, std::forward<Args>(_args)...);
    return invoke<operation::construct>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::assemble(Args&&... _args)
{
    return invoke<operation::assemble>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::derive(Args&&... _args)
{
    return invoke<operation::derive>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::mark(Args&&... _args)
{
    return invoke<operation::mark>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::mark_begin(Args&&... _args)
{
    return invoke<operation::mark_begin>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::mark_end(Args&&... _args)
{
    return invoke<operation::mark_end>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::store(Args&&... _args)
{
    if(!m_enabled())
        return get_this_type();

    m_is_active(true);
    invoke<operation::store>(std::forward<Args>(_args)...);
    m_is_active(false);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::audit(Args&&... _args)
{
    return invoke<operation::audit>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::add_secondary(Args&&... _args)
{
    return invoke<operation::add_secondary>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::update_statistics(Args&&... _args)
{
    return invoke<operation::add_statistics>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::set_data(Args&&... _args)
{
    return invoke<operation::set_data>(std::forward<Args>(_args)...);
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <template <typename> class OpT, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::invoke(Args&&... _args)
{
    if(!m_enabled())
        return get_this_type();

    invoke::invoke<OpT, Tag>(m_data, std::forward<Args>(_args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <template <typename> class OpT, typename... Tp, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::invoke(mpl::piecewise_select<Tp...>, Args&&... _args)
{
    if(!m_enabled())
        return get_this_type();
    invoke_piecewise<OpT>(mpl::available_t<type_list<Tp...>>{},
                          std::forward<Args>(_args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <template <typename> class OpT, typename... Tp, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::invoke(mpl::piecewise_ignore<Tp...>, Args&&... _args)
{
    if(!m_enabled())
        return get_this_type();
    invoke_piecewise<OpT>(
        mpl::subtract_t<mpl::available_t<type_list_type>, type_list<Tp...>>{},
        std::forward<Args>(_args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <template <typename> class OpT, typename... Tp, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::invoke_with(type_list<Tp...>&&, Args&&... _args)
{
    if(!m_enabled())
        return get_this_type();

    auto _data = std::make_tuple(this->get<Tp>()...);
    invoke::invoke<OpT, Tag>(_data, std::forward<Args>(_args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <template <typename> class OpT, size_t... Idx, typename... Args>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::invoke_with(std::index_sequence<Idx...>&&, Args&&... _args)
{
    if(!m_enabled())
        return get_this_type();

    auto _data = std::make_tuple(this->get<component::enumerator_t<Idx>>()...);
    invoke::invoke<OpT, Tag>(_data, std::forward<Args>(_args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <template <typename> class OpT, typename... Tp, typename... Args>
void
bundle<Tag, BundleT, TupleT>::invoke_piecewise(type_list<Tp...>, Args&&... _args)
{
    TIMEMORY_FOLD_EXPRESSION(operation::generic_operator<Tp, OpT<Tp>, Tag>(
        this->get<Tp>(), std::forward<Args>(_args)...));
}

//--------------------------------------------------------------------------------------//
// get data
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
auto
bundle<Tag, BundleT, TupleT>::get(Args&&... args) const
{
    return invoke::get<Tag>(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// get labeled data
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Args>
auto
bundle<Tag, BundleT, TupleT>::get_labeled(Args&&... args) const
{
    return invoke::get_labeled<Tag>(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::data_type&
bundle<Tag, BundleT, TupleT>::data()
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
const typename bundle<Tag, BundleT, TupleT>::data_type&
bundle<Tag, BundleT, TupleT>::data() const
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::get(void*& ptr, size_t _hash) const
{
    if(!m_enabled())
        return get_this_type();

    tim::variadic::impl::get<Tag>(m_data, ptr, _hash);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... Tail>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::disable()
{
    TIMEMORY_FOLD_EXPRESSION(operation::generic_deleter<remove_pointer_t<Tail>>{
        this->get_reference<Tail>() });
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename... T, typename... Args>
std::array<bool, sizeof...(T)>
bundle<Tag, BundleT, TupleT>::initialize(Args&&... args)
{
    if(!m_enabled())
        return std::array<bool, sizeof...(T)>{};

    constexpr auto N = sizeof...(T);
    return TIMEMORY_FOLD_EXPANSION(bool, N, init<T>(std::forward<Args>(args)...));
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename T, typename Func, typename... Args,
          enable_if_t<trait::is_available<T>::value, int>>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::type_apply(Func&& _func, Args&&... _args)
{
    if(!m_enabled())
        return get_this_type();

    auto* _obj = get<T>();
    if(_obj)
        ((*_obj).*(_func))(std::forward<Args>(_args)...);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename T, typename Func, typename... Args,
          enable_if_t<!trait::is_available<T>::value, int>>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::type_apply(Func&&, Args&&...)
{
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
void
bundle<Tag, BundleT, TupleT>::set_prefix(const string_t& _key) const
{
    if(!m_enabled())
        return;

    invoke::set_prefix<Tag>(m_data, m_hash, _key);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::set_prefix(hash_value_t _hash) const
{
    if(!m_enabled())
        return get_this_type();

    const auto& _hash_ids = hash::get_hash_ids();
    auto itr = hash::find_hash_identifier(_hash_ids, hash::get_hash_aliases(), _hash);
    if(itr != _hash_ids->end())
    {
        invoke::set_prefix<Tag>(m_data, _hash, itr->second);
    }
    else
    {
        const char* _prefix = nullptr;
        if(hash::get_hash_identifier(_hash, _prefix))
            invoke::set_prefix<Tag>(m_data, _hash, _prefix);
    }
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::set_prefix(captured_location_t _loc) const
{
    return set_prefix(_loc.get_hash());
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::set_scope(scope::config val)
{
    if(!m_enabled())
        return get_this_type();

    m_scope = val;
    invoke::set_scope<Tag>(m_data, m_scope);
    return get_this_type();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
scope::transient_destructor
bundle<Tag, BundleT, TupleT>::get_scope_destructor()
{
    return scope::transient_destructor{ [&]() { this->stop(); } };
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
scope::transient_destructor
bundle<Tag, BundleT, TupleT>::get_scope_destructor(
    utility::transient_function<void(this_type&)> _func)
{
    return scope::transient_destructor{ [&, _func]() { _func(get_this_type()); } };
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename T>
void
bundle<Tag, BundleT, TupleT>::set_prefix(T* obj, internal_tag) const
{
    if(!m_enabled())
        return;

    using PrefixOpT   = operation::generic_operator<T, operation::set_prefix<T>, Tag>;
    std::string* _key = nullptr;
    if(get_hash_identifier_fast(m_hash, _key))
        PrefixOpT{ obj, m_hash, *_key };
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename T>
void
bundle<Tag, BundleT, TupleT>::set_scope(T* obj, internal_tag) const
{
    if(!m_enabled())
        return;

    using PrefixOpT = operation::generic_operator<T, operation::set_scope<T>, Tag>;
    PrefixOpT(obj, m_scope);
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <bool PrintPrefix, bool PrintLaps>
typename bundle<Tag, BundleT, TupleT>::this_type&
bundle<Tag, BundleT, TupleT>::print(std::ostream& os, bool _endl) const
{
    using printer_t = typename bundle_type::print_type;
    if(size() == 0 || m_hash == 0)
        return get_this_type();
    std::stringstream ss_data;
    apply_v::access_with_indices<printer_t>(m_data, std::ref(ss_data), false);
    if(PrintPrefix)
    {
        bundle_type::update_width();
        std::stringstream ss_prefix;
        std::stringstream ss_id;
        ss_id << get_prefix() << " " << std::left << key();
        ss_prefix << std::setw(bundle_type::output_width()) << std::left << ss_id.str()
                  << " : ";
        os << ss_prefix.str();
    }
    std::string _s = ss_data.str();
    if(_s.empty())
        return get_this_type();
    while(!_s.empty() && _s.find_last_of(", ") == _s.length() - 1)
        _s = _s.substr(0, _s.length() - 1);
    if(_s.empty())
        return get_this_type();
    os << _s;
    if(m_laps > 0 && PrintLaps)
        os << " [laps: " << m_laps << "]";
    if(_endl)
        os << '\n';
    return get_this_type();
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename Archive>
void
bundle<Tag, BundleT, TupleT>::serialize(Archive& ar, const unsigned int)
{
    std::string _key   = {};
    auto        keyitr = get_hash_ids()->find(m_hash);
    if(keyitr != get_hash_ids()->end())
        _key = keyitr->second;

    ar(cereal::make_nvp("hash", m_hash), cereal::make_nvp("key", _key),
       cereal::make_nvp("laps", m_laps));

    if(keyitr == get_hash_ids()->end())
    {
        auto _hash = add_hash_id(_key);
        if(_hash != m_hash)
        {
            TIMEMORY_PRINT_HERE("Warning! Hash for '%s' (%llu) != %llu", _key.c_str(),
                                (unsigned long long) _hash, (unsigned long long) m_hash);
        }
    }

    ar.setNextName("data");
    ar.startNode();
    invoke::serialize(ar, m_data);
    ar.finishNode();
    // ar(cereal::make_nvp("data", m_data));
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

#endif
