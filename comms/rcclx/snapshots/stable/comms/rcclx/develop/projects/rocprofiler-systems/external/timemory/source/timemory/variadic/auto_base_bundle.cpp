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
//

#ifndef TIMEMORY_VARIADIC_AUTO_BASE_BUNDLE_CPP_
#define TIMEMORY_VARIADIC_AUTO_BASE_BUNDLE_CPP_

#include "timemory/variadic/auto_base_bundle.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle()
: m_enabled{ settings::enabled() }
, m_report_at_exit{ false }
, m_reference_object{ nullptr }
, m_temporary{}
{
    if(m_enabled)
    {
        internal_init(get_initializer());
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start>::value) { m_temporary.start(); }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
template <typename... T>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(string_view_cref_t  key,
                                                        quirk::config<T...> _config,
                                                        transient_func_t    init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(quirk_config<quirk::exit_report, T...>::value)
, m_reference_object(nullptr)
, m_temporary(key, m_enabled, _config)
{
    if(m_enabled)
    {
        internal_init(init_func);
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start, T...>::value)
        {
            m_temporary.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
template <typename... T>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(const captured_location_t& loc,
                                                        quirk::config<T...> _config,
                                                        transient_func_t    init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(quirk_config<quirk::exit_report, T...>::value)
, m_reference_object(nullptr)
, m_temporary(loc, m_enabled, _config)
{
    if(m_enabled)
    {
        internal_init(init_func);
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start, T...>::value)
        {
            m_temporary.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(string_view_cref_t key,
                                                        scope::config      _scope,
                                                        bool               report_at_exit,
                                                        transient_func_t   init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit || quirk_config<quirk::exit_report>::value)
, m_reference_object(nullptr)
, m_temporary(key, m_enabled, _scope)
{
    if(m_enabled)
    {
        internal_init(init_func);
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start>::value) { m_temporary.start(); }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(const captured_location_t& loc,
                                                        scope::config              _scope,
                                                        bool             report_at_exit,
                                                        transient_func_t init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit || quirk_config<quirk::exit_report>::value)
, m_reference_object(nullptr)
, m_temporary(loc, m_enabled, _scope)
{
    if(m_enabled)
    {
        internal_init(init_func);
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start>::value) { m_temporary.start(); }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(size_t hash, scope::config _scope,
                                                        bool             report_at_exit,
                                                        transient_func_t init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit || quirk_config<quirk::exit_report>::value)
, m_reference_object(nullptr)
, m_temporary(hash, m_enabled, _scope)
{
    if(m_enabled)
    {
        internal_init(init_func);
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start>::value) { m_temporary.start(); }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(component_type& tmp,
                                                        scope::config   _scope,
                                                        bool            report_at_exit)
: m_report_at_exit(report_at_exit || quirk_config<quirk::exit_report>::value)
, m_reference_object(&tmp)
, m_temporary(component_type(tmp.clone(true, _scope)))
{
    if(m_enabled)
    {
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start>::value) { m_temporary.start(); }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
template <typename Arg, typename... Args>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(string_view_cref_t key,
                                                        bool store, scope::config _scope,
                                                        transient_func_t init_func,
                                                        Arg&& arg, Args&&... args)
: m_enabled(store && settings::enabled())
, m_report_at_exit(settings::destructor_report() ||
                   quirk_config<quirk::exit_report>::value)
, m_reference_object(nullptr)
, m_temporary(key, m_enabled, _scope)
{
    if(m_enabled)
    {
        internal_init(init_func, std::forward<Arg>(arg), std::forward<Args>(args)...);
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start>::value) { m_temporary.start(); }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
template <typename Arg, typename... Args>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(const captured_location_t& loc,
                                                        bool store, scope::config _scope,
                                                        transient_func_t init_func,
                                                        Arg&& arg, Args&&... args)
: m_enabled(store && settings::enabled())
, m_report_at_exit(settings::destructor_report() ||
                   quirk_config<quirk::exit_report>::value)
, m_reference_object(nullptr)
, m_temporary(loc, m_enabled, _scope)
{
    if(m_enabled)
    {
        internal_init(init_func, std::forward<Arg>(arg), std::forward<Args>(args)...);
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start>::value) { m_temporary.start(); }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
template <typename Arg, typename... Args>
auto_base_bundle<Tag, CompT, BundleT>::auto_base_bundle(size_t hash, bool store,
                                                        scope::config    _scope,
                                                        transient_func_t init_func,
                                                        Arg&& arg, Args&&... args)
: m_enabled(store && settings::enabled())
, m_report_at_exit(settings::destructor_report() ||
                   quirk_config<quirk::exit_report>::value)
, m_reference_object(nullptr)
, m_temporary(hash, m_enabled, _scope)
{
    if(m_enabled)
    {
        internal_init(init_func, std::forward<Arg>(arg), std::forward<Args>(args)...);
        IF_CONSTEXPR(!quirk_config<quirk::explicit_start>::value) { m_temporary.start(); }
    }
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename CompT, typename BundleT>
auto_base_bundle<Tag, CompT, BundleT>::~auto_base_bundle()
{
    IF_CONSTEXPR(!quirk_config<quirk::explicit_stop>::value)
    {
        if(m_enabled)
        {
            // stop the timer
            m_temporary.stop();

            // report timer at exit
            if(m_report_at_exit)
            {
                std::stringstream ss;
                ss << m_temporary;
                if(ss.str().length() > 0)
                    std::cout << ss.str() << std::endl;
            }

            if(m_reference_object)
            {
                *m_reference_object += m_temporary;
            }
        }
    }
}

//
template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::push(Args&&... args)
{
    m_temporary.push(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::pop(Args&&... args)
{
    m_temporary.pop(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::measure(Args&&... args)
{
    m_temporary.measure(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::record(Args&&... args)
{
    m_temporary.record(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::sample(Args&&... args)
{
    m_temporary.sample(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::start(Args&&... args)
{
    m_temporary.start(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::stop(Args&&... args)
{
    m_temporary.stop(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::assemble(Args&&... args)
{
    m_temporary.assemble(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::derive(Args&&... args)
{
    m_temporary.derive(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::mark(Args&&... args)
{
    m_temporary.mark(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::mark_begin(Args&&... args)
{
    m_temporary.mark_begin(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::mark_end(Args&&... args)
{
    m_temporary.mark_end(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::store(Args&&... args)
{
    m_temporary.store(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::audit(Args&&... args)
{
    m_temporary.audit(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::add_secondary(Args&&... args)
{
    m_temporary.add_secondary(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::reset(Args&&... args)
{
    m_temporary.reset(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::set_scope(Args&&... args)
{
    m_temporary.set_scope(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::set_prefix(Args&&... args)
{
    m_temporary.set_prefix(std::forward<Args>(args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
template <template <typename> class OpT, typename... Args>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::invoke(Args&&... _args)
{
    m_temporary.template invoke<OpT>(std::forward<Args>(_args)...);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
bool
auto_base_bundle<Tag, CompT, BundleT>::enabled() const
{
    return m_enabled;
}

template <typename Tag, typename CompT, typename BundleT>
bool
auto_base_bundle<Tag, CompT, BundleT>::report_at_exit() const
{
    return m_report_at_exit;
}

template <typename Tag, typename CompT, typename BundleT>
bool
auto_base_bundle<Tag, CompT, BundleT>::store() const
{
    return m_temporary.store();
}

template <typename Tag, typename CompT, typename BundleT>
int64_t
auto_base_bundle<Tag, CompT, BundleT>::laps() const
{
    return m_temporary.laps();
}

template <typename Tag, typename CompT, typename BundleT>
uint64_t
auto_base_bundle<Tag, CompT, BundleT>::hash() const
{
    return m_temporary.hash();
}

template <typename Tag, typename CompT, typename BundleT>
std::string
auto_base_bundle<Tag, CompT, BundleT>::key() const
{
    return m_temporary.key();
}

template <typename Tag, typename CompT, typename BundleT>
typename auto_base_bundle<Tag, CompT, BundleT>::data_type&
auto_base_bundle<Tag, CompT, BundleT>::data()
{
    return m_temporary.data();
}

template <typename Tag, typename CompT, typename BundleT>
const typename auto_base_bundle<Tag, CompT, BundleT>::data_type&
auto_base_bundle<Tag, CompT, BundleT>::data() const
{
    return m_temporary.data();
}

template <typename Tag, typename CompT, typename BundleT>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::report_at_exit(bool val)
{
    m_report_at_exit = val;
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::rekey(string_view_cref_t _key)
{
    m_temporary.rekey(_key);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::rekey(captured_location_t _loc)
{
    m_temporary.rekey(_loc);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
BundleT&
auto_base_bundle<Tag, CompT, BundleT>::rekey(uint64_t _hash)
{
    m_temporary.rekey(_hash);
    return get_this_type();
}

template <typename Tag, typename CompT, typename BundleT>
scope::transient_destructor
auto_base_bundle<Tag, CompT, BundleT>::get_scope_destructor()
{
    return scope::transient_destructor{ [&]() { this->stop(); } };
}

template <typename Tag, typename CompT, typename BundleT>
scope::transient_destructor
auto_base_bundle<Tag, CompT, BundleT>::get_scope_destructor(
    utility::transient_function<void(this_type&)> _func)
{
    return scope::transient_destructor{ [&, _func]() {
        _func(static_cast<this_type&>(*this));
    } };
}

//
template <typename Tag, typename CompT, typename BundleT>
template <typename... Tail>
void
auto_base_bundle<Tag, CompT, BundleT>::disable()
{
    m_temporary.template disable<Tail...>();
}

template <typename Tag, typename CompT, typename BundleT>
void
auto_base_bundle<Tag, CompT, BundleT>::internal_init(transient_func_t _init)
{
    if(m_enabled)
        _init(static_cast<this_type&>(*this));
}

template <typename Tag, typename CompT, typename BundleT>
template <typename Arg, typename... Args>
void
auto_base_bundle<Tag, CompT, BundleT>::internal_init(transient_func_t _init, Arg&& _arg,
                                                     Args&&... _args)
{
    if(m_enabled)
    {
        _init(static_cast<this_type&>(*this));
        m_temporary.construct(std::forward<Arg>(_arg), std::forward<Args>(_args)...);
    }
}

}  // namespace tim

#endif
