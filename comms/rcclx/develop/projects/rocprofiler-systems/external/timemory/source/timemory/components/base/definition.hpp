//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include "timemory/components/base/declaration.hpp"
#include "timemory/components/base/types.hpp"
#include "timemory/components/metadata.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/demangle.hpp"

namespace tim
{
namespace component
{
//
//======================================================================================//
//
//                              NON-VOID BASE
//
//======================================================================================//
//
template <typename Tp, typename Value>
void
base<Tp, Value>::reset()
{
    laps_type::reset();
    base_state::reset();
    data_type::reset();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
void
base<Tp, Value>::get(void*& ptr, size_t _typeid_hash) const
{
    if(!ptr && _typeid_hash == typeid_hash<Tp>())
        ptr = reinterpret_cast<void*>(const_cast<base_type*>(this));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
void
base<Tp, Value>::set_started()
{
    set_is_running(true);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
void
base<Tp, Value>::set_stopped()
{
    if(get_is_running())
    {
        laps_type::increment_laps(1);
        set_is_transient(true);
        set_is_running(false);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
typename base<Tp, Value>::base_storage_type*
base<Tp, Value>::get_storage()
{
    return tim::base::storage::template base_instance<Tp, Value>();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
std::string
base<Tp, Value>::label()
{
    return metadata<Tp>::label();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
std::string
base<Tp, Value>::description()
{
    return metadata<Tp>::description();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
std::string
base<Tp, Value>::get_label()
{
    static std::string _instance = Type::label();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
std::string
base<Tp, Value>::get_description()
{
    static std::string _instance = Type::description();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
//              operator + - * /    (Type)
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp&
base<Tp, Value>::plus_oper(const Tp& rhs)
{
    data_type::plus(rhs);
    return static_cast<Type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp&
base<Tp, Value>::minus_oper(const Tp& rhs)
{
    data_type::minus(rhs);
    return static_cast<Type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp&
base<Tp, Value>::multiply_oper(const Tp& rhs)
{
    data_type::multiply(rhs);
    return static_cast<Type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp&
base<Tp, Value>::divide_oper(const Tp& rhs)
{
    data_type::divide(rhs);
    return static_cast<Type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
//              operator + - * /    (Value)
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp&
base<Tp, Value>::plus_oper(const Value& rhs)
{
    data_type::plus(rhs);
    return static_cast<Type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp&
base<Tp, Value>::minus_oper(const Value& rhs)
{
    data_type::minus(rhs);
    return static_cast<Type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp&
base<Tp, Value>::multiply_oper(const Value& rhs)
{
    data_type::minus(rhs);
    return static_cast<Type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp&
base<Tp, Value>::divide_oper(const Value& rhs)
{
    data_type::divide(rhs);
    return static_cast<Type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
//              operator + - * / <<
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp
operator+(const base<Tp, Value>& lhs, const base<Tp, Value>& rhs)
{
    return Tp(static_cast<const Tp&>(lhs)) += static_cast<const Tp&>(rhs);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp
operator-(const base<Tp, Value>& lhs, const base<Tp, Value>& rhs)
{
    return Tp(static_cast<const Tp&>(lhs)) -= static_cast<const Tp&>(rhs);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp operator*(const base<Tp, Value>& lhs, const base<Tp, Value>& rhs)
{
    return Tp(static_cast<const Tp&>(lhs)) *= static_cast<const Tp&>(rhs);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
Tp
operator/(const base<Tp, Value>& lhs, const base<Tp, Value>& rhs)
{
    return Tp(static_cast<const Tp&>(lhs)) /= static_cast<const Tp&>(rhs);
}
//
//======================================================================================//
//
//                              VOID BASE
//
//======================================================================================//
//
template <typename Tp>
void
base<Tp, void>::reset()
{
    base_state::reset();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
base<Tp, void>::set_started()
{
    set_is_running(true);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
base<Tp, void>::set_stopped()
{
    if(get_is_running())
        set_is_transient(true);
    set_is_running(false);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
base<Tp, void>::get(void*& ptr, size_t _typeid_hash) const
{
    if(!ptr && _typeid_hash == typeid_hash<Tp>())
        ptr = reinterpret_cast<void*>(const_cast<base_type*>(this));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
std::string
base<Tp, void>::label()
{
    return metadata<Tp>::label();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
std::string
base<Tp, void>::description()
{
    return metadata<Tp>::description();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
std::string
base<Tp, void>::get_label()
{
    static std::string _instance = Type::label();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
std::string
base<Tp, void>::get_description()
{
    static std::string _instance = Type::description();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim

#include "timemory/components/opaque/definition.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/get.hpp"
#include "timemory/operations/types/sample.hpp"
#include "timemory/variadic/functional.hpp"

namespace tim
{
namespace component
{
//
template <typename Tp, typename Value>
opaque
base<Tp, Value>::get_opaque(scope::config _scope)
{
    auto _typeid_hash = typeid_hash<Tp>();

    opaque _obj{};

    _obj.m_valid = true;

    _obj.m_typeid = _typeid_hash;

    _obj.m_setup = [](void* v_result, string_view_cref_t _prefix,
                      scope::config arg_scope) {
        TIMEMORY_DEBUG_PRINT_HERE("Setting up %s", demangle<Tp>().c_str());
        Tp* _result = static_cast<Tp*>(v_result);
        if(!_result)
            _result = new Tp{};
        invoke::set_prefix<TIMEMORY_API>(std::tie(*_result), _prefix);
        invoke::set_scope<TIMEMORY_API>(std::tie(*_result), arg_scope);
        return (void*) _result;
    };

    _obj.m_push = [_scope](void*& v_result, string_view_cref_t _prefix,
                           scope::config arg_scope) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Pushing %s", demangle<Tp>().c_str());
            auto _hash   = add_hash_id(_prefix);
            Tp*  _result = static_cast<Tp*>(v_result);
            invoke::push<TIMEMORY_API>(std::tie(*_result), _scope + arg_scope, _hash);
        }
    };

    _obj.m_sample = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Sampling %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::invoke<operation::sample, TIMEMORY_API>(std::tie(*_result));
        }
    };

    _obj.m_start = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Starting %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::start<TIMEMORY_API>(std::tie(*_result));
        }
    };

    _obj.m_stop = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Stopping %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::stop<TIMEMORY_API>(std::tie(*_result));
        }
    };

    _obj.m_pop = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Popping %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::pop<TIMEMORY_API>(std::tie(*_result));
        }
    };

    _obj.m_get = [_typeid_hash](void* v_result, void*& _ptr, size_t _hash) {
        if(_hash == _typeid_hash && v_result && !_ptr)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Getting %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            // invoke::get<TIMEMORY_API>(std::tie(*_result), _ptr, _hash);
            // operation::get<Tp>{ *_result, _ptr, _hash };
            invoke::invoke<operation::get, TIMEMORY_API>(std::tie(*_result), _ptr, _hash);
        }
    };

    _obj.m_del = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Deleting %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            delete _result;
        }
    };

    _obj.m_update_stats = [](void* v_result, bool _v) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Updating statistics %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::invoke<operation::add_statistics, TIMEMORY_API>(std::tie(*_result),
                                                                    _v);
        }
    };

    return _obj;
}
//
template <typename Tp>
opaque
base<Tp, void>::get_opaque(scope::config _scope)
{
    auto _typeid_hash = typeid_hash<Tp>();

    opaque _obj{};

    _obj.m_valid = true;

    _obj.m_typeid = _typeid_hash;

    _obj.m_setup = [](void* v_result, string_view_cref_t _prefix,
                      scope::config arg_scope) {
        TIMEMORY_DEBUG_PRINT_HERE("Setting up %s", demangle<Tp>().c_str());
        Tp* _result = static_cast<Tp*>(v_result);
        if(!_result)
            _result = new Tp{};
        invoke::set_prefix<TIMEMORY_API>(std::tie(*_result), _prefix);
        invoke::set_scope<TIMEMORY_API>(std::tie(*_result), arg_scope);
        return (void*) _result;
    };

    _obj.m_push = [_scope](void*& v_result, string_view_cref_t _prefix,
                           scope::config arg_scope) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Pushing %s", demangle<Tp>().c_str());
            auto _hash   = add_hash_id(_prefix);
            Tp*  _result = static_cast<Tp*>(v_result);
            invoke::push<TIMEMORY_API>(std::tie(*_result), _scope + arg_scope, _hash);
        }
    };

    _obj.m_sample = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Sampling %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::invoke<operation::sample, TIMEMORY_API>(std::tie(*_result));
        }
    };

    _obj.m_start = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Starting %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::start<TIMEMORY_API>(std::tie(*_result));
        }
    };

    _obj.m_stop = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Stopping %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::stop<TIMEMORY_API>(std::tie(*_result));
        }
    };

    _obj.m_pop = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Popping %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::pop<TIMEMORY_API>(std::tie(*_result));
        }
    };

    _obj.m_get = [_typeid_hash](void* v_result, void*& _ptr, size_t _hash) {
        if(_hash == _typeid_hash && v_result && !_ptr)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Getting %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::invoke<operation::get, TIMEMORY_API>(std::tie(*_result), _ptr, _hash);
        }
    };

    _obj.m_del = [](void* v_result) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Deleting %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            delete _result;
        }
    };

    _obj.m_update_stats = [](void* v_result, bool _v) {
        if(v_result)
        {
            TIMEMORY_DEBUG_PRINT_HERE("Updating statistics %s", demangle<Tp>().c_str());
            Tp* _result = static_cast<Tp*>(v_result);
            invoke::invoke<operation::add_statistics, TIMEMORY_API>(std::tie(*_result),
                                                                    _v);
        }
    };

    return _obj;
}
//
}  // namespace component
}  // namespace tim
