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

#include "timemory/components/base/types.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

#include <utility>

namespace tim
{
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_prefix
/// \tparam Tp Component type
///
/// \brief Call the set_prefix member function. These instantiations are always inlined
/// because of the use of string_view. Without inlining, you will get undefined symbols
/// in C++14 code when timemory was compiled with C++17.
///
template <typename Tp>
struct set_prefix
{
    using type     = Tp;
    using string_t = string_view_t;

    TIMEMORY_DEFAULT_OBJECT(set_prefix)

    TIMEMORY_INLINE set_prefix(type& obj, const string_t& _prefix);
    TIMEMORY_INLINE set_prefix(type& obj, hash_value_t _nhash, const string_t& _prefix);

    TIMEMORY_INLINE auto operator()(type& obj, const string_t& _prefix) const
    {
        return sfinae_str(obj, 0, 0, 0, _prefix);
    }

    TIMEMORY_HOT auto operator()(type& obj, hash_value_t _nhash) const
    {
        return sfinae_hash(obj, 0, _nhash);
    }

private:
    //  If the component has a set_prefix(const string_t&) member function
    template <typename U>
    TIMEMORY_INLINE auto sfinae_str(U& obj, int, int, int, const string_t& _prefix) const
        -> decltype(obj.set_prefix(_prefix))
    {
        return obj.set_prefix(_prefix);
    }

    template <typename U, typename S>
    TIMEMORY_HOT auto sfinae_str(U& obj, int, int, long, const S& _prefix) const
        -> decltype(obj.set_prefix(_prefix.c_str()))
    {
        return obj.set_prefix(_prefix.c_str());
    }

    template <typename U, typename S>
    TIMEMORY_HOT auto sfinae_str(U& obj, int, long, long, const S& _prefix) const
        -> decltype(obj.set_prefix(_prefix.data()))
    {
        return obj.set_prefix(_prefix.data());
    }

    //  If the component does not have a set_prefix(const string_t&) member function
    template <typename U>
    TIMEMORY_INLINE void sfinae_str(U&, long, long, long, const string_t&) const
    {}

private:
    //  If the component has a set_prefix(hash_value_t) member function
    template <typename U>
    TIMEMORY_HOT auto sfinae_hash(U& obj, int, hash_value_t _nhash) const
        -> decltype(obj.set_prefix(_nhash))
    {
        return obj.set_prefix(_nhash);
    }

    //  If the component does not have a set_prefix(hash_value_t) member function
    template <typename U>
    TIMEMORY_INLINE void sfinae_hash(U&, long, hash_value_t) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
set_prefix<Tp>::set_prefix(type& obj, const string_t& _prefix)
{
    sfinae_str(obj, 0, 0, 0, _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
set_prefix<Tp>::set_prefix(type& obj, hash_value_t _nhash, const string_t& _prefix)
{
    sfinae_hash(obj, 0, _nhash);
    sfinae_str(obj, 0, 0, 0, _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct set_scope
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(set_scope)

    TIMEMORY_HOT set_scope(type& obj, scope::config _data);

    TIMEMORY_HOT auto operator()(type& obj, scope::config _data) const
    {
        return sfinae(obj, 0, _data);
    }

private:
    //  If the component has a set_scope(...) member function
    template <typename T>
    TIMEMORY_HOT auto sfinae(T& obj, int, scope::config _data) const
        -> decltype(obj.set_scope(_data))
    {
        return obj.set_scope(_data);
    }

    //  If the component does not have a set_scope(...) member function
    template <typename T>
    TIMEMORY_INLINE void sfinae(T&, long, scope::config) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
set_scope<Tp>::set_scope(type& obj, scope::config _data)
{
    sfinae(obj, 0, _data);
}
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct set_state
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(set_state)

    TIMEMORY_HOT set_state(type& obj, component::base_state* _data);

    TIMEMORY_HOT auto operator()(type& obj, component::base_state* _data)
    {
        return sfinae(obj, 0, _data);
    }

private:
    //  If the component has a set_scope(...) member function
    template <typename T>
    TIMEMORY_HOT auto sfinae(T& obj, int, component::base_state* _data)
        -> decltype(obj.set_state(_data))
    {
        return obj.set_state(_data);
    }

    //  If the component does not have a set_scope(...) member function
    template <typename T>
    TIMEMORY_INLINE void sfinae(T&, long, component::base_state*)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
set_state<Tp>::set_state(type& obj, component::base_state* _data)
{
    sfinae(obj, 0, _data);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct set_depth_change
{
    TIMEMORY_DEFAULT_OBJECT(set_depth_change)

    template <typename Up>
    TIMEMORY_HOT auto operator()(Up& obj, bool v) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0, v);
    }

    constexpr auto operator()() const { return sfinae<Tp>(0); }

private:
    template <typename Up>
    static TIMEMORY_HOT auto sfinae(Up& obj, int, bool v)
        -> decltype(obj.set_depth_change(v))
    {
        return obj.set_depth_change(v);
    }

    template <typename Up>
    static TIMEMORY_INLINE auto sfinae(Up&, long, bool)
    {}

    template <typename Up>
    constexpr auto sfinae(int) const
        -> decltype(std::declval<Up>().set_depth_change(std::declval<bool>()), bool())
    {
        return true;
    }

    template <typename Up>
    constexpr bool sfinae(long) const
    {
        return false;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct set_is_flat
{
    TIMEMORY_DEFAULT_OBJECT(set_is_flat)

    template <typename Up>
    TIMEMORY_HOT auto operator()(Up& obj, bool v) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0, v);
    }

    constexpr auto operator()() const { return sfinae<Tp>(0); }

private:
    template <typename Up>
    static TIMEMORY_HOT auto sfinae(Up& obj, int, bool v) -> decltype(obj.set_is_flat(v))
    {
        return obj.set_is_flat(v);
    }

    template <typename Up>
    static TIMEMORY_INLINE auto sfinae(Up&, long, bool)
    {}

    template <typename Up>
    constexpr auto sfinae(int) const
        -> decltype(std::declval<Up>().set_is_flat(std::declval<bool>()), bool())
    {
        return true;
    }

    template <typename Up>
    constexpr bool sfinae(long) const
    {
        return false;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct set_is_on_stack
{
    TIMEMORY_DEFAULT_OBJECT(set_is_on_stack)

    template <typename Up>
    TIMEMORY_HOT auto operator()(Up& obj, bool v) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0, v);
    }

private:
    template <typename Up>
    static TIMEMORY_HOT auto sfinae(Up& obj, int, bool v)
        -> decltype(obj.set_is_on_stack(v))
    {
        return obj.set_is_on_stack(v);
    }

    template <typename Up>
    static TIMEMORY_INLINE auto sfinae(Up&, long, bool)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct set_is_invalid
{
    TIMEMORY_DEFAULT_OBJECT(set_is_invalid)

    template <typename Up>
    TIMEMORY_HOT auto operator()(Up& obj, bool v) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0, v);
    }

    constexpr auto operator()() const { return sfinae<Tp>(0); }

private:
    template <typename Up>
    static TIMEMORY_HOT auto sfinae(Up& obj, int, bool v)
        -> decltype(obj.set_is_invalid(v))
    {
        return obj.set_is_invalid(v);
    }

    template <typename Up>
    static TIMEMORY_INLINE auto sfinae(Up&, long, bool)
    {}

    template <typename Up>
    constexpr auto sfinae(int) const
        -> decltype(std::declval<Up>().set_is_invalid(std::declval<bool>()), bool())
    {
        return true;
    }

    template <typename Up>
    constexpr bool sfinae(long) const
    {
        return false;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct set_iterator
{
    TIMEMORY_DEFAULT_OBJECT(set_iterator)

    template <typename Up>
    TIMEMORY_HOT auto operator()(Tp& obj, Up&& v)
    {
        return sfinae(obj, 0, std::forward<Up>(v));
    }

private:
    template <typename ObjT, typename ItrT>
    TIMEMORY_HOT auto sfinae(ObjT& obj, int, ItrT&& v)
        -> decltype(std::declval<ObjT>().set_iterator(std::forward<ItrT>(v)))
    {
        return obj.set_iterator(std::forward<ItrT>(v));
    }

    template <typename ObjT, typename ItrT>
    TIMEMORY_INLINE auto sfinae(ObjT&, long, ItrT&&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct set_is_running
{
    TIMEMORY_DEFAULT_OBJECT(set_is_running)

    template <typename Up>
    TIMEMORY_HOT auto operator()(Up& obj, bool v) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0, v);
    }

private:
    template <typename Up>
    static TIMEMORY_HOT auto sfinae(Up& obj, int, bool v)
        -> decltype(obj.set_is_running(v))
    {
        return obj.set_is_running(v);
    }

    template <typename Up>
    static TIMEMORY_INLINE auto sfinae(Up&, long, bool)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
