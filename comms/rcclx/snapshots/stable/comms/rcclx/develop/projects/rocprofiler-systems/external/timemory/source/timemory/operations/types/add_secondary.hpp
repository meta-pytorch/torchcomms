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
 * \file timemory/operations/types/add_secondary.hpp
 * \brief Definition for various functions for add_secondary in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/demangle.hpp"

#include <stdexcept>

namespace tim
{
namespace operation
{
namespace internal
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, bool>
struct add_secondary;
//
template <typename Tp>
struct add_secondary<Tp, true>
{
    using type     = Tp;
    using string_t = std::string;

    TIMEMORY_DEFAULT_OBJECT(add_secondary)

    //----------------------------------------------------------------------------------//
    // if secondary data explicitly specified
    //
    template <typename Storage, typename Iterator>
    add_secondary(Storage* _storage, Iterator _itr, const type& _rhs)
    {
        if(!_storage || !settings::add_secondary())
            return;
        (*this)(_storage, _itr, _rhs);
    }

    //----------------------------------------------------------------------------------//
    // add_secondary called on type
    //
    template <typename... Args>
    add_secondary(type& _rhs, Args&&... args)
    {
        if(!settings::add_secondary())
            return;
        (*this)(_rhs, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // if secondary data explicitly specified
    //
    template <typename Storage, typename Iterator, typename Up>
    auto operator()(Storage* _storage, Iterator _itr, const Up& _rhs) const
    {
        if(!_storage)
            return;
        using map_type         = decay_t<decltype(_rhs.get_secondary())>;
        using value_type       = typename map_type::mapped_type;
        using secondary_data_t = std::tuple<Iterator, const string_t&, value_type>;
        for(const auto& _data : _rhs.get_secondary())
        {
            storage_sfinae(_storage, 0,
                           secondary_data_t{ _itr, _data.first, _data.second });
        }
    }

    //----------------------------------------------------------------------------------//
    // add_secondary called on type
    //
    template <typename... Args>
    auto operator()(type& _rhs, Args&&... args) const
    {
        return sfinae(_rhs, 0, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // if no storage type
    //
    template <typename... Args>
    add_secondary(std::nullptr_t, Args...)
    {}

    template <typename... Args>
    auto operator()(std::nullptr_t, Args...)
    {}

private:
    //----------------------------------------------------------------------------------//
    //  If the component has a add_secondary(Args...) member function
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& _obj, int, Args&&... args) const
        -> decltype(_obj.add_secondary(std::forward<Args>(args)...))
    {
        return _obj.add_secondary(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a add_secondary(Args...) member function
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, Args&&...) const
    {}

    //----------------------------------------------------------------------------------//
    //  If the storage can append secondary data
    //
    template <typename Storage, typename DataT>
    auto storage_sfinae(Storage* _storage, int, DataT&& _data) const
        -> decltype(_storage->append(std::forward<DataT>(_data)))
    {
        return _storage->append(std::forward<DataT>(_data));
    }

    //----------------------------------------------------------------------------------//
    //  If the storage can NOT append secondary data
    //
    template <typename Storage, typename DataT>
    void storage_sfinae(Storage*, long, DataT&&) const
    {}
};
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct add_secondary<Tp, false>
{
    using type     = Tp;
    using string_t = std::string;

    TIMEMORY_DEFAULT_OBJECT(add_secondary)

    //----------------------------------------------------------------------------------//
    // check if secondary data implicitly specified
    //
    template <typename Storage, typename Iterator>
    add_secondary(Storage* _storage, Iterator _itr, const type& _rhs)
    {
        (*this)(_storage, _itr, _rhs);
    }

    //----------------------------------------------------------------------------------//
    // add_secondary called on type
    //
    template <typename... Args>
    add_secondary(type& _rhs, Args&&... args)
    {
        (*this)(_rhs, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // if secondary data explicitly specified
    //
    template <typename Storage, typename Iterator>
    auto operator()(Storage* _storage, Iterator _itr, const type& _rhs) const
    {
        return storage_sfinae(_storage, _itr, _rhs, 0);
    }

    //----------------------------------------------------------------------------------//
    // add_secondary called on type
    //
    template <typename... Args>
    auto operator()(type& _rhs, Args&&... args) const
    {
        return sfinae(_rhs, 0, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // if no storage type
    //
    template <typename... Args>
    add_secondary(std::nullptr_t, Args...)
    {}

    template <typename... Args>
    auto operator()(std::nullptr_t, Args...)
    {}

private:
    //----------------------------------------------------------------------------------//
    //  If the component has a get_secondary() member function
    //
    template <typename Storage, typename Iterator, typename Up>
    auto storage_sfinae(Storage* _storage, Iterator _itr, const Up& _rhs, int) const
        -> decltype(_rhs.get_secondary(), void())
    {
        if(_storage == nullptr || !settings::add_secondary())
            return;

        using map_type         = decay_t<decltype(_rhs.get_secondary())>;
        using value_type       = typename map_type::mapped_type;
        using secondary_data_t = std::tuple<Iterator, const string_t&, value_type>;
        for(const auto& _data : _rhs.get_secondary())
            storage_append_sfinae(_storage, 0,
                                  secondary_data_t{ _itr, _data.first, _data.second });
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a get_secondary() member function
    //
    template <typename Storage, typename Iterator>
    void storage_sfinae(Storage*, Iterator, const type&, long) const
    {}

    //----------------------------------------------------------------------------------//
    //  If the component has a add_secondary(Args...) member function
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& _obj, int, Args&&... args) const
        -> decltype(_obj.add_secondary(std::forward<Args>(args)...), void())
    {
        if(!settings::add_secondary())
            return;

        _obj.add_secondary(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a add_secondary(Args...) member function
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, Args&&...) const
    {}

    //----------------------------------------------------------------------------------//
    //  If the storage can append secondary data
    //
    template <typename Storage, typename DataT>
    auto storage_append_sfinae(Storage* _storage, int, DataT&& _data) const
        -> decltype(_storage->append(std::forward<DataT>(_data)))
    {
        return _storage->append(std::forward<DataT>(_data));
    }

    //----------------------------------------------------------------------------------//
    //  If the storage can NOT append secondary data
    //
    template <typename Storage, typename DataT>
    void storage_append_sfinae(Storage*, long, DataT&&) const
    {}
};
//
}  // namespace internal
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::add_secondary
/// \tparam Tp Component type
///
/// \brief
/// component contains secondary data resembling the original data
/// but should be another node entry in the graph. These types
/// must provide a get_secondary() member function and that member function
/// must return a pair-wise iterable container, e.g. std::map, of types:
///     - std::string
///     - value_type or Tp
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct add_secondary
: public internal::add_secondary<Tp, trait::secondary_data<Tp>::value>
{
    using type      = Tp;
    using string_t  = std::string;
    using base_type = internal::add_secondary<Tp, trait::secondary_data<Tp>::value>;

    TIMEMORY_DEFAULT_OBJECT(add_secondary)

    add_secondary(const type& _rhs, typename type::storage_type* _storage)
    : base_type{ _storage, operation::get_iterator<Tp>{}(_rhs), _rhs }
    {}

    //----------------------------------------------------------------------------------//
    // if secondary data explicitly specified
    //
    template <typename Storage, typename Iterator>
    add_secondary(Storage* _storage, Iterator _itr, const type& _rhs)
    : base_type{ _storage, _itr, _rhs }
    {}

    //----------------------------------------------------------------------------------//
    // add_secondary called on type
    //
    template <typename... Args>
    add_secondary(type& _rhs, Args&&... args)
    : base_type{ _rhs, std::forward<Args>(args)... }
    {}

    //----------------------------------------------------------------------------------//
    // add_secondary called without storage
    //
    template <typename... Args>
    add_secondary(std::nullptr_t, Args...)
    {}

    //----------------------------------------------------------------------------------//
    // add_secondary function call operator
    //
    template <typename Storage, typename... Args>
    auto operator()(Storage* _storage, Args&&... _args) const
    {
        return base_type::operator()(_storage, std::forward<Args>(_args)...);
    }

    template <typename... Args>
    auto operator()(std::nullptr_t, Args...)
    {}

    template <typename... Args>
    auto operator()(type& _rhs, Args&&... _args) const
    {
        return base_type::operator()(_rhs, std::forward<Args>(_args)...);
    }

    template <typename... Args>
    auto operator()(type* _rhs, Args&&... _args) const
    {
        using return_type =
            decltype(base_type::operator()(*_rhs, std::forward<Args>(_args)...));
        if(_rhs)
            return base_type::operator()(*_rhs, std::forward<Args>(_args)...);
        return return_type();
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
