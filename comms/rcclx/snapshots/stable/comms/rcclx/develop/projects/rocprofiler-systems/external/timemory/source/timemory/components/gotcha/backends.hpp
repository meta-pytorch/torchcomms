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

/**
 * \file timemory/components/gotcha/backends.hpp
 * \brief Implementation of the gotcha functions/utilities
 */

#pragma once

#include "timemory/backends/gotcha.hpp"
#include "timemory/mpl/quirks.hpp"
#include "timemory/operations/types/set_data.hpp"
#include "timemory/utility/mangler.hpp"
#include "timemory/variadic/types.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

//======================================================================================//
//
namespace tim
{
namespace trait
{
template <typename Tp>
struct fast_gotcha : std::false_type
{};
//
template <size_t N, typename CompT, typename DiffT>
struct static_data<component::gotcha<N, CompT, DiffT>>
{
    using type                  = std::conditional_t<is_one_of<DiffT, CompT>::value ||
                                        (mpl::get_tuple_size<CompT>::value == 0 &&
                                         concepts::is_component<DiffT>::value),
                                    std::true_type, std::false_type>;
    static constexpr bool value = type::value;
};
}  // namespace trait
//
namespace backend
{
namespace gotcha
{
//
template <typename DiffT, typename DataT>
struct replaces
{
    static constexpr bool value =
        is_one_of<DiffT, DataT>::value ||
        (mpl::get_tuple_size<DataT>::value == 0 && concepts::is_component<DiffT>::value);
};
}  // namespace gotcha
}  // namespace backend
//
namespace component
{
struct gotcha_data;
//
//======================================================================================//
///
/// \struct tim::component::gotcha_invoker
///
///
template <typename Tp, typename Ret, bool SetDataV>
struct gotcha_invoker
{
    using Type       = Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    TIMEMORY_DEFAULT_OBJECT(gotcha_invoker)

    template <typename FuncT, typename... Args>
    static TIMEMORY_INLINE decltype(auto) invoke(Tp& _obj, gotcha_data&& _data,
                                                 FuncT&& _func, Args&&... _args)
    {
        return gotcha_invoker{}(_obj, std::forward<gotcha_data>(_data),
                                std::forward<FuncT>(_func), std::forward<Args>(_args)...);
    }

    template <typename FuncT, typename... Args>
    TIMEMORY_INLINE decltype(auto) operator()(Tp& _obj, gotcha_data&& _data,
                                              FuncT&& _func, Args&&... _args) const
    {
        IF_CONSTEXPR(SetDataV)
        {
            // if object has set_data(gotcha_data) member function
            operation::set_data<Tp>{}(_obj, std::forward<gotcha_data>(_data));
            // if object has set_data(<function-pointer>) member function
            operation::set_data<Tp>{}(_obj, std::forward<FuncT>(_func));
        }
        //
        return invoke_impl(_obj, std::forward<gotcha_data>(_data),
                           std::forward<FuncT>(_func), std::forward<Args>(_args)...);
    }

private:
    template <typename DataT, typename FuncT, typename... Args>
    static decltype(auto) invoke_impl(Tp& _obj, DataT&& _data, FuncT&& _func,
                                      Args&&... _args)
    {
        if constexpr(std::is_invocable<Tp, DataT&&, FuncT&&, Args&&...>::value)
        {
            // Ret Type::operator{}(gotcha_data, <function-pointer>, Args...)
            return _obj(std::forward<DataT>(_data), std::forward<FuncT>(_func),
                        std::forward<Args>(_args)...);
        }
        else if constexpr(std::is_invocable<Tp, DataT&&, Args&&...>::value)
        {
            // Ret Type::operator{}(gotcha_data, Args...)
            return _obj(std::forward<DataT>(_data), std::forward<Args>(_args)...);
        }
        else if constexpr(std::is_invocable<Tp, FuncT&&, Args&&...>::value)
        {
            //  Ret Type::operator{}(<function-pointer>, Args...)
            return _obj(std::forward<FuncT>(_func), std::forward<Args>(_args)...);
        }
        else if constexpr(std::is_invocable<Tp, Args&&...>::value)
        {
            // Ret Type::operator{}(Args...)
            return _obj(std::forward<Args>(_args)...);
        }
        else if constexpr(std::is_invocable<FuncT&&, Args&&...>::value)
        {
            // call the original function pointer
            return std::forward<FuncT>(_func)(std::forward<Args>(_args)...);
        }
        else
        {
            static_assert(std::is_same<Tp, FuncT>::value,
                          "Error! no possible invocation");
        }
    }
};
//
//======================================================================================//
//
///
/// \struct tim::component::gotcha_data
/// \brief Holds the properties for wrapping and unwrapping a GOTCHA binding
struct gotcha_data
{
    using binding_t     = backend::gotcha::binding_t;
    using wrappee_t     = backend::gotcha::wrappee_t;
    using wrappid_t     = backend::gotcha::string_t;
    using constructor_t = std::function<void()>;
    using destructor_t  = std::function<void()>;

    gotcha_data()  = default;
    ~gotcha_data() = default;

    gotcha_data(const gotcha_data&) = delete;
    gotcha_data(gotcha_data&&)      = delete;
    gotcha_data& operator=(const gotcha_data&) = delete;
    gotcha_data& operator=(gotcha_data&&) = delete;

    bool          ready        = false;        /// ready to be used
    bool          filled       = false;        /// structure is populated
    bool          is_active    = false;        /// is currently wrapping
    bool          is_finalized = false;        /// no more wrapping is allowed
    int           verbose      = 0;            /// warning verbosity
    int           priority     = 0;            /// current priority
    size_t        index        = 0;            /// index in gotcha wrapper
    binding_t     binding      = binding_t{};  /// hold the binder set
    wrappee_t     wrapper      = nullptr;      /// the func pointer doing wrapping
    wrappee_t     wrappee      = nullptr;      /// the func pointer being wrapped
    wrappid_t     wrap_id      = {};           /// function name (possibly mangled)
    wrappid_t     tool_id      = {};           /// function name (unmangled)
    bool*         suppression  = nullptr;      /// turn on/off some suppression var
    bool*         debug        = nullptr;      /// enable debugging
    void*         instance     = nullptr;      /// static instance of caller
    constructor_t constructor  = []() {};      /// wrap the function
    destructor_t  destructor   = []() {};      /// unwrap the function
};
//
//======================================================================================//
//
///
/// \struct tim::component::gotcha_config
/// \brief A simple type definition for specifying the index, return value, and the
/// arguments
template <size_t Idx, typename Ret, typename... Args>
struct gotcha_config
{
    gotcha_config(std::string _name, int _prio = 0, std::string _tool = {})
    : priority{ _prio }
    , tool{ std::move(_tool) }
    , names{ std::vector<std::string>{ std::move(_name) } }
    {}

    gotcha_config(std::vector<std::string> _names, int _prio = 0, std::string _tool = {})
    : priority{ _prio }
    , tool{ std::move(_tool) }
    , names{ std::move(_names) }
    {}

    int                      priority = 0;
    std::string              tool     = {};
    std::vector<std::string> names    = {};
};
}  // namespace component
}  // namespace tim
//
//======================================================================================//
