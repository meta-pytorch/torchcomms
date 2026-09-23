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
 * \file timemory/operations/declaration.hpp
 * \brief The declaration for the types for operations without definitions
 */

#pragma once

#include "timemory/compat/macros.h"
#include "timemory/enum.h"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/demangle.hpp"

#include <algorithm>
#include <bitset>
#include <cstdio>
#include <iostream>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

//
//--------------------------------------------------------------------------------------//
//
#if !defined(SFINAE_WARNING)
#    if defined(DEBUG) || !defined(NDEBUG)
#        define SFINAE_WARNING(TYPE)                                                     \
            if(::tim::trait::is_available<TYPE>::value)                                  \
            {                                                                            \
                static bool _once = false;                                               \
                if(!_once)                                                               \
                {                                                                        \
                    _once = true;                                                        \
                    fprintf(stderr,                                                      \
                            "%s[%s][%s:%i]> Warning! SFINAE disabled for %s\n%s",        \
                            ::tim::log::color::warning(), TIMEMORY_PRETTY_FUNCTION,      \
                            __FILE__, __LINE__, ::tim::demangle<TYPE>().c_str(),         \
                            ::tim::log::color::end());                                   \
                    timemory_print_demangled_backtrace<16>(std::cerr, std::string{},     \
                                                           ::tim::demangle<TYPE>());     \
                }                                                                        \
            }
#    else
#        define SFINAE_WARNING(...)
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//

namespace tim
{
//
namespace component
{
struct empty_storage;
}
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct storage_initializer
/// \brief This provides an object that can initialize the storage opaquely, e.g.
/// \code{.cpp}
/// namespace
/// {
///     tim::storage_initializer storage = tim::storage_initalizer::get<T>();
/// }
/// \endcode
///
struct storage_initializer
{
    TIMEMORY_DEFAULT_OBJECT(storage_initializer)

    template <typename T>
    static enable_if_t<trait::uses_storage<T>::value, storage_initializer> get(
        std::true_type) TIMEMORY_VISIBILITY("default");

    template <typename T>
    static enable_if_t<!trait::uses_storage<T>::value, storage_initializer> get(
        std::false_type) TIMEMORY_VISIBILITY("default");

    template <typename T>
    static storage_initializer get() TIMEMORY_VISIBILITY("default");

    template <size_t Idx, enable_if_t<Idx != TIMEMORY_COMPONENTS_END> = 0>
    static storage_initializer get() TIMEMORY_VISIBILITY("default");

    template <size_t Idx, enable_if_t<Idx == TIMEMORY_COMPONENTS_END> = 0>
    static auto get() TIMEMORY_VISIBILITY("default");

    template <typename... Tp, enable_if_t<sizeof...(Tp) != 1> = 0>
    static auto get() TIMEMORY_VISIBILITY("default");

    template <size_t... Idx, enable_if_t<sizeof...(Idx) != 1> = 0>
    static auto get() TIMEMORY_VISIBILITY("default");

    template <size_t... Idx>
    static auto get(std::index_sequence<Idx...>) TIMEMORY_VISIBILITY("default");
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
storage_initializer
storage_initializer::get()
{
    return get<T>(trait::is_available_t<T>{});
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, enable_if_t<Idx == TIMEMORY_COMPONENTS_END>>
auto
storage_initializer::get()
{
    return get(std::make_index_sequence<Idx>{});
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp, enable_if_t<sizeof...(Tp) != 1>>
auto
storage_initializer::get()
{
    return TIMEMORY_RETURN_FOLD_EXPRESSION(storage_initializer::get<Tp>());
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx, enable_if_t<sizeof...(Idx) != 1>>
auto
storage_initializer::get()
{
    return TIMEMORY_RETURN_FOLD_EXPRESSION(storage_initializer::get<Idx>());
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx>
auto storage_initializer::get(std::index_sequence<Idx...>)
{
    return TIMEMORY_RETURN_FOLD_EXPRESSION(storage_initializer::get<Idx>());
}
//
//--------------------------------------------------------------------------------------//
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
struct non_vexing
{};
//
//--------------------------------------------------------------------------------------//
//
/// \struct common_utils
/// \brief common string manipulation utilities
///
//
//--------------------------------------------------------------------------------------//
//
struct common_utils
{
    using attributes_t   = std::map<string_t, string_t>;
    using strset_t       = std::set<string_t>;
    using stringstream_t = std::stringstream;
    using strvec_t       = std::vector<string_t>;

public:
    template <typename Head, typename... Tail>
    static constexpr bool not_string(enable_if_t<sizeof...(Tail) == 0, int> = 0)
    {
        return !std::is_same<decay_t<Head>, std::string>::value;
    }

    template <typename Head, typename... Tail>
    static constexpr bool not_string(enable_if_t<sizeof...(Tail) != 0, int> = 0)
    {
        return !std::is_same<decay_t<Head>, std::string>::value && not_string<Tail...>();
    }

    template <typename Head, typename... Tail>
    static constexpr bool is_string(enable_if_t<sizeof...(Tail) == 0, int> = 0)
    {
        return std::is_same<decay_t<Head>, std::string>::value;
    }

    template <typename Head, typename... Tail>
    static constexpr bool is_string(enable_if_t<sizeof...(Tail) != 0, int> = 0)
    {
        return std::is_same<decay_t<Head>, std::string>::value && is_string<Tail...>();
    }

public:
    template <typename Tp>
    static size_t get_distance(const Tp& _data)
    {
        return get_distance_sfinae(_data);
    }

private:
    template <typename Tp>
    static auto get_distance_sfinae(const Tp& _data, int)
        -> decltype(std::distance(_data.begin(), _data.end()), size_t())
    {
        return std::distance(_data.begin(), _data.end());
    }

    template <typename Tp>
    static auto get_distance_sfinae(const Tp&, long) -> size_t
    {
        return size_t(1);
    }

    template <typename Tp>
    static auto get_distance_sfinae(const Tp& _data)
        -> decltype(get_distance_sfinae(_data, 0))
    {
        return get_distance_sfinae(_data, 0);
    }

public:
    template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value, int> = 0>
    static Tp get_entry(const Tp& _data, size_t)
    {
        return _data;
    }

    template <typename Tp, enable_if_t<!std::is_arithmetic<Tp>::value, int> = 0>
    static auto get_entry(const Tp& _data, size_t _idx)
        -> decltype(get_entry_sfinae_(_data, _idx))
    {
        return get_entry_sfinae_<Tp>(_data, _idx);
    }

    template <typename Tp, size_t Idx>
    static Tp get_entry(const Tp& _data, size_t)
    {
        return _data;
    }

private:
    template <typename Tp>
    static auto get_entry_sfinae(const Tp& _data, int, size_t _idx)
        -> decltype(_data.begin(), typename Tp::value_type())
    {
        auto sz  = std::distance(_data.begin(), _data.end());
        auto n   = _idx % sz;
        auto itr = _data.begin();
        std::advance(itr, n);
        return *itr;
    }

    template <typename Tp>
    static Tp get_entry_sfinae(const Tp& _data, long, size_t)
    {
        return _data;
    }

    template <typename Tp>
    static auto get_entry_sfinae_(const Tp& _data, size_t _idx)
        -> decltype(get_entry_sfinae(_data, 0, _idx))
    {
        return get_entry_sfinae<Tp>(_data, 0, _idx);
    }

public:
    template <typename Tp, typename Wp, typename Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const Tp& _data, const Wp& _width,
                      const Pp& _prec)
    {
        size_t num_data = get_distance(_data);

        for(size_t i = 0; i < num_data; ++i)
        {
            auto  _idata  = get_entry<Tp>(_data, i);
            auto  _iwidth = get_entry<Wp>(_width, i);
            auto  _iprec  = get_entry<Pp>(_prec, i);
            auto* ss      = new std::stringstream;
            ss->setf(_format);
            (*ss) << std::setw(_iwidth) << std::setprecision(_iprec) << _idata;
            _os.emplace_back(ss);
        }
    }

    template <typename... Tp, size_t... Idx, typename Wp, typename Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const std::tuple<Tp...>& _data,
                      const Wp& _width, const Pp& _prec, index_sequence<Idx...>)
    {
        TIMEMORY_FOLD_EXPRESSION(
            write(_os, _format, std::get<Idx>(_data), _width, _prec));
    }

    template <typename... Tp, typename Wp, typename Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const std::tuple<Tp...>& _data,
                      const Wp& _width, const Pp& _prec)
    {
        constexpr size_t N = sizeof...(Tp);
        write(_os, _format, _data, _width, _prec, make_index_sequence<N>{});
    }

public:
    template <typename Tp>
    static int64_t get_labels_size(const Tp& _data)
    {
        return get_labels_size_sfinae(_data, 0);
    }

private:
    template <typename Tp>
    static auto get_labels_size_sfinae(const Tp& _data, int)
        -> decltype((void) _data.label_array(), int64_t())
    {
        return _data.label_array().size();
    }

    template <typename Tp>
    static auto get_labels_size_sfinae(const Tp&, long) -> int64_t
    {
        return 1;
    }

public:
    template <typename Tp>
    static strvec_t get_labels(const Tp& _data)
    {
        return get_labels_sfinae(_data, 0, 0);
    }

private:
    template <typename Tp>
    static auto get_labels_sfinae(const Tp& _data, int, int)
        -> decltype((void) _data.label_array(), strvec_t{})
    {
        strvec_t _ret;
        for(const auto& itr : _data.label_array())
            _ret.push_back(itr);
        return _ret;
    }

    template <typename Tp>
    static auto get_labels_sfinae(const Tp& _data, int, long)
        -> decltype((void) _data.get_label(), strvec_t{})
    {
        return strvec_t{ _data.get_label() };
    }

    template <typename Tp>
    static auto get_labels_sfinae(const Tp&, long, long) -> strvec_t
    {
        return strvec_t{ "" };
    }

public:
    template <typename T>
    static strvec_t as_string_vec(const T& _data)
    {
        return strvec_t{ _data };
    }

    template <typename Tp>
    static std::string as_string(const Tp& _obj)
    {
        std::stringstream ss;
        ss << _obj;
        return ss.str();
    }

    template <typename... T, size_t... Idx>
    static strvec_t as_string_vec(const std::tuple<T...>& _obj, index_sequence<Idx...>)
    {
        using init_list_type = std::initializer_list<std::string>;
        auto&& ret           = init_list_type{ (as_string(std::get<Idx>(_obj)))... };
        return strvec_t(ret);
    }

    template <typename... T>
    static strvec_t as_string_vec(const std::tuple<T...>& _obj)
    {
        constexpr size_t N = sizeof...(T);
        return as_string_vec(_obj, make_index_sequence<N>{});
    }

public:
    template <typename Tp>
    static strvec_t get_display_units(const Tp& _data)
    {
        return get_display_units_sfinae(_data, 0, 0);
    }

private:
    template <typename Tp>
    static auto get_display_units_sfinae(const Tp& _data, int, int)
        -> decltype((void) _data.display_unit_array(), strvec_t{})
    {
        strvec_t _ret;
        for(const auto& itr : _data.display_unit_array())
            _ret.push_back(itr);
        return _ret;
    }

    template <typename Tp>
    static auto get_display_units_sfinae(const Tp& _data, int, long)
        -> decltype((void) _data.get_display_unit(), strvec_t{})

    {
        return as_string_vec(Tp::get_display_unit());
    }

    template <typename Tp>
    static auto get_display_units_sfinae(const Tp&, long, long) -> strvec_t
    {
        return strvec_t{ "" };
    }

public:
    using sizevector_t = std::vector<size_t>;

    template <typename Tp>
    static sizevector_t get_widths(const Tp& _data)
    {
        return get_widths_sfinae(_data, 0);
    }

private:
    template <typename Tp>
    static auto get_widths_sfinae(const Tp& _data, int)
        -> decltype((void) _data.width_array(), sizevector_t())
    {
        return _data.width_array();
    }

    template <typename Tp>
    static auto get_widths_sfinae(const Tp&, long) -> sizevector_t
    {
        return sizevector_t{ Tp::get_width() };
    }

public:
    //----------------------------------------------------------------------------------//
    /// generate an attribute
    ///
    static string_t attribute_string(const string_t& key, const string_t& item)
    {
        return mpl::apply<string_t>::join("", key, "=", "\"", item, "\"");
    }

    //----------------------------------------------------------------------------------//
    /// replace matching values in item with str
    ///
    static string_t replace(string_t& item, const string_t& str, const strset_t& values)
    {
        for(const auto& itr : values)
        {
            while(item.find(itr) != string_t::npos)
                item = item.replace(item.find(itr), itr.length(), str);
        }
        return item;
    }

    //----------------------------------------------------------------------------------//
    /// convert to lowercase
    ///
    static string_t lowercase(string_t _str)
    {
        for(auto& itr : _str)
            itr = tolower(itr);
        return _str;
    }

    //----------------------------------------------------------------------------------//
    /// convert to uppercase
    ///
    static string_t uppercase(string_t _str)
    {
        for(auto& itr : _str)
            itr = toupper(itr);
        return _str;
    }

    //----------------------------------------------------------------------------------//
    /// check if str contains any of the string items
    ///
    static bool contains(const string_t& str, const strset_t& items)
    {
        auto lstr = lowercase(str);
        return std::any_of(items.begin(), items.end(), [&lstr](const auto& itr) {
            return lstr.find(itr) != string_t::npos;
        });
    }

    //----------------------------------------------------------------------------------//
    /// shorthand for apply<string_t>::join(...)
    ///
    template <typename Tp, typename... Args>
    static string_t join(Tp&& _delim, Args&&... _args)
    {
        return mpl::apply<string_t>::join(std::forward<Tp>(_delim),
                                          std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    static bool is_empty(const std::string& obj) { return obj.empty(); }

    //----------------------------------------------------------------------------------//

    template <typename Tp, typename... ExtraT>
    static bool is_empty(const std::vector<Tp, ExtraT...>& obj)
    {
        for(const auto& itr : obj)
        {
            if(!itr.empty())
                return false;
        }
        return true;
    }

    //----------------------------------------------------------------------------------//

    template <template <typename...> class Tuple, typename... Tp>
    static bool is_empty(const Tuple<Tp...>& obj)
    {
        using input_type   = Tuple<Tp...>;
        constexpr size_t N = sizeof...(Tp);
        std::bitset<N>   _bits;
        TIMEMORY_FOLD_EXPRESSION(
            _bits[index_of<Tp, input_type>::value] =
                (std::get<index_of<Tp, input_type>::value>(obj).empty()));
        return _bits.all();
    }

    //----------------------------------------------------------------------------------//

    template <bool EnabledV, typename Arg, enable_if_t<EnabledV, int> = 0>
    static void print_tag(std::ostream& os, const Arg& _arg)
    {
        if(!is_empty(_arg))
            os << " " << _arg;
    }

    //----------------------------------------------------------------------------------//

    template <bool EnabledV, typename Arg, enable_if_t<!EnabledV, int> = 0>
    static void print_tag(std::ostream&, const Arg&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct init_storage
{
    using type      = Tp;
    using string_t  = std::string;
    using this_type = init_storage<Tp>;
    using pointer_t = tim::base::storage*;
    using get_type  = std::tuple<pointer_t, bool, bool, bool>;

    template <typename Up = Tp>
    init_storage(enable_if_t<trait::uses_value_storage<Up>::value, int> = 0);

    template <typename Up = Tp>
    init_storage(enable_if_t<!trait::uses_value_storage<Up>::value, int> = 0)
    {}

    template <typename U = Tp, typename V = typename U::value_type>
    static get_type get(enable_if_t<trait::uses_value_storage<U, V>::value, int> = 0);

    template <typename U = Tp, typename V = typename U::value_type>
    static get_type get(enable_if_t<!trait::uses_value_storage<U, V>::value &&
                                        !std::is_same<typename U::storage_type,
                                                      component::empty_storage>::value,
                                    int> = 0);

    template <typename U = Tp, typename V = typename U::value_type>
    static get_type get(enable_if_t<!trait::uses_value_storage<U, V>::value &&
                                        std::is_same<typename U::storage_type,
                                                     component::empty_storage>::value,
                                    int> = 0);

    static void init();
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct fini_storage
{
    fini_storage();

private:
    // first check if type is available
    template <typename Up = Tp>
    TIMEMORY_INLINE void sfinae(
        int, enable_if_t<trait::is_available<Up>::value, int> = 0) const;
    //
    template <typename Up = Tp>
    TIMEMORY_INLINE void sfinae(
        long, enable_if_t<!trait::is_available<Up>::value, int> = 0) const;
    //
    // second check for whether storage has finalize function
    template <typename StorageT>
    TIMEMORY_INLINE auto sfinae(StorageT* ptr, int) const
        -> decltype(ptr->finalize(), void())
    {
        ptr->finalize();
    }
    //
    template <typename StorageT>
    TIMEMORY_INLINE void sfinae(StorageT*, long) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
