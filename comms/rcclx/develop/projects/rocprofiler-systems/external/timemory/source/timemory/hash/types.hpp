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

#include "timemory/api.hpp"
#include "timemory/hash/macros.hpp"
#include "timemory/hash/static_string.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/concepts.hpp"

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              GENERAL PURPOSE TLS DATA
//
//--------------------------------------------------------------------------------------//
//
using shared_ptr_pair_callback_t = std::function<void(int64_t)>;
//
shared_ptr_pair_callback_t*&
get_shared_ptr_pair_callback();
//
template <typename Tp, typename Tag = TIMEMORY_API, typename PtrT = std::shared_ptr<Tp>,
          typename PairT = std::pair<PtrT, PtrT>>
PairT*&
get_shared_ptr_pair() TIMEMORY_VISIBLE;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag = TIMEMORY_API, typename PtrT = std::shared_ptr<Tp>,
          typename PairT = std::pair<PtrT, PtrT>>
PtrT
get_shared_ptr_pair_instance() TIMEMORY_VISIBLE;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag = TIMEMORY_API, typename PtrT = std::shared_ptr<Tp>,
          typename PairT = std::pair<PtrT, PtrT>>
PtrT
get_shared_ptr_pair_main_instance() TIMEMORY_VISIBLE;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag = TIMEMORY_API, typename PtrT = std::shared_ptr<Tp>>
PtrT
get_shared_ptr_lone_instance() TIMEMORY_VISIBLE;
//
//--------------------------------------------------------------------------------------//
//
//                                  HASH ALIASES
//
//--------------------------------------------------------------------------------------//
//
inline namespace hash
{
using hash_value_t        = size_t;
using hash_map_t          = std::unordered_map<hash_value_t, std::string>;
using hash_alias_map_t    = std::unordered_map<hash_value_t, hash_value_t>;
using hash_map_ptr_t      = std::shared_ptr<hash_map_t>;
using hash_map_ptr_pair_t = std::pair<hash_map_ptr_t, hash_map_ptr_t>;
using hash_alias_ptr_t    = std::shared_ptr<hash_alias_map_t>;
using hash_resolver_t     = std::function<bool(hash_value_t, std::string&)>;
using hash_resolver_vec_t = std::vector<hash_resolver_t>;
//
//--------------------------------------------------------------------------------------//
//
//                              Generic hash functions
//
//--------------------------------------------------------------------------------------//
//
namespace hash_impl
{
template <typename Tp>
inline auto
typeid_name()
{
    return typeid(Tp).name();
}

template <typename Tp, typename Up = Tp>
inline auto
typeid_hash(int) -> decltype(typeid_name<Tp>(), size_t{})
{
    return std::type_index(typeid(Tp)).hash_code();
}
//
template <typename Tp, typename Up = Tp>
inline auto
typeid_hash(long)
{
    return 0;
}
}  // namespace hash_impl

template <typename Tp>
inline auto
typeid_hash()
{
    return hash_impl::typeid_hash<Tp>(0);
}
//
template <typename T>
TIMEMORY_INLINE size_t
get_hash(T&& obj)
{
    return std::hash<std::decay_t<T>>()(std::forward<T>(obj));
}

TIMEMORY_INLINE size_t
get_hash(string_view_cref_t str)
{
    return std::hash<string_view_t>{}(str);
}

TIMEMORY_INLINE size_t
get_hash(const char* cstr)
{
    return std::hash<string_view_t>{}(cstr);
}

template <typename T>
struct hasher
{
    inline size_t operator()(T&& val) const { return get_hash(std::forward<T>(val)); }
    inline size_t operator()(const T& val) const { return get_hash(val); }
};
//
//--------------------------------------------------------------------------------------//
//
//                              HASH DATA STRUCTURES
//
//--------------------------------------------------------------------------------------//
//
hash_map_ptr_t&
get_hash_ids() TIMEMORY_HOT TIMEMORY_VISIBLE;
//
hash_map_ptr_t&
get_main_hash_ids() TIMEMORY_HOT TIMEMORY_VISIBLE;
//
//--------------------------------------------------------------------------------------//
//
hash_alias_ptr_t&
get_hash_aliases() TIMEMORY_HOT TIMEMORY_VISIBLE;
//
hash_alias_ptr_t&
get_main_hash_aliases() TIMEMORY_HOT TIMEMORY_VISIBLE;
//
//--------------------------------------------------------------------------------------//
//
std::shared_ptr<hash_resolver_vec_t>&
get_hash_resolvers() TIMEMORY_HOT TIMEMORY_VISIBLE;
//
//--------------------------------------------------------------------------------------//
//
//                              STRING -> HASH CONVERSION
//
//--------------------------------------------------------------------------------------//
//
// declarations
//
template <typename Tp,
          std::enable_if_t<concepts::is_string_type<std::decay_t<Tp>>::value, int> = 0>
TIMEMORY_INLINE hash_value_t
get_hash_id(Tp&& _v);
//
template <typename Tp,
          std::enable_if_t<!concepts::is_string_type<std::decay_t<Tp>>::value, int> = 0>
TIMEMORY_INLINE hash_value_t
get_hash_id(Tp&& _v);
//
TIMEMORY_INLINE hash_value_t
get_combined_hash_id(hash_value_t _lhs, hash_value_t _rhs);
//
template <typename Tp,
          std::enable_if_t<!std::is_integral<std::decay_t<Tp>>::value, int> = 0>
TIMEMORY_INLINE hash_value_t
get_combined_hash_id(hash_value_t _lhs, Tp&& _rhs);
//
template <typename Tp, typename Arg, typename... Args>
TIMEMORY_INLINE hash_value_t
get_combined_hash_id(hash_value_t _lhs, Tp&& _rhs, Arg&& _arg, Args&&... _args);
//
template <typename Tp, typename... Args>
TIMEMORY_INLINE hash_value_t
get_hash_id(hash_value_t _lhs, Tp&& _rhs, Args&&... _args);
//
//  get hash of a string type
//
template <typename Tp,
          std::enable_if_t<concepts::is_string_type<std::decay_t<Tp>>::value, int>>
hash_value_t
get_hash_id(Tp&& _v)
{
    return std::hash<string_view_t>{}(std::forward<Tp>(_v));
}
//
//  get hash of a non-string type
//
template <typename Tp,
          std::enable_if_t<!concepts::is_string_type<std::decay_t<Tp>>::value, int>>
hash_value_t
get_hash_id(Tp&& _v)
{
    return std::hash<std::decay_t<Tp>>{}(std::forward<Tp>(_v));
}
//
//  combine two existing hashes
//
hash_value_t
get_combined_hash_id(hash_value_t _lhs, hash_value_t _rhs)
{
    constexpr hash_value_t offset = 0x9e3779b97f4a7c17ul;
    return (_lhs ^= _rhs + offset + (_lhs << 6) + (_lhs >> 2));
}
//
//  compute the hash and combine it with an existing hash
//
template <typename Tp, std::enable_if_t<!std::is_integral<std::decay_t<Tp>>::value, int>>
hash_value_t
get_combined_hash_id(hash_value_t _lhs, Tp&& _rhs)
{
    return get_combined_hash_id(_lhs, get_hash_id(std::forward<Tp>(_rhs)));
}
//
//  recursively process extra arguments
//
template <typename Tp, typename Arg, typename... Args>
hash_value_t
get_combined_hash_id(hash_value_t _lhs, Tp&& _rhs, Arg&& _arg, Args&&... _args)
{
    return get_combined_hash_id(_lhs,
                                get_combined_hash_id(get_hash_id(std::forward<Tp>(_rhs)),
                                                     std::forward<Arg>(_arg),
                                                     std::forward<Args>(_args)...));
}
//
//
//
template <typename Tp, typename... Args>
hash_value_t
get_hash_id(hash_value_t _lhs, Tp&& _rhs, Args&&... _args)
{
    return get_combined_hash_id(_lhs, get_hash_id(std::forward<Tp>(_rhs)),
                                std::forward<Args>(_args)...);
}
//
//--------------------------------------------------------------------------------------//
//
hash_value_t
get_hash_id(const hash_alias_ptr_t& _hash_alias, hash_value_t _hash_id) TIMEMORY_HOT;
//
//--------------------------------------------------------------------------------------//
//
/// \fn hash_value_t add_hash_id(hash_map_ptr_t&, string_view_cref_t)
/// \brief add an string to the given hash-map (if it doesn't already exist) and return
/// the hash
///
hash_value_t
add_hash_id(hash_map_ptr_t& _hash_map, string_view_cref_t _prefix) TIMEMORY_HOT;
//
//--------------------------------------------------------------------------------------//
//
/// \fn hash_value_t add_hash_id(string_view_cref_t)
/// \brief add an string to the default hash-map (if it doesn't already exist) and return
/// the hash
///
hash_value_t
add_hash_id(string_view_cref_t _prefix) TIMEMORY_HOT;
//
inline hash_value_t
add_hash_id(string_view_cref_t _prefix)
{
    return add_hash_id(get_hash_ids(), _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
void
add_hash_id(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
            hash_value_t _hash_id, hash_value_t _alias_hash_id) TIMEMORY_HOT;
//
void
add_hash_id(const hash_alias_ptr_t& _hash_alias, hash_value_t _hash_id,
            hash_value_t _alias_hash_id) TIMEMORY_HOT;
//
void
add_hash_id(hash_value_t _hash_id, hash_value_t _alias_hash_id) TIMEMORY_HOT;
//
void
hash_identifier_error(const hash_map_ptr_t&   _hash_map,
                      const hash_alias_ptr_t& _hash_alias, hash_value_t _hash_id);
//
//--------------------------------------------------------------------------------------//
//
//                              HASH -> STRING CONVERSION
//
//--------------------------------------------------------------------------------------//
//
/// \fn string_view_t get_hash_identifier_fast(hash_value_t)
/// \brief this does not check other threads or aliases. Only call this function when
/// you know that the hash exists on the thread and is not an alias
//
string_view_t
get_hash_identifier_fast(hash_value_t _hash) TIMEMORY_HOT;
//
bool
get_hash_identifier_fast(hash_value_t _hash, std::string*& _v) TIMEMORY_HOT;
//
bool
get_hash_identifier_fast(hash_value_t _hash, const char*& _v) TIMEMORY_HOT;
//
//--------------------------------------------------------------------------------------//
//
typename hash_map_t::const_iterator
find_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                     hash_value_t _hash_id);
//
typename hash_map_t::const_iterator
find_hash_identifier(hash_value_t _hash_id);
//
bool
get_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                    hash_value_t _hash_id, std::string*& _ret);
//
bool
get_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                    hash_value_t _hash_id, const char*& _ret);
//
bool
get_hash_identifier(hash_value_t _hash_id, std::string*& _ret);
//
bool
get_hash_identifier(hash_value_t _hash_id, const char*& _ret);
//
std::string
get_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                    hash_value_t _hash_id);
//
std::string
get_hash_identifier(hash_value_t _hash_id);
//
//--------------------------------------------------------------------------------------//
//
template <typename FuncT>
size_t
add_hash_resolver(FuncT&& _func)
{
    auto _resolvers = get_hash_resolvers();
    if(!_resolvers)
        return 0;
    _resolvers->emplace_back(std::forward<FuncT>(_func));
    return _resolvers->size();
}
//
//--------------------------------------------------------------------------------------//
//
//                              HASH STRING DEMANGLING
//
//--------------------------------------------------------------------------------------//
//
std::string
demangle_hash_identifier(std::string, char bdelim = '[', char edelim = ']');
//
//--------------------------------------------------------------------------------------//
//
template <typename... Args>
auto
get_demangled_hash_identifier(Args&&... _args)
{
    return demangle_hash_identifier(get_hash_identifier(std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace hash
}  // namespace tim
