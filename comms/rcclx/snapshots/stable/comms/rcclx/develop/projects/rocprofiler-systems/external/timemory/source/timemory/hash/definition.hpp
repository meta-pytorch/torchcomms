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
#include "timemory/hash/declaration.hpp"
#include "timemory/hash/macros.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/locking.hpp"

#include <cstdint>
#include <iomanip>
#include <iosfwd>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#if defined(TIMEMORY_HASH_SOURCE) || !defined(TIMEMORY_USE_HASH_EXTERN)

namespace tim
{
inline namespace hash
{
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_map_ptr_t&)
get_hash_ids()
{
    static thread_local auto _inst =
        get_shared_ptr_pair_instance<hash_map_t, TIMEMORY_API>();
    static thread_local auto _dtor = scope::destructor{ []() {
        auto                 _main = get_shared_ptr_pair_main_instance<hash_map_t, TIMEMORY_API>();
        if(!_inst || !_main || _inst == _main)
            return;
        auto_lock_t          _lk{ type_mutex<hash_map_t>(), std::defer_lock };
        if(!_lk.owns_lock())
            _lk.lock();
        for(const auto& itr : *_inst)
        {
            if(_main->find(itr.first) == _main->end())
                _main->emplace(itr.first, itr.second);
        }
    } };
    return _inst;
    (void) _dtor;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_map_ptr_t&)
get_main_hash_ids()
{
    static thread_local auto _inst =
        get_shared_ptr_pair_main_instance<hash_map_t, TIMEMORY_API>();
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_alias_ptr_t&)
get_hash_aliases()
{
    static thread_local auto _inst =
        get_shared_ptr_pair_instance<hash_alias_map_t, TIMEMORY_API>();
    static thread_local auto _dtor = scope::destructor{ []() {
        auto                 _main = get_shared_ptr_pair_main_instance<hash_alias_map_t, TIMEMORY_API>();
        if(!_inst || !_main || _inst == _main)
            return;
        auto_lock_t          _lk{ type_mutex<hash_alias_map_t>(), std::defer_lock };
        if(!_lk.owns_lock())
            _lk.lock();
        for(const auto& itr : *_inst)
        {
            if(_main->find(itr.first) == _main->end())
                _main->emplace(itr.first, itr.second);
        }
    } };
    return _inst;
    (void) _dtor;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_alias_ptr_t&)
get_main_hash_aliases()
{
    static thread_local auto _inst =
        get_shared_ptr_pair_main_instance<hash_alias_map_t, TIMEMORY_API>();
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(std::shared_ptr<hash_resolver_vec_t>&)
get_hash_resolvers()
{
    static auto _inst = []() {
        auto _subinst = get_shared_ptr_lone_instance<hash_resolver_vec_t, TIMEMORY_API>();
        if(_subinst && _subinst->empty())
            _subinst->reserve(10);
        return _subinst;
    }();
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_value_t)
get_hash_id(const hash_alias_ptr_t& _hash_alias, hash_value_t _hash_id)
{
    auto _alias_itr = _hash_alias->find(_hash_id);
    if(_alias_itr != _hash_alias->end())
        return _alias_itr->second;
    return _hash_id;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_value_t)
add_hash_id(hash_map_ptr_t& _hash_map, string_view_cref_t _prefix)
{
    hash_value_t _hash_id = std::hash<std::string_view>{}(_prefix);
    if(_hash_map && _hash_map->count(_hash_id) == 0)
        _hash_map->emplace(_hash_id, std::string{ _prefix });
    return _hash_id;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(void)
add_hash_id(const hash_map_ptr_t&, const hash_alias_ptr_t& _hash_alias,
            hash_value_t _hash_id, hash_value_t _alias_hash_id)
{
    _hash_alias->emplace(_alias_hash_id, _hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(void)
add_hash_id(const hash_alias_ptr_t& _hash_alias, hash_value_t _hash_id,
            hash_value_t _alias_hash_id)
{
    _hash_alias->emplace(_alias_hash_id, _hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(void)
add_hash_id(hash_value_t _hash_id, hash_value_t _alias_hash_id)
{
    add_hash_id(get_hash_aliases(), _hash_id, _alias_hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(void)
hash_identifier_error(const hash_map_ptr_t&   _hash_map,
                      const hash_alias_ptr_t& _hash_alias, hash_value_t _hash_id)
{
    static thread_local std::set<hash_value_t> _reported{};
    if(_reported.count(_hash_id) > 0)
        return;

    _reported.insert(_hash_id);

    if(!_hash_map)
    {
        fprintf(stderr,
                "[%s@%s:%i]> hash identifier %llu could not be found bc the pointer to "
                "the hash map is null\n",
                __FUNCTION__, TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(), __LINE__,
                (unsigned long long) _hash_id);
        return;
    }

    if(!_hash_alias)
    {
        fprintf(stderr,
                "[%s@%s:%i]> hash identifier %llu could not be found bc the pointer to "
                "the hash alias map is null\n",
                __FUNCTION__, TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(), __LINE__,
                (unsigned long long) _hash_id);
        return;
    }

    for(const auto& aitr : *_hash_alias)
    {
        if(_hash_id == aitr.first)
        {
            for(const auto& mitr : *_hash_map)
            {
                if(mitr.first == aitr.second)
                {
                    fprintf(stderr,
                            "[%s@%s:%i]> found hash identifier %llu in alias map via "
                            "iteration after uomap->find failed! This might be an ABI or "
                            "an integer overflow problem\n",
                            __FUNCTION__,
                            TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(), __LINE__,
                            (unsigned long long) _hash_id);
                }
            }
        }
    }

    for(auto& mitr : *_hash_map)
    {
        if(_hash_id == mitr.first)
        {
            fprintf(stderr,
                    "[%s@%s:%i]> found hash identifier %llu in hash map via iteration "
                    "after uomap->find failed! This might be an ABI or an integer "
                    "overflow problem\n",
                    __FUNCTION__, TIMEMORY_TRUNCATED_FILE_STRING(__FILE__).c_str(),
                    __LINE__, (unsigned long long) _hash_id);
        }
    }

    if(_hash_id > 0)
    {
        std::stringstream ss;
        ss << "Error! node with hash " << _hash_id
           << " does not have an associated string!\n";
        if(_reported.count(_hash_id) == 0)
        {
            _reported.emplace(_hash_id);
            bool _found_direct = (_hash_map->find(_hash_id) != _hash_map->end());
            ss << "    Found in map       : " << std::boolalpha << _found_direct << '\n';
            bool _found_alias = (_hash_alias->find(_hash_id) != _hash_alias->end());
            ss << "    Found in alias map : " << std::boolalpha << _found_alias << '\n';
            if(_found_alias)
            {
                auto aitr = _hash_alias->find(_hash_id);
                ss << "    Found aliasing : " << aitr->first << " -> " << aitr->second
                   << '\n';
                auto mitr = _hash_map->find(aitr->second);
                if(mitr != _hash_map->end())
                    ss << "    Found mapping  : " << mitr->first << " -> " << mitr->second
                       << '\n';
                else
                    ss << "    Missing mapping\n";
            }
            else
            {
                ss << "    Missing aliasing\n";
            }
            ss << "    Hash map:\n";
            auto _w = 20;
            for(const auto& itr : *_hash_map)
                ss << "        " << std::setw(_w) << itr.first << " : " << (itr.second)
                   << "\n";
            if(!_hash_alias->empty())
            {
                ss << "    Alias hash map:\n";
                for(const auto& itr : *_hash_alias)
                    ss << "        " << std::setw(_w) << itr.first << " : " << itr.second
                       << "\n";
            }
            auto _registry = static_string::get_registry();
            if(!_registry.empty())
            {
                ss << "    Static strings:\n";
                for(const auto* itr : _registry)
                {
                    ss << "        " << std::setw(_w)
                       << reinterpret_cast<std::size_t>(itr) << " : " << itr << "\n";
                }
            }
            fprintf(stderr, "%s", ss.str().c_str());
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(typename hash_map_t::const_iterator)
find_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                     hash_value_t _hash_id)
{
    auto _map_itr = _hash_map->find(_hash_id);
    if(_map_itr != _hash_map->end())
        return _map_itr;

    auto _alias_itr = _hash_alias->find(_hash_id);
    if(_alias_itr != _hash_alias->end())
    {
        return find_hash_identifier(_hash_map, _hash_alias, _alias_itr->second);
    }

    return _hash_map->end();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(typename hash_map_t::const_iterator)
find_hash_identifier(hash_value_t _hash_id)
{
    return find_hash_identifier(get_hash_ids(), get_hash_aliases(), _hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(bool)
get_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                    hash_value_t _hash_id, std::string*& _ret)
{
    // NOTE: for brevity, all statements returning true use comma operator to assign to
    // result before returning true
    if(!_hash_map)
        return false;

    auto _map_itr = _hash_map->find(_hash_id);
    if(_map_itr != _hash_map->end())
        return (_ret = &_map_itr->second, true);

    // static_string
    if(static_string::is_registered(_hash_id))
    {
        auto itr = _hash_map->emplace(_hash_id, reinterpret_cast<const char*>(_hash_id));
        return (_ret = &itr.first->second, true);
    }

    if(_hash_alias)
    {
        auto _alias_itr = _hash_alias->find(_hash_id);
        if(_alias_itr != _hash_alias->end())
        {
            return get_hash_identifier(_hash_map, _hash_alias, _alias_itr->second, _ret);
        }
    }

    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(bool)
get_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                    hash_value_t _hash_id, const char*& _ret)
{
    // NOTE: for brevity, all statements returning true use comma operator to assign to
    // result before returning true
    if(!_hash_map)
        return false;

    auto _map_itr = _hash_map->find(_hash_id);
    if(_map_itr != _hash_map->end())
        return (_ret = _map_itr->second.c_str(), true);

    // static_string
    if(static_string::is_registered(_hash_id))
        return (_ret = reinterpret_cast<const char*>(_hash_id), true);

    if(_hash_alias)
    {
        auto _alias_itr = _hash_alias->find(_hash_id);
        if(_alias_itr != _hash_alias->end())
        {
            return get_hash_identifier(_hash_map, _hash_alias, _alias_itr->second, _ret);
        }
    }

    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(std::string)
get_hash_identifier(hash_value_t _hash_id)
{
    std::string* _ret = nullptr;
    if(get_hash_identifier(get_hash_ids(), get_hash_aliases(), _hash_id, _ret))
        return *_ret;

    // search the "main" (global) hash id and alias maps
    // these are a shared resource so locking is necessary
    {
        auto_lock_t _mlk{ type_mutex<hash_map_t>(), std::defer_lock };
        auto_lock_t _alk{ type_mutex<hash_alias_map_t>(), std::defer_lock };
        if(!_mlk.owns_lock())
            _mlk.lock();
        if(!_alk.owns_lock())
            _alk.lock();
        if(get_hash_identifier(get_main_hash_ids(), get_main_hash_aliases(), _hash_id,
                               _ret))
            return *_ret;
    }

    hash_identifier_error(get_hash_ids(), get_hash_aliases(), _hash_id);
    return std::string("unknown-hash=") + std::to_string(_hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(bool)
get_hash_identifier(hash_value_t _hash_id, std::string*& _ret)
{
    return get_hash_identifier(get_hash_ids(), get_hash_aliases(), _hash_id, _ret);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(bool)
get_hash_identifier(hash_value_t _hash_id, const char*& _ret)
{
    return get_hash_identifier(get_hash_ids(), get_hash_aliases(), _hash_id, _ret);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(std::string)
get_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                    hash_value_t _hash_id)
{
    std::string* _ret = nullptr;
    if(get_hash_identifier(_hash_map, _hash_alias, _hash_id, _ret))
        return *_ret;
    else if(get_hash_identifier(get_main_hash_ids(), get_main_hash_aliases(), _hash_id,
                                _ret))
        return *_ret;
    hash_identifier_error(_hash_map, _hash_alias, _hash_id);
    return std::string("unknown-hash=") + std::to_string(_hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(string_view_t)
get_hash_identifier_fast(hash_value_t _hash)
{
    auto& _hash_ids = get_hash_ids();
    if(_hash_ids)
    {
        auto itr = _hash_ids->find(_hash);
        if(itr == _hash_ids->end())
            return string_view_t{};
        return _hash_ids->at(_hash);
    }
    return string_view_t{};
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(bool)
get_hash_identifier_fast(hash_value_t _hash, std::string*& _v)
{
    auto& _hash_ids = get_hash_ids();
    if(_hash_ids)
    {
        auto itr = _hash_ids->find(_hash);
        if(itr == _hash_ids->end())
            return false;
        return (_v = &_hash_ids->at(_hash), true);
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(bool)
get_hash_identifier_fast(hash_value_t _hash, const char*& _v)
{
    auto& _hash_ids = get_hash_ids();
    if(_hash_ids)
    {
        auto itr = _hash_ids->find(_hash);
        if(itr != _hash_ids->end())
            return (_v = itr->second.c_str(), true);
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(std::string)
demangle_hash_identifier(std::string inp, char bdelim, char edelim)
{
    inp         = demangle(inp);
    size_t _beg = inp.find_first_of(bdelim);
    while(_beg != std::string::npos)
    {
        size_t _end = inp.find_first_of(edelim, _beg);
        if(_end == std::string::npos || _end <= _beg)
            break;
        auto _sz = _end - _beg - 1;
        inp      = inp.replace(_beg + 1, _sz, demangle(inp.substr(_beg + 1, _sz)));
        _beg     = inp.find_first_of(bdelim, _end + 1);
    }
    return demangle(inp);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace hash
}  // namespace tim

#endif
