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

#include "timemory/defines.h"
#include "timemory/hash/types.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/tpls/cereal/cereal/cereal.hpp"
#include "timemory/unwind/bfd.hpp"
#include "timemory/unwind/cache.hpp"
#include "timemory/unwind/common.hpp"
#include "timemory/unwind/entry.hpp"
#include "timemory/unwind/processed_entry.hpp"
#include "timemory/unwind/types.hpp"
#include "timemory/utility/optional.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tim
{
namespace unwind
{
template <size_t N>
struct stack
{
    using array_type     = std::array<stl::optional<entry>, N>;
    using iterator       = typename array_type::iterator;
    using const_iterator = typename array_type::const_iterator;
    using cache_type     = cache;
    using file_map_t     = std::unordered_map<std::string, std::shared_ptr<bfd_file>>;

    TIMEMORY_DEFAULT_OBJECT(stack)

    explicit stack(unw_frame_regnum_t _regnum)
    : regnum{ _regnum }
    {}

    template <size_t RhsN, std::enable_if_t<N != RhsN, int> = 0>
    stack& operator=(stack<RhsN> _rhs);

    auto&       operator[](size_t _idx) { return call_stack[_idx]; }
    const auto& operator[](size_t _idx) const { return call_stack[_idx]; }

    auto&       at(size_t _idx) { return call_stack.at(_idx); }
    const auto& at(size_t _idx) const { return call_stack.at(_idx); }

    size_t       size() const;
    bool         valid() const { return size() > 0; }
    bool         empty() const { return size() == 0; }
    hash_value_t hash(hash_value_t = 1) const;

    template <typename Tp>
    auto& emplace_back(Tp&&);

    // shift the call-stack
    stack& shift(int64_t _n = -1);

    iterator       begin() { return call_stack.begin(); }
    const_iterator begin() const { return call_stack.begin(); }

    iterator       end();
    const_iterator end() const;

    unw_frame_regnum_t regnum     = {};
    unw_cursor_t       cursor     = {};
    unw_context_t      context    = {};
    array_type         call_stack = {};

    template <size_t DefaultBufferSize = 4096, bool Shrink = true>
    std::vector<processed_entry> get(cache_type* _cache              = nullptr,
                                     bool        _include_with_error = false,
                                     file_map_t* _files              = nullptr) const;

    template <size_t DefaultBufferSize = 4096, bool Shrink = true>
    std::vector<processed_entry> get(bool        _include_with_error,
                                     file_map_t* _files = nullptr) const
    {
        return get<DefaultBufferSize, Shrink>(nullptr, _include_with_error, _files);
    }

    template <size_t RhsN>
    bool operator==(stack<RhsN> _rhs) const;

    template <size_t RhsN>
    bool operator!=(stack<RhsN> _v) const;

    template <typename ArchiveT>
    void save(ArchiveT& ar, const unsigned) const;

    template <typename ArchiveT>
    void load(ArchiveT& ar, const unsigned);

    static hash_value_t hash(stack<N> _v, hash_value_t _init = 1);

private:
    template <size_t... Idx>
    static hash_value_t compute_hash(stack<N>     _v, std::index_sequence<Idx...>,
                                     hash_value_t _init = 1);
};
//
template <size_t N>
inline size_t
stack<N>::size() const
{
    size_t _n = 0;
    for(auto&& itr : call_stack)
        _n += (itr) ? 1 : 0;
    return _n;
}
//
template <size_t N>
inline hash_value_t
stack<N>::hash(hash_value_t _init) const
{
    return compute_hash(*this, std::make_index_sequence<N>{}, _init);
}
//
template <size_t N>
inline typename stack<N>::iterator
stack<N>::end()
{
    auto itr = call_stack.begin();
    std::advance(itr, size());
    return itr;
}
//
template <size_t N>
inline typename stack<N>::const_iterator
stack<N>::end() const
{
    auto itr = call_stack.begin();
    std::advance(itr, size());
    return itr;
}
//
template <size_t N>
template <typename Tp>
inline auto&
stack<N>::emplace_back(Tp&& _v)
{
    size_t _n = size();
    if(_n < N)
    {
        call_stack.at(_n) = std::forward<Tp>(_v);
        return call_stack.at(_n);
    }
    return call_stack.back();
}
//
template <size_t N>
inline stack<N>&
stack<N>::shift(int64_t _n)
{
    using entry_t = stl::optional<entry>;
    std::vector<entry_t> _stack{};
    int64_t              _size = size();
    _stack.reserve(_size);
    for(auto itr : *this)
        _stack.emplace_back(itr);
    call_stack.fill(entry_t{});

    for(int64_t i = 0; i < _size; ++i)
    {
        int64_t _idx = i + _n;
        if(_idx >= 0 && _idx < static_cast<int64_t>(N))
        {
            call_stack.at(_idx) = _stack.at(i);
        }
    }
    return *this;
}
//
template <size_t N>
template <size_t DefaultBufferSize, bool Shrink>
std::vector<processed_entry>
stack<N>::get(cache_type* _cache, bool _include_with_error, file_map_t* _files) const
{
    if(!_files && _cache && _cache->use_files)
        _files = &_cache->files;

    std::vector<processed_entry> _data{};
    auto                         _sz = size();
    if(_sz > 0)
        _data.reserve(_sz);
    for(auto itr : *this)
    {
        if(itr)
        {
            if(_cache)
            {
                auto citr = _cache->entries.find(*itr);
                if(citr != _cache->entries.end())
                {
                    if(citr->second.error == 0 || _include_with_error)
                        _data.emplace_back(citr->second);
                    continue;
                }
            }

            processed_entry _v{};
            _v.address = itr->address();
            _v.name    = itr->template get_name<DefaultBufferSize, Shrink>(
                context, &_v.offset, &_v.error);

            processed_entry::construct(_v, _files);

            if(_v.error == 0 || _include_with_error)
                _data.emplace_back(_v);

            if(_cache)
                _cache->entries.emplace(*itr, _v);
        }
    }
    return _data;
}
//
template <size_t N>
template <size_t RhsN, std::enable_if_t<N != RhsN, int>>
inline stack<N>&
stack<N>::operator=(stack<RhsN> _rhs)
{
    static_assert(N != RhsN, "Error! Bad overload resolution");
    regnum  = _rhs.regnum;
    cursor  = _rhs.cursor;
    context = _rhs.context;
    for(size_t i = 0; i < std::min<size_t>(N, RhsN); ++i)
        call_stack[i] = _rhs.call_stack[i];
    return *this;
}
//
template <size_t N>
template <size_t RhsN>
inline bool
stack<N>::operator==(stack<RhsN> _rhs) const
{
    if constexpr(N == RhsN)
    {
        constexpr size_t LhsN = N;
        const auto&      _lhs = *this;
        for(size_t i = 0; i < LhsN; ++i)
        {
            size_t _n = ((_lhs[i]) ? 1 : 0) + ((_rhs[i]) ? 1 : 0);
            // if one has an entry and the other doesn't, not equal
            if(_n == 1)
                return false;
            else if(_n == 2)
            {
                if(_lhs[i]->address() != _rhs[i]->address())
                    return false;
            }
        }
        return true;
    }
    else
    {
        auto _common = get_common_stack(*this, _rhs);
        return (std::get<0>(_common) == std::get<1>(_common));
    }
}
//
template <size_t N>
template <size_t RhsN>
inline bool
stack<N>::operator!=(stack<RhsN> _v) const
{
    return !(*this == _v);
}
//
template <size_t N>
template <typename ArchiveT>
void
stack<N>::save(ArchiveT& ar, const unsigned) const
{
    auto _data = get();
    ar(cereal::make_nvp("regnum", regnum), cereal::make_nvp("data", _data));
}
//
template <size_t N>
template <typename ArchiveT>
void
stack<N>::load(ArchiveT& ar, const unsigned)
{
    auto _data = std::vector<processed_entry>{};
    ar(cereal::make_nvp("regnum", regnum), cereal::make_nvp("data", _data));
    size_t _idx = 0;
    for(const auto& itr : _data)
        call_stack.at(_idx++) = entry{ itr.address };
}
//
template <size_t N>
inline hash_value_t
stack<N>::hash(stack<N> _v, hash_value_t _init)
{
    return compute_hash(std::move(_v), std::make_index_sequence<N>{}, _init);
}
//
template <size_t N>
template <size_t... Idx>
inline hash_value_t
stack<N>::compute_hash(stack<N> _v, std::index_sequence<Idx...>, hash_value_t _init)
{
    static_assert(sizeof...(Idx) <= N, "Error! Index sequence exceeds size");
    static_assert(std::is_trivially_copyable<stack<N>>::value,
                  "Error! Not trivially copyable");
    auto _access = [](auto&& _e) { return (_e) ? _e->address() : 0; };
    return get_hash_id(_init, TIMEMORY_FOLD_EXPRESSION(_access(_v[Idx])));
}
}  // namespace unwind
}  // namespace tim
