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

#include "timemory/hash/declaration.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/delimit.hpp"

#include <string>
#include <utility>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::decode
/// \tparam ApiT Timemory project API, e.g. \ref tim::project::timemory
///
/// \brief This class post-processes strings for a given API
///
//
//--------------------------------------------------------------------------------------//
//
template <typename ApiT>
struct decode
{
    TIMEMORY_DEFAULT_OBJECT(decode)

    static auto tokenized_demangle(std::string inp)
    {
        using pair_t = std::pair<std::string, std::string>;
        for(auto&& itr : { pair_t{ "_Z", " " }, pair_t{ "_Z", "]" }, pair_t{ " ", " " } })
        {
            inp = str_transform(inp, itr.first, itr.second,
                                [](const std::string& _s) { return demangle(_s); });
        }
        return inp;
    }

    auto operator()(const char* inp)
    {
        return tokenized_demangle(demangle_backtrace(demangle_hash_identifier(inp)));
    }

    auto operator()(const std::string& inp)
    {
        return tokenized_demangle(demangle_backtrace(demangle_hash_identifier(inp)));
    }

    auto operator()(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                    hash_value_t _hash_id)
    {
        auto        _resolvers = *get_hash_resolvers();
        std::string _resolved{};
        for(auto& itr : _resolvers)
        {
            if(itr(_hash_id, _resolved))
                return tokenized_demangle(demangle_hash_identifier(_resolved));
            return _resolved;
        }

        return tokenized_demangle(demangle_hash_identifier(
            get_hash_identifier(_hash_map, _hash_alias, _hash_id)));
    }

    auto operator()(hash_value_t _hash_id)
    {
        auto        _resolvers = *get_hash_resolvers();
        std::string _resolved{};
        for(auto& itr : _resolvers)
        {
            if(itr(_hash_id, _resolved))
                return tokenized_demangle(demangle_hash_identifier(_resolved));
            return _resolved;
        }

        return tokenized_demangle(
            demangle_hash_identifier(get_hash_identifier(_hash_id)));
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
