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

#include "timemory/tpls/cereal/cereal/cereal.hpp"
#include "timemory/unwind/addr2line.hpp"
#include "timemory/unwind/bfd.hpp"
#include "timemory/unwind/dlinfo.hpp"
#include "timemory/unwind/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>

namespace tim
{
namespace unwind
{
struct processed_entry
{
    using file_map_t = std::unordered_map<std::string, std::shared_ptr<bfd_file>>;

    int            error        = 0;
    unsigned int   lineno       = 0;
    uintptr_t      address      = 0;
    uintptr_t      offset       = 0;
    uintptr_t      line_address = 0;   // line address in file
    std::string    name         = {};  // function name
    std::string    location     = {};  // file location
    dlinfo         info         = {};  // dynamic library info
    addr2line_info lineinfo     = {};  // address-to-line info

    static void construct(processed_entry&, file_map_t* = nullptr,
                          bool _prefer_dlinfo = false);

    bool operator==(const processed_entry& _v) const;
    bool operator<(const processed_entry& _v) const;
    bool operator>(const processed_entry& _v) const;
    bool operator!=(const processed_entry& _v) const { return !(*this == _v); }
    bool operator<=(const processed_entry& _v) const { return !(*this > _v); }
    bool operator>=(const processed_entry& _v) const { return !(*this < _v); }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        ar(cereal::make_nvp("error", error), cereal::make_nvp("address", address),
           cereal::make_nvp("offset", offset),
           cereal::make_nvp("line_address", line_address), cereal::make_nvp("name", name),
           cereal::make_nvp("location", location), cereal::make_nvp("lineno", lineno),
           cereal::make_nvp("dlinfo", info));
    }
};
//
inline bool
processed_entry::operator==(const processed_entry& _v) const
{
    return std::tie(error, address, offset, lineno, name, location) ==
           std::tie(_v.error, _v.address, _v.offset, _v.lineno, _v.name, _v.location);
}

inline bool
processed_entry::operator<(const processed_entry& _v) const
{
    return std::tie(name, location, lineno, offset, address, error) <
           std::tie(_v.name, _v.location, _v.lineno, _v.offset, _v.address, _v.error);
}

inline bool
processed_entry::operator>(const processed_entry& _v) const
{
    return !(*this == _v && *this < _v);
}
}  // namespace unwind
}  // namespace tim
