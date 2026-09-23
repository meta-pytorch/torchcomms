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
#include "timemory/unwind/bfd.hpp"
#include "timemory/unwind/entry.hpp"
#include "timemory/unwind/processed_entry.hpp"
#include "timemory/unwind/types.hpp"
#include "timemory/utility/macros.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace std
{
template <>
struct hash<tim::unwind::entry>
{
    size_t operator()(tim::unwind::entry _v) const
    {
        return std::hash<unw_word_t>{}(_v.address());
    }
};
}  // namespace std

namespace tim
{
namespace unwind
{
struct cache
{
    using entry_map_t = std::unordered_map<entry, processed_entry>;
    using file_map_t  = std::unordered_map<std::string, std::shared_ptr<bfd_file>>;

    TIMEMORY_DEFAULT_OBJECT(cache)

    explicit cache(bool _use_files)
    : use_files{ _use_files }
    {}

    bool        use_files = true;
    entry_map_t entries   = {};
    file_map_t  files     = {};
};
}  // namespace unwind
}  // namespace tim
