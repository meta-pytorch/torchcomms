// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "timemory/log/macros.hpp"
#include "timemory/unwind/bfd.hpp"
#include "timemory/unwind/types.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tim
{
namespace unwind
{
struct addr2line_info
{
    struct lineinfo;
    using callback_t = bool (*)(const lineinfo&);

    explicit addr2line_info(std::shared_ptr<bfd_file>);

    TIMEMORY_DEFAULT_OBJECT(addr2line_info)

    explicit operator bool() const;

    struct lineinfo
    {
        bool         inlined  = false;
        unsigned int line     = 0;
        std::string  name     = {};
        std::string  location = {};

        explicit operator bool() const;
    };

    void add_lineinfo(const char* _func, const char* _file, unsigned int _line,
                      bool _inlined);

    /// default behavior returns the first non-inlined lineinfo instance
    lineinfo get(callback_t _cb = nullptr) const;

    /// default behavior returns the last non-inlined lineinfo instance
    lineinfo rget(callback_t _cb = nullptr) const;

    bool                      found         = false;
    unsigned int              discriminator = 0;
    unsigned long             address       = 0;
    std::vector<lineinfo>     lines         = {};  // include inlines
    std::shared_ptr<bfd_file> input         = {};
};

inline addr2line_info::lineinfo::operator bool() const
{
    return (!name.empty() || !location.empty());
}
}  // namespace unwind
}  // namespace tim
