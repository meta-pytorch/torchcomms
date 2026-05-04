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

#ifndef TIMEMORY_UNWIND_BACKTRACE_HPP_
#    define TIMEMORY_UNWIND_BACKTRACE_HPP_
#endif

#include "timemory/defines.h"
#include "timemory/macros/attributes.hpp"
#include "timemory/unwind/macros.hpp"

#include <cstdlib>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

namespace tim
{
namespace unwind
{
struct bfd_file;

using file_map_t = std::unordered_map<std::string, std::shared_ptr<bfd_file>>;

file_map_t&
library_file_maps();

void
update_file_maps();

struct detailed_backtrace_config
{
    bool force_color      = false;
    bool native_mangled   = true;
    bool native_demangled = true;
    bool proc_pid_maps    = true;
    bool unwind_demangled = true;
    bool unwind_lineinfo  = true;
};

template <size_t OffsetV, size_t DepthV = 64>
void
detailed_backtrace(std::ostream&             os,
                   detailed_backtrace_config cfg = {}) TIMEMORY_INTERNAL;

// only instantiates offsets 0-3 and depths 8, 16, 32, 64
template <size_t OffsetV, size_t DepthV = 64>
void
detailed_backtrace(std::ostream& os, bool&& force_color,
                   detailed_backtrace_config cfg = {}) TIMEMORY_INTERNAL;
}  // namespace unwind
}  // namespace tim

#if defined(TIMEMORY_UNWIND_HEADER_MODE) && TIMEMORY_UNWIND_HEADER_MODE > 0
#    include "timemory/unwind/backtrace.cpp"
#endif
