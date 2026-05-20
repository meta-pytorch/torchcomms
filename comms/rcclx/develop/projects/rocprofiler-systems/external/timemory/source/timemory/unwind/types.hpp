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
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(TIMEMORY_USE_LIBUNWIND)
#    include <libunwind.h>
#    if defined(unw_get_proc_name_by_ip)
#        define TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP 1
#    else
#        define TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP 0
#    endif
#endif

namespace tim
{
namespace unwind
{
struct entry;
struct processed_entry;
struct bfd_file;
struct addr2line_info;
struct cache;
struct dlinfo;

template <size_t N>
struct stack;

template <size_t LhsN, size_t RhsN>
auto get_common_stack(stack<LhsN>, stack<RhsN>);

addr2line_info
addr2line(std::shared_ptr<bfd_file>, const std::vector<uint64_t>&);

addr2line_info
addr2line(std::shared_ptr<bfd_file>, unsigned long);

addr2line_info
addr2line(const std::string&, unsigned long);
}  // namespace unwind
}  // namespace tim
