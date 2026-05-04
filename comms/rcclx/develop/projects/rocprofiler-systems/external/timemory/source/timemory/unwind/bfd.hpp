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
#include "timemory/log/macros.hpp"
#include "timemory/unwind/types.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tim
{
namespace unwind
{
int
bfd_error(const char* string);

int
bfd_message(int _lvl, std::string_view);

void
set_bfd_verbose(int);

enum class bfd_binding : uint8_t
{
    Local = 0,
    Global,
    Weak,
    Unknown
};

enum class bfd_visibility : uint8_t
{
    Default = 0,
    Internal,
    Hidden,
    Protected,
    Unknown
};

#if defined(TIMEMORY_USE_BFD)

struct bfd_file
{
    struct symbol
    {
        bfd_binding      binding    = bfd_binding::Unknown;
        bfd_visibility   visibility = bfd_visibility::Unknown;
        uintptr_t        address    = 0;
        uint64_t         symsize    = 0;
        void*            section    = nullptr;
        std::string_view name       = {};
    };

    explicit bfd_file(std::string);
    ~bfd_file();

    static void* open(const std::string&, int* _fd = nullptr);

    int  read_symtab();
    bool is_good() const { return (data != nullptr) && !name.empty(); }

    explicit operator bool() const { return is_good(); }

    std::vector<symbol> get_symbols(bool _include_undefined = false) const;

    int         fd    = -1;
    int64_t     nsyms = 0;
    std::string name  = {};
    void*       data  = nullptr;
    void**      syms  = nullptr;
};

#else

struct bfd_file
{
    struct symbol
    {
        bfd_binding      binding    = bfd_binding::Unknown;
        bfd_visibility   visibility = bfd_visibility::Unknown;
        uintptr_t        address    = 0;
        uint64_t         symsize    = 0;
        void*            section    = nullptr;
        std::string_view name       = {};
    };

    explicit bfd_file(std::string) {}
    ~bfd_file() = default;

    static void* open(const std::string&) { return nullptr; }
    static int   read_symtab() { return -1; }
    static bool  is_good() { return false; }
    static auto  get_symbols(bool _include_undefined = false);

    explicit operator bool() const { return false; }

    int              fd    = -1;
    int64_t          nsyms = 0;
    std::string_view name  = {};
    void*            data  = nullptr;
    void**           syms  = nullptr;
};

inline auto
bfd_file::get_symbols(bool)
{
    return std::vector<symbol>{};
}
#endif
}  // namespace unwind
}  // namespace tim
