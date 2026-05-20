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
#include "timemory/tpls/cereal/cereal/cereal.hpp"
#include "timemory/unwind/types.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#if defined(TIMEMORY_USE_LIBUNWIND)
#    include <libunwind.h>
#endif

#if defined(TIMEMORY_UNIX)
#    include <dlfcn.h>
#endif

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace tim
{
namespace unwind
{
#if !defined(TIMEMORY_USE_LIBUNWIND)
using unw_word_t = uint64_t;
#endif

struct dlinfo
{
    struct data
    {
        std::string_view name = {};
        void*            addr = nullptr;

        template <typename Tp = uintptr_t>
        Tp address() const;

        template <typename ArchiveT>
        void save(ArchiveT& ar, const unsigned) const;

        template <typename ArchiveT>
        void load(ArchiveT& ar, const unsigned);

        explicit operator bool() const { return (!name.empty() && addr != nullptr); }
    };

    static dlinfo construct(unw_word_t _ip);
    static dlinfo construct(unw_word_t _start, unw_word_t _offset);

    TIMEMORY_DEFAULT_OBJECT(dlinfo)

#if defined(TIMEMORY_UNIX)
    dlinfo(Dl_info);
#endif

    data location = {};
    data symbol   = {};

    explicit operator bool() const { return (symbol && location); }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned);
};

template <typename ArchiveT>
void
dlinfo::serialize(ArchiveT& ar, const unsigned)
{
    ar(cereal::make_nvp("location", location), cereal::make_nvp("symbol", symbol));
}

template <typename Tp>
Tp
dlinfo::data::address() const
{
    return reinterpret_cast<Tp>(addr);
}

template <typename ArchiveT>
void
dlinfo::data::save(ArchiveT& ar, const unsigned) const
{
    ar(cereal::make_nvp("name", std::string{ name }),
       cereal::make_nvp("address", address()));
}

template <typename ArchiveT>
void
dlinfo::data::load(ArchiveT& ar, const unsigned)
{
    auto*     _name = new std::string{};
    uintptr_t _addr = 0;
    ar(cereal::make_nvp("name", *_name), cereal::make_nvp("address", _addr));
    name = *_name;
    addr = reinterpret_cast<void*>(_addr);
}
}  // namespace unwind
}  // namespace tim
