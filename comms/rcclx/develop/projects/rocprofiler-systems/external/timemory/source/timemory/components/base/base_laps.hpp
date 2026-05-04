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

#include "timemory/components/base/types.hpp"
#include "timemory/macros/attributes.hpp"

#include <cstdint>

namespace tim
{
namespace component
{
/// \struct base_laps
/// \brief Tracks the lap counter of a component
///
template <typename Tp>
struct base_laps
{
    TIMEMORY_DEFAULT_OBJECT(base_laps)

    auto get_laps() const { return laps; }
    auto set_laps(int64_t v) { laps = v; }

    auto increment_laps(int64_t _v = 1) { return (laps += _v); }
    void reset_laps() { set_laps(0); }
    void reset() { set_laps(0); }

    auto operator++() { return ++laps; }
    auto operator++(int) { return laps++; }

    auto operator--() { return --laps; }
    auto operator--(int) { return laps--; }

    auto operator+=(int64_t _v) { return (laps += _v); }
    auto operator-=(int64_t _v) { return (laps -= _v); }

    base_laps& operator+=(base_laps _rhs)
    {
        laps += _rhs.laps;
        return *this;
    }

    base_laps& operator-=(base_laps _rhs)
    {
        laps -= _rhs.laps;
        return *this;
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int = 0)
    {
        ar(cereal::make_nvp("laps", laps));
    }

protected:
    int64_t laps = 0;
};
//
template <>
struct base_laps<null_type>
{
    TIMEMORY_DEFAULT_OBJECT(base_laps)

    int64_t laps() const { return 0; }
    int64_t get_laps() const { return 0; }
    int64_t set_laps(int64_t) { return 0; }

    int64_t increment_laps(int64_t = 1) { return 0; }
    void    reset_laps() {}
    void    reset() {}

    int64_t operator++() { return 0; }
    int64_t operator++(int) { return 0; }

    int64_t operator--() { return 0; }
    int64_t operator--(int) { return 0; }

    int64_t operator+=(int64_t) { return 0; }
    int64_t operator-=(int64_t) { return 0; }

    base_laps& operator+=(base_laps) { return *this; }
    base_laps& operator-=(base_laps) { return *this; }

    template <typename ArchiveT>
    void serialize(ArchiveT&, const unsigned int = 0)
    {}
};
}  // namespace component
}  // namespace tim
