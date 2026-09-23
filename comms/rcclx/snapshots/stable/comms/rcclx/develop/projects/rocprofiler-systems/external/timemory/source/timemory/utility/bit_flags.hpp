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

#include "timemory/macros/attributes.hpp"

#include <cstdint>
#include <type_traits>

namespace tim
{
namespace utility
{
template <size_t Nopts>
struct bit_flags
{
    static_assert(Nopts <= 64, "Error! bit_flags does not support more than 64 options");
    using base_type  = bit_flags<Nopts>;
    using value_type = std::conditional_t<
        (Nopts <= 8), uint8_t,
        std::conditional_t<(Nopts <= 16), uint16_t,
                           std::conditional_t<(Nopts <= 32), uint32_t, uint64_t>>>;

    template <size_t Idx>
    TIMEMORY_INLINE bool test() const;
    TIMEMORY_INLINE bool test(value_type idx) const;

    template <size_t Idx>
    TIMEMORY_INLINE void set(bool v);
    TIMEMORY_INLINE void set(value_type idx, bool v);

    TIMEMORY_INLINE void       reset() { m_state_value = 0; }
    TIMEMORY_INLINE void       set_state_value(value_type v) { m_state_value = v; }
    TIMEMORY_INLINE value_type get_state_value() const { return m_state_value; }

private:
    template <size_t Idx>
    static constexpr value_type index();
    static value_type           index(value_type idx);

    value_type m_state_value{ 0 };
};
//
template <size_t Nopts>
template <size_t Idx>
bool
bit_flags<Nopts>::test() const
{
    return (m_state_value & index<Idx>());
}
//
template <size_t Nopts>
bool
bit_flags<Nopts>::test(value_type idx) const
{
    return (m_state_value & index(idx));
}
//
template <size_t Nopts>
template <size_t Idx>
void
bit_flags<Nopts>::set(bool v)
{
    bool _curr = test<Idx>();
    if(_curr != v)
    {
        if(!_curr)
            m_state_value |= index<Idx>();
        else
            m_state_value &= (~index<Idx>());
    }
}
//
template <size_t Nopts>
void
bit_flags<Nopts>::set(value_type idx, bool v)
{
    bool _curr = test(idx);
    if(_curr != v)
    {
        if(!_curr)
            m_state_value |= index(idx);
        else
            m_state_value &= (~index(idx));
    }
}
//
template <size_t Nopts>
template <size_t Idx>
constexpr typename bit_flags<Nopts>::value_type
bit_flags<Nopts>::index()
{
    return 1 << Idx;
}
//
template <size_t Nopts>
typename bit_flags<Nopts>::value_type
bit_flags<Nopts>::index(value_type idx)
{
    return 1 << idx;
}
//
}  // namespace utility
}  // namespace tim
