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
#include "timemory/mpl/concepts.hpp"  // component::gotcha forward decl

#include <type_traits>

namespace tim
{
namespace component
{
//
class gotcha_suppression
{
private:
    template <size_t Nt, typename Components, typename Differentiator>
    friend struct gotcha;

    static TIMEMORY_NOINLINE bool& get()
    {
        static thread_local bool _instance = false;
        return _instance;
    }

public:
    struct auto_toggle
    {
        explicit auto_toggle(bool& _value, bool _if_equal = false);
        auto_toggle(std::false_type);
        auto_toggle(std::true_type);
        ~auto_toggle();
        auto_toggle(const auto_toggle&) = delete;
        auto_toggle(auto_toggle&&)      = delete;
        auto_toggle& operator=(const auto_toggle&) = delete;
        auto_toggle& operator=(auto_toggle&&) = delete;

    private:
        bool& m_value;
        bool  m_if_equal;
        bool  m_did_toggle = false;
    };
};
//
inline gotcha_suppression::auto_toggle::auto_toggle(bool& _value, bool _if_equal)
: m_value{ _value }
, m_if_equal{ _if_equal }
{
    if(m_value == m_if_equal)
    {
        m_value      = !m_value;
        m_did_toggle = true;
    }
}
//
inline gotcha_suppression::auto_toggle::auto_toggle(std::false_type)
: auto_toggle{ get(), false }
{}
//
inline gotcha_suppression::auto_toggle::auto_toggle(std::true_type)
: auto_toggle{ get(), true }
{}
//
inline gotcha_suppression::auto_toggle::~auto_toggle()
{
    if(m_value != m_if_equal && m_did_toggle)
    {
        m_value = !m_value;
    }
}
//
}  // namespace component
}  // namespace tim
