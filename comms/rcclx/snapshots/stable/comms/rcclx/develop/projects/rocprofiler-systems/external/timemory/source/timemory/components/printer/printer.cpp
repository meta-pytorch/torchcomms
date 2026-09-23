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

#ifndef TIMEMORY_COMPONENT_PRINTER_PRINTER_CPP_
#define TIMEMORY_COMPONENT_PRINTER_PRINTER_CPP_

#include "timemory/components/printer/types.hpp"

#if !defined(TIMEMORY_PRINTER_COMPONENT_HEADER_MODE)
#    include "timemory/components/printer/printer.hpp"
#    define TIMEMORY_COMPONENT_PRINTER_INLINE
#else
#    define TIMEMORY_COMPONENT_PRINTER_INLINE inline
#endif

namespace tim
{
namespace component
{
TIMEMORY_COMPONENT_PRINTER_INLINE
std::string
printer::label()
{
    return "printer";
}

TIMEMORY_COMPONENT_PRINTER_INLINE
std::string
printer::description()
{
    return "Provides an interface for printing out debug messages";
}

TIMEMORY_COMPONENT_PRINTER_INLINE
std::string
printer::get_label()
{
    return label();
}

TIMEMORY_COMPONENT_PRINTER_INLINE
std::string
printer::get_description()
{
    return description();
}

TIMEMORY_COMPONENT_PRINTER_INLINE
printer::printer(const printer& rhs)
: m_prefix{ rhs.m_prefix }
, m_ts{ rhs.m_ts }
, m_stream{ rhs.m_stream.str() }
{}

TIMEMORY_COMPONENT_PRINTER_INLINE
printer&
printer::operator=(const printer& rhs)
{
    if(this == &rhs)
        return *this;

    m_prefix = rhs.m_prefix;
    m_ts     = rhs.m_ts;
    m_stream << rhs.m_stream.str();

    return *this;
}

TIMEMORY_COMPONENT_PRINTER_INLINE
std::string
printer::get() const
{
    return m_stream.str();
}

TIMEMORY_COMPONENT_PRINTER_INLINE
void
printer::set_prefix(const char* _prefix)
{
    m_prefix = _prefix;
}

TIMEMORY_COMPONENT_PRINTER_INLINE
bool
printer::assemble(timestamp* _wc)
{
    m_ts = _wc;
    return (m_ts != nullptr);
}

}  // namespace component
}  // namespace tim
#endif
