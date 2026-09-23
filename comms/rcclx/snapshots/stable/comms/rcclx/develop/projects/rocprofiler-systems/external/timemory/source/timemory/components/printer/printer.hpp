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

#include "timemory/components/base.hpp"
#include "timemory/components/printer/types.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace tim
{
namespace component
{
/// \struct tim::component::printer
/// \brief A diagnostic component when prints messages via start(...) and stores messages
/// via store(...). The stored messages are returned via the get() member function. If
/// bundled alongside the timestamp component, the timestamp will be added to the stored
/// message
struct printer : empty_base
{
    using value_type = void;

    static std::string label();
    static std::string description();
    static std::string get_label();
    static std::string get_description();

    printer()          = default;
    ~printer()         = default;
    printer(printer&&) = default;
    printer(const printer&);

    printer& operator=(printer&&) = default;
    printer& operator             =(const printer&);

    std::string get() const;
    void        set_prefix(const char*);
    bool        assemble(timestamp*);

    template <typename... Args>
    auto start(Args&&... args)
        -> decltype(TIMEMORY_FOLD_EXPRESSION(std::declval<std::stringstream>() << args),
                    void());

    template <typename... Args>
    auto store(Args&&... args)
        -> decltype(TIMEMORY_FOLD_EXPRESSION(std::declval<std::stringstream>() << args),
                    void());

private:
    const char*       m_prefix = nullptr;
    timestamp*        m_ts     = nullptr;
    std::stringstream m_stream{};
};
}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/components/timestamp/timestamp.hpp"  // for timestamp::as_string
#include "timemory/variadic/macros.hpp"                 // for TIMEMORY_JOIN

template <typename... Args>
auto
tim::component::printer::start(Args&&... args)
    -> decltype(TIMEMORY_FOLD_EXPRESSION(std::declval<std::stringstream>() << args),
                void())
{
    // only print message if arguments provided
    if(sizeof...(Args) == 0)
        return;

    std::cerr << std::flush;
    if(m_prefix)
        std::cerr << "[" << m_prefix << "]";
    std::cerr << "> " << TIMEMORY_JOIN("", std::forward<Args>(args)...) << '\n';
}

template <typename... Args>
auto
tim::component::printer::store(Args&&... args)
    -> decltype(TIMEMORY_FOLD_EXPRESSION(std::declval<std::stringstream>() << args),
                void())
{
    // only store message if arguments provided
    if(sizeof...(Args) == 0)
        return;

    std::string _tp{};
    if(m_prefix)
        _tp += TIMEMORY_JOIN("", '[', m_prefix, ']');
    if(m_ts)
        _tp += "[" + timestamp::as_string(m_ts->get()) + "]";
    m_stream << _tp << "> " << TIMEMORY_JOIN("", std::forward<Args>(args)...) << '\n';
}

#if defined(TIMEMORY_PRINTER_COMPONENT_HEADER_MODE) &&                                   \
    TIMEMORY_PRINTER_COMPONENT_HEADER_MODE > 0
#    include "timemory/components/printer/printer.cpp"
#endif
