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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

//======================================================================================//
// This global method should be used on LINUX or MacOSX platforms with gcc,
// clang, or intel compilers for activating signal detection and forcing
// exception being thrown that can be handled when detected.
//======================================================================================//

#pragma once

#ifndef TIMEMORY_SIGNALS_SIGNAL_HANDLERS_HPP_
#    define TIMEMORY_SIGNALS_SIGNAL_HANDLERS_HPP_
#endif

#include "timemory/backends/signals.hpp"
#include "timemory/defines.h"
#include "timemory/signals/signal_settings.hpp"
#include "timemory/utility/macros.hpp"

#include <initializer_list>
#include <set>
#include <type_traits>

namespace tim
{
namespace signals
{
//
bool
enable_signal_detection(signal_settings::signal_set_t = signal_settings::get_default(),
                        const signal_settings::signal_function_t& = {});

void disable_signal_detection(
    signal_settings::signal_set_t = signal_settings::get_active());

void
update_signal_detection(const signal_settings::signal_set_t& _signals);

//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_SIGNAL_AVAILABLE)

void
termination_signal_message(int sig, siginfo_t* sinfo, std::ostream& message);

void
termination_signal_handler(int sig, siginfo_t* sinfo, void* context);

void
update_file_maps();

#else  // Not a supported architecture

inline bool
enable_signal_detection(signal_settings::signal_set_t,
                        const signal_settings::signal_function_t&)
{
    return false;
}

inline void disable_signal_detection(signal_settings::signal_set_t) {}
#endif
//
template <typename Tp,
          std::enable_if_t<!std::is_enum<Tp>::value && std::is_integral<Tp>::value> = 0>
inline bool
enable_signal_detection(std::initializer_list<Tp>&&               _signals,
                        const signal_settings::signal_function_t& _func = {})
{
    auto operations = signal_settings::signal_set_t{};
    for(const auto& itr : _signals)
        operations.insert(static_cast<sys_signal>(itr));
    return enable_signal_detection(operations, _func);
}
}  // namespace signals
}  // namespace tim

#if defined(TIMEMORY_SIGNALS_HEADER_MODE)
#    include "timemory/signals/signal_handlers.cpp"
#endif
