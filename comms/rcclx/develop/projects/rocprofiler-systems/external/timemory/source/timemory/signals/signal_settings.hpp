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

#ifndef TIMEMORY_SIGNALS_SIGNAL_SETTINGS_HPP_
#    define TIMEMORY_SIGNALS_SIGNAL_SETTINGS_HPP_
#endif

#include "timemory/backends/signals.hpp"
#include "timemory/defines.h"
#include "timemory/macros/os.hpp"
#include "timemory/signals/types.hpp"
#include "timemory/utility/macros.hpp"

#include <cfenv>
#include <csignal>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace tim
{
namespace signals
{
class signal_settings
{
public:
    using signal_set_t      = std::set<sys_signal>;
    using signal_function_t = std::function<void(int)>;
    using descript_tuple_t  = std::tuple<std::string, int, std::string>;
    using sigaction_t       = struct sigaction;

public:
    static bool&            allow();
    static bool             is_active(int = -1);
    static bool             is_active(sys_signal);
    static void             enable(const sys_signal&);
    static void             disable(const sys_signal&);
    static std::string      str(const sys_signal&);
    static std::string      str(bool report_disabled = false);
    static void             check_environment();
    static bool             set_action(sys_signal, signal_function_t _f);
    static void             set_exit_action(signal_function_t _f);
    static void             exit_action(int errcode);
    static void             exit_action(int errcode, void*, void*);
    static descript_tuple_t get_info(const sys_signal&);

    static signal_set_t get_enabled();
    static signal_set_t get_disabled();
    static signal_set_t get_default();
    static signal_set_t get_active();
    static bool&        enable_all();
    static bool&        disable_all();

protected:
    struct signals_data
    {
        signals_data();
        ~signals_data()                   = default;
        signals_data(const signals_data&) = default;
        signals_data(signals_data&&)      = default;
        signals_data& operator=(const signals_data&) = default;
        signals_data& operator=(signals_data&&) = default;

        struct entry
        {
            bool              active  = false;
            signal_function_t functor = {};
#if defined(TIMEMORY_UNIX)
            sigaction_t current  = {};
            sigaction_t previous = {};
#endif
        };

        bool              enable_all        = false;
        bool              disable_all       = false;
        signal_function_t signals_exit_func = [](int) {};
        signal_set_t      signals_enabled   = {};
        signal_set_t      signals_disabled  = {
            sys_signal::Hangup,       sys_signal::Interrupt, sys_signal::Trap,
            sys_signal::Emulate,      sys_signal::FPE,       sys_signal::Kill,
            sys_signal::System,       sys_signal::Pipe,      sys_signal::Alarm,
            sys_signal::Terminate,    sys_signal::Urgent,    sys_signal::Stop,
            sys_signal::CPUtime,      sys_signal::FileSize,  sys_signal::VirtualAlarm,
            sys_signal::ProfileAlarm, sys_signal::User1,     sys_signal::User2
        };
        // default signals to catch
        signal_set_t signals_default        = { sys_signal::Quit, sys_signal::Illegal,
                                         sys_signal::Abort, sys_signal::Bus,
                                         sys_signal::SegFault };
        std::map<sys_signal, entry> entries = {};
    };

    static signals_data& f_signals();

public:
    static auto* get(sys_signal _v) { return &f_signals().entries[_v]; }
};
}  // namespace signals
}  // namespace tim

#if defined(TIMEMORY_SIGNALS_HEADER_MODE)
#    include "timemory/signals/signal_settings.cpp"
#endif
