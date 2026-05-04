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

#ifndef TIMEMORY_SIGNALS_SIGNAL_SETTINGS_CPP_
#    define TIMEMORY_SIGNALS_SIGNAL_SETTINGS_CPP_
#endif

#include "timemory/signals/macros.hpp"

#if !defined(TIMEMORY_SIGNALS_SIGNAL_SETTINGS_HPP_) &&                                   \
    !defined(TIMEMORY_SIGNALS_HEADER_MODE)
#    include "timemory/signals/signal_settings.hpp"
#endif

#include "timemory/environment/declaration.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/signals/types.hpp"
#include "timemory/utility/declaration.hpp"

#include <cfenv>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>

#if defined(TIMEMORY_SIGNAL_AVAILABLE)
#    include <dlfcn.h>
#endif

namespace tim
{
namespace signals
{
TIMEMORY_SIGNALS_INLINE
signal_settings::signals_data::signals_data()
{
#if defined(DEBUG)
    signals_default.insert(sys_signal::FPE);
    signals_enabled.insert(sys_signal::FPE);
#else
    signals_disabled.insert(sys_signal::FPE);
#endif
}

TIMEMORY_SIGNALS_INLINE
void
insert_and_remove(const sys_signal&              _type,  // signal type
                  signal_settings::signal_set_t* _ins,   // set to insert into
                  signal_settings::signal_set_t* _rem)   // set to remove from

{
    if(!_ins || !_rem)
        return;
    _ins->insert(_type);
    auto itr = _rem->find(_type);
    if(itr != _rem->end())
        _rem->erase(itr);
}

TIMEMORY_SIGNALS_INLINE
void
signal_settings::enable(const sys_signal& _type)
{
    insert_and_remove(_type, &f_signals().signals_enabled, &f_signals().signals_disabled);
}

TIMEMORY_SIGNALS_INLINE
void
signal_settings::disable(const sys_signal& _type)
{
    insert_and_remove(_type, &f_signals().signals_disabled, &f_signals().signals_enabled);
}

TIMEMORY_SIGNALS_INLINE
void
signal_settings::check_environment()
{
    using match_t = std::pair<std::string, sys_signal>;

    auto _list = {
        match_t("HANGUP", sys_signal::Hangup),
        match_t("INTERRUPT", sys_signal::Interrupt),
        match_t("QUIT", sys_signal::Quit),
        match_t("ILLEGAL", sys_signal::Illegal),
        match_t("TRAP", sys_signal::Trap),
        match_t("ABORT", sys_signal::Abort),
        match_t("EMULATE", sys_signal::Emulate),
        match_t("FPE", sys_signal::FPE),
        match_t("KILL", sys_signal::Kill),
        match_t("BUS", sys_signal::Bus),
        match_t("SEGFAULT", sys_signal::SegFault),
        match_t("SYSTEM", sys_signal::System),
        match_t("PIPE", sys_signal::Pipe),
        match_t("ALARM", sys_signal::Alarm),
        match_t("TERMINATE", sys_signal::Terminate),
        match_t("URGENT", sys_signal::Urgent),
        match_t("STOP", sys_signal::Stop),
        match_t("CPUTIME", sys_signal::CPUtime),
        match_t("FILESIZE", sys_signal::FileSize),
        match_t("VIRTUALALARM", sys_signal::VirtualAlarm),
        match_t("PROFILEALARM", sys_signal::ProfileAlarm),
        match_t("USER1", sys_signal::User1),
        match_t("USER2", sys_signal::User2),
    };

    for(const auto& itr : _list)
    {
        auto _name   = std::get<0>(get_info(itr.second));
        auto _enable = get_env(
            TIMEMORY_SETTINGS_PREFIX "SIGNAL_ENABLE_" + _name,
            get_env(TIMEMORY_SETTINGS_PREFIX "SIGNAL_ENABLE_" + itr.first, false, false));
        auto _disable =
            get_env(TIMEMORY_SETTINGS_PREFIX "SIGNAL_DISABLE_" + _name,
                    get_env(TIMEMORY_SETTINGS_PREFIX "SIGNAL_DISABLE_" + itr.first, false,
                            false));

        if(_enable)
            signal_settings::enable(itr.second);
        if(_disable)
            signal_settings::disable(itr.second);
    }

    if(enable_all())
    {
        for(const auto& itr : _list)
            signal_settings::enable(itr.second);
    }

    if(disable_all())
    {
        for(const auto& itr : _list)
            signal_settings::disable(itr.second);
    }
}

TIMEMORY_SIGNALS_INLINE
signal_settings::descript_tuple_t
signal_settings::get_info(const sys_signal& _type)
{
    // some of these signals are not handled but added in case they are
    // enabled in the future
    static std::vector<descript_tuple_t> descript_data = {
        descript_tuple_t("SIGHUP", SIGHUP, "terminal line hangup"),
        descript_tuple_t("SIGINT", SIGINT, "interrupt program"),
        descript_tuple_t("SIGQUIT", SIGQUIT, "quit program"),
        descript_tuple_t("SIGILL", SIGILL, "illegal instruction"),
        descript_tuple_t("SIGTRAP", SIGTRAP, "trace trap"),
        descript_tuple_t("SIGABRT", SIGABRT, "abort program (formerly SIGIOT)"),
        descript_tuple_t("SIGEMT", SIGEMT, "emulate instruction executed"),
        descript_tuple_t("SIGFPE", SIGFPE, "floating-point exception"),
        descript_tuple_t("SIGKILL", SIGKILL, "kill program"),
        descript_tuple_t("SIGBUS", SIGBUS, "bus error"),
        descript_tuple_t("SIGSEGV", SIGSEGV, "segmentation violation"),
        descript_tuple_t("SIGSYS", SIGSYS, "non-existent system call invoked"),
        descript_tuple_t("SIGPIPE", SIGPIPE, "write on a pipe with no reader"),
        descript_tuple_t("SIGALRM", SIGALRM, "real-time timer expired"),
        descript_tuple_t("SIGTERM", SIGTERM, "software termination signal"),
        descript_tuple_t("SIGURG", SIGURG, "urgent condition present on socket"),
        descript_tuple_t("SIGSTOP", SIGSTOP, "stop (cannot be caught or ignored)"),
        descript_tuple_t("SIGTSTP", SIGTSTP, "stop signal generated from keyboard"),
        descript_tuple_t("SIGCONT", SIGCONT, "continue after stop"),
        descript_tuple_t("SIGCHLD", SIGCHLD, "child status has changed"),
        descript_tuple_t("SIGTTIN", SIGTTIN,
                         "background read attempted from control terminal"),
        descript_tuple_t("SIGTTOU", SIGTTOU,
                         "background write attempted to control terminal"),
        descript_tuple_t("SIGIO ", SIGIO, "I/O is possible on a descriptor"),
        descript_tuple_t("SIGXCPU", SIGXCPU, "cpu time limit exceeded"),
        descript_tuple_t("SIGXFSZ", SIGXFSZ, "file size limit exceeded"),
        descript_tuple_t("SIGVTALRM", SIGVTALRM, "virtual time alarm"),
        descript_tuple_t("SIGPROF", SIGPROF, "profiling timer alarm"),
        descript_tuple_t("SIGWINCH", SIGWINCH, "Window size change"),
        descript_tuple_t("SIGINFO", SIGINFO, "status request from keyboard"),
        descript_tuple_t("SIGUSR1", SIGUSR1, "User defined signal 1"),
        descript_tuple_t("SIGUSR2", SIGUSR2, "User defined signal 2")
    };

    int _key = (int) _type;
    for(const auto& itr : descript_data)
    {
        if(std::get<1>(itr) == _key)
            return itr;
    }

    return descript_tuple_t{ "", _key, "" };
}

TIMEMORY_SIGNALS_INLINE
std::string
signal_settings::str(const sys_signal& _type)
{
    std::stringstream ss;
    auto              _descript = [&](const descript_tuple_t& _data) {
        ss << " Signal: " << std::setw(10) << std::get<0>(_data)
           << " (signal number: " << std::setw(3) << std::get<1>(_data) << ") "
           << std::setw(40) << std::get<2>(_data);
    };

    auto _v = get_info(_type);
    if(!std::get<0>(_v).empty())
        _descript(_v);

    return ss.str();
}

TIMEMORY_SIGNALS_INLINE
std::string
signal_settings::str(bool report_disabled)
{
    std::stringstream ss;
    auto              spacer = []() { return "    "; };

#if defined(TIMEMORY_SIGNAL_AVAILABLE)

    ss << std::endl
       << spacer() << "Signal detection activated. Signal exception settings:\n"
       << std::endl;

    if(report_disabled)
        ss << spacer() << "Enabled:" << std::endl;
    for(const auto& itr : f_signals().signals_enabled)
        ss << spacer() << spacer() << signal_settings::str(itr) << '\n';

    if(report_disabled)
    {
        ss << "\n" << spacer() << "Disabled:" << std::endl;
        for(const auto& itr : f_signals().signals_disabled)
            ss << spacer() << spacer() << signal_settings::str(itr) << '\n';
    }

#else

    ss << std::endl << spacer() << "Signal detection not available" << std::endl;

    (void) report_disabled;
#endif

    return ss.str();
}

TIMEMORY_SIGNALS_INLINE
bool
signal_settings::is_active(int _v)
{
    if(_v < 0)
    {
        for(auto& itr : f_signals().entries)  // NOLINT
            if(itr.second.active)
                return true;
        return false;
    }
    return is_active(static_cast<sys_signal>(_v));
}

TIMEMORY_SIGNALS_INLINE
bool
signal_settings::is_active(sys_signal _v)
{
    auto itr = f_signals().entries.find(_v);
    return (itr == f_signals().entries.end()) ? false : itr->second.active;
}

TIMEMORY_SIGNALS_INLINE bool&
signal_settings::allow()
{
    static bool _instance = true;
    return _instance;
}

TIMEMORY_SIGNALS_INLINE
bool&
signal_settings::enable_all()
{
    return f_signals().enable_all;
}

TIMEMORY_SIGNALS_INLINE
bool&
signal_settings::disable_all()
{
    return f_signals().disable_all;
}

TIMEMORY_SIGNALS_INLINE
bool
signal_settings::set_action(sys_signal _v, signal_function_t _f)
{
    auto& itr = f_signals().entries[_v];
    if(!itr.active)
    {
        itr.functor = std::move(_f);
        return true;
    }
    return false;
}

TIMEMORY_SIGNALS_INLINE
void
signal_settings::set_exit_action(signal_function_t _f)
{
    f_signals().signals_exit_func = std::move(_f);
}

TIMEMORY_SIGNALS_INLINE
void
signal_settings::exit_action(int _v)
{
    auto& _entries = f_signals().entries;
    auto  itr      = _entries.find(static_cast<sys_signal>(_v));

    if(itr != _entries.end() && itr->second.functor)
        itr->second.functor(_v);

    auto& _exit_func = f_signals().signals_exit_func;
    if(_exit_func)
        _exit_func(_v);
}

#if defined(TIMEMORY_WINDOWS)
TIMEMORY_SIGNALS_INLINE
void
signal_settings::exit_action(int, void*, void*)
{}
#else
TIMEMORY_SIGNALS_INLINE
void
signal_settings::exit_action(int _signum, void* _siginfo, void* _context)
{
    auto& _entries = f_signals().entries;
    auto itr = _entries.find(static_cast<sys_signal>(_signum));
    if(itr != _entries.end())
    {
        auto& _former = itr->second.previous;
        if((_former.sa_flags & (1L << SA_SIGINFO)) != 0)
        {
            if(_former.sa_sigaction)
                _former.sa_sigaction(_signum, reinterpret_cast<siginfo_t*>(_siginfo),
                                     _context);
        }
        else
        {
            if(_former.sa_handler)
                _former.sa_handler(_signum);
        }
    }
}
#endif

TIMEMORY_SIGNALS_INLINE
signal_settings::signal_set_t
signal_settings::get_enabled()
{
    return f_signals().signals_enabled;
}

TIMEMORY_SIGNALS_INLINE
signal_settings::signal_set_t
signal_settings::get_disabled()
{
    return f_signals().signals_disabled;
}

TIMEMORY_SIGNALS_INLINE
signal_settings::signal_set_t
signal_settings::get_default()
{
    return f_signals().signals_default;
}

TIMEMORY_SIGNALS_INLINE
signal_settings::signal_set_t
signal_settings::get_active()
{
    auto _v = signal_set_t{};
    for(auto& itr : f_signals().entries)
        if(itr.second.active)
            _v.emplace(itr.first);
    return _v;
}

// clang-format off
TIMEMORY_SIGNALS_INLINE
signal_settings::signals_data&
signal_settings::f_signals()
// clang-format on
{
    static auto* _v = new signals_data{};
    return *_v;
}
}  // namespace signals
}  // namespace tim
