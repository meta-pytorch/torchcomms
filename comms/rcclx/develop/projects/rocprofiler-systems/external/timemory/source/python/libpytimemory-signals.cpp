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

#if !defined(TIMEMORY_PYSIGNALS_SOURCE)
#    define TIMEMORY_PYSIGNALS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/utility/signals.hpp"

//======================================================================================//
//
namespace pysignals
{
using sys_signal_t      = tim::signals::sys_signal;
using signal_settings_t = tim::signals::signal_settings;
using signal_set_t      = signal_settings_t::signal_set_t;
//
//--------------------------------------------------------------------------------------//
//
signal_set_t
signal_list_to_set(py::list signal_list)
{
    signal_set_t signal_set;
    for(auto itr : signal_list)
        signal_set.insert(itr.cast<sys_signal_t>());
    return signal_set;
}
//
//--------------------------------------------------------------------------------------//
//
signal_set_t
get_default_signal_set()
{
    return tim::signals::signal_settings::get_enabled();
}
//
//--------------------------------------------------------------------------------------//
//
void
enable_signal_detection(py::list signal_list = py::list{})
{
    auto _sig_set = (signal_list.size() == 0) ? get_default_signal_set()
                                              : signal_list_to_set(signal_list);
    tim::signals::enable_signal_detection(_sig_set);
}
//
//--------------------------------------------------------------------------------------//
//
void
disable_signal_detection()
{
    tim::signals::disable_signal_detection();
}
//
//--------------------------------------------------------------------------------------//
//
py::module
generate(py::module& _pymod)
{
    //----------------------------------------------------------------------------------//
    //
    //      Module and enumeration
    //
    //----------------------------------------------------------------------------------//
    py::module              sig = _pymod.def_submodule("signals", "Signals submodule");
    py::enum_<sys_signal_t> sys_signal_enum(sig, "Signal", py::arithmetic(),
                                            "Signals for timemory module");

    //----------------------------------------------------------------------------------//
    //
    //      Global functions
    //
    //----------------------------------------------------------------------------------//
    _pymod.def("enable_signal_detection", &enable_signal_detection,
               "Enable signal detection", py::arg("signal_list") = py::list());
    //----------------------------------------------------------------------------------//
    _pymod.def("disable_signal_detection", &disable_signal_detection,
               "Enable signal detection");
    //----------------------------------------------------------------------------------//
    _pymod.def(
        "set_exit_action",
        [](py::function func) {
            auto _func              = [func](int errcode) -> void { func(errcode); };
            using signal_function_t = std::function<void(int)>;
            using std::placeholders::_1;
            signal_function_t _f = std::bind<void>(_func, _1);
            tim::signals::signal_settings::set_exit_action(_f);
        },
        "Set the exit action when a signal is raised -- function must accept "
        "integer");

    //----------------------------------------------------------------------------------//
    //
    //      Enumeration definition
    //
    //----------------------------------------------------------------------------------//
    sys_signal_enum.value("Hangup", sys_signal_t::Hangup)
        .value("Interrupt", sys_signal_t::Interrupt)
        .value("Quit", sys_signal_t::Quit)
        .value("Illegal", sys_signal_t::Illegal)
        .value("Trap", sys_signal_t::Trap)
        .value("Abort", sys_signal_t::Abort)
        .value("Emulate", sys_signal_t::Emulate)
        .value("FPE", sys_signal_t::FPE)
        .value("Kill", sys_signal_t::Kill)
        .value("Bus", sys_signal_t::Bus)
        .value("SegFault", sys_signal_t::SegFault)
        .value("System", sys_signal_t::System)
        .value("Pipe", sys_signal_t::Pipe)
        .value("Alarm", sys_signal_t::Alarm)
        .value("Terminate", sys_signal_t::Terminate)
        .value("Urgent", sys_signal_t::Urgent)
        .value("Stop", sys_signal_t::Stop)
        .value("CPUtime", sys_signal_t::CPUtime)
        .value("FileSize", sys_signal_t::FileSize)
        .value("VirtualAlarm", sys_signal_t::VirtualAlarm)
        .value("ProfileAlarm", sys_signal_t::ProfileAlarm)
        .value("User1", sys_signal_t::User1)
        .value("User2", sys_signal_t::User2);
    sys_signal_enum.export_values();

    return sig;
}
}  // namespace pysignals
//
//======================================================================================//
