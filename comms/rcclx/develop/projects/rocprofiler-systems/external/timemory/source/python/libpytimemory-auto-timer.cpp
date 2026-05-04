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

#if !defined(TIMEMORY_PYAUTO_TIMER_SOURCE)
#    define TIMEMORY_PYAUTO_TIMER_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/containers/definition.hpp"
#include "timemory/containers/extern.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/properties.hpp"
#include "timemory/variadic/definition.hpp"

using namespace tim::component;

using auto_timer_t = typename tim::auto_timer::type;
using tim_timer_t  = typename tim::auto_timer::component_type;

//======================================================================================//
//
namespace pyauto_timer
{
//
//--------------------------------------------------------------------------------------//
//
class auto_timer_decorator
{
public:
    auto_timer_decorator(auto_timer_t* _ptr = nullptr)
    : m_ptr(_ptr)
    {}

    ~auto_timer_decorator() { delete m_ptr; }

    auto_timer_decorator& operator=(auto_timer_t* _ptr)
    {
        if(m_ptr)
            delete m_ptr;
        m_ptr = _ptr;
        return *this;
    }

private:
    auto_timer_t* m_ptr;
};
//
//--------------------------------------------------------------------------------------//
//
namespace init
{
//
tim_timer_t*
timer(std::string key)
{
    return new tim_timer_t{ key };
}
//
auto_timer_t*
auto_timer(std::string key, bool report_at_exit)
{
    return new auto_timer_t(key, tim::scope::get_default(), report_at_exit);
}
//
auto_timer_decorator*
timer_decorator(const std::string& key, bool report_at_exit)
{
    auto_timer_decorator* _ptr = new auto_timer_decorator();
    if(!tim::settings::enabled())
        return _ptr;
    return &(*_ptr = new auto_timer_t(key, tim::scope::get_default(), report_at_exit));
}
//
}  // namespace init
//
//--------------------------------------------------------------------------------------//
//
void
generate(py::module& _pymod)
{
    py::class_<tim_timer_t> timer(_pymod, "timer",
                                  "Auto-timer that does not start/stop based on scope");

    py::class_<auto_timer_t> auto_timer(_pymod, "auto_timer", "Pre-configured bundle");

    py::class_<auto_timer_decorator> timer_decorator(
        _pymod, "timer_decorator", "Auto-timer type used in decorators");

    //==================================================================================//
    //
    //                          TIMER
    //
    //==================================================================================//
    timer.def(py::init(&init::timer), "Initialization", py::arg("key") = "",
              py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    timer.def(
        "real_elapsed",
        [](py::object pytimer) {
            tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
            auto&        obj    = *(_timer.get<wall_clock>());
            return obj.get();
        },
        "Elapsed wall clock");
    //----------------------------------------------------------------------------------//
    timer.def(
        "sys_elapsed",
        [](py::object pytimer) {
            tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
            auto&        obj    = *(_timer.get<system_clock>());
            return obj.get();
        },
        "Elapsed system clock");
    //----------------------------------------------------------------------------------//
    timer.def(
        "user_elapsed",
        [](py::object pytimer) {
            tim_timer_t& _timer = *(pytimer.cast<tim_timer_t*>());
            auto&        obj    = *(_timer.get<user_clock>());
            return obj.get();
        },
        "Elapsed user time");
    //----------------------------------------------------------------------------------//
    timer.def(
        "start", [](py::object pytimer) { pytimer.cast<tim_timer_t*>()->start(); },
        "Start timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "stop", [](py::object pytimer) { pytimer.cast<tim_timer_t*>()->stop(); },
        "Stop timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "report",
        [](py::object pytimer) {
            std::cout << *(pytimer.cast<tim_timer_t*>()) << std::endl;
        },
        "Report timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "__str__",
        [](py::object pytimer) {
            std::stringstream ss;
            ss << *(pytimer.cast<tim_timer_t*>());
            return ss.str();
        },
        "Stringify timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "reset", [](py::object self) { self.cast<tim_timer_t*>()->reset(); },
        "Reset the timer");
    //----------------------------------------------------------------------------------//
    timer.def(
        "get_raw", [](py::object self) { return (*self.cast<tim_timer_t*>()).get(); },
        "Get the timer data");
    //----------------------------------------------------------------------------------//
    timer.def(
        "get",
        [](tim_timer_t* self) { return pytim::dict::construct(self->get_labeled()); },
        "Get the timer data");
    //==================================================================================//
    //
    //                      AUTO TIMER
    //
    //==================================================================================//
    auto_timer.def(py::init(&init::auto_timer), "Initialization", py::arg("key") = "",
                   py::arg("report_at_exit") = false,
                   py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    auto_timer.def(
        "__str__",
        [](py::object self) {
            std::stringstream _ss;
            auto_timer_t*     _self = self.cast<auto_timer_t*>();
            _ss << *_self;
            return _ss.str();
        },
        "Print the auto timer");
    //----------------------------------------------------------------------------------//
    auto_timer.def(
        "get_raw", [](py::object self) { return (*self.cast<auto_timer_t*>()).get(); },
        "Get the component list data");
    //----------------------------------------------------------------------------------//
    auto_timer.def(
        "get",
        [](auto_timer_t* self) { return pytim::dict::construct(self->get_labeled()); },
        "Get the component list data");
    //==================================================================================//
    //
    //                      TIMER DECORATOR
    //
    //==================================================================================//
    timer_decorator.def(py::init(&init::timer_decorator), "Initialization",
                        py::return_value_policy::automatic);
    //==================================================================================//
    //
    //                      AUTO TIMER DECORATOR
    //
    //==================================================================================//
    /*
    decorate_auto_timer.def(
        py::init(&pytim::decorators::init::auto_timer), "Initialization",
        py::arg("key") = "", py::arg("line") = pytim::get_line(1),
        py::arg("report_at_exit") = false, py::return_value_policy::take_ownership);
    decorate_auto_timer.def("__call__", &pytim::decorators::auto_timer::call,
                            "Call operator");
                            */
}
}  // namespace pyauto_timer
//
//======================================================================================//
