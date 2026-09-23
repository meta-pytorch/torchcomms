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

#if !defined(TIMEMORY_PYRSS_USAGE_SOURCE)
#    define TIMEMORY_PYRSS_USAGE_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/components/rusage/components.hpp"
#include "timemory/components/rusage/extern.hpp"
#include "timemory/operations/definition.hpp"
#include "timemory/operations/types/get.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/properties.hpp"
#include "timemory/variadic/definition.hpp"

using namespace tim::component;

//======================================================================================//
//
namespace pyrss_usage
{
//--------------------------------------------------------------------------------------//
//
using rss_usage_t =
    tim::component_bundle_t<TIMEMORY_API, page_rss, peak_rss, num_minor_page_faults,
                            num_major_page_faults, voluntary_context_switch,
                            priority_context_switch>;
//
//--------------------------------------------------------------------------------------//
//
namespace init
{
rss_usage_t*
rss_usage(std::string key, bool record)
{
    rss_usage_t* _rss = new rss_usage_t(key, true);
    if(record)
        _rss->measure();
    return _rss;
}
}  // namespace init
//
//--------------------------------------------------------------------------------------//
//
py::class_<rss_usage_t>
generate(py::module& _pymod, py::module& _pyunits)
{
    py::class_<rss_usage_t> rss_usage(_pymod, "rss_usage",
                                      "Pre-configured memory usage bundle");

    rss_usage.def(py::init(&init::rss_usage), "Initialization of RSS measurement class",
                  py::arg("key") = "", py::arg("record") = false,
                  py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "record", [](py::object self) { self.cast<rss_usage_t*>()->record(); },
        "Record the RSS usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__str__",
        [](py::object self) {
            std::stringstream ss;
            ss << *(self.cast<rss_usage_t*>());
            return ss.str();
        },
        "Stringify the rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__iadd__",
        [](py::object self, py::object rhs) {
            *(self.cast<rss_usage_t*>()) += *(rhs.cast<rss_usage_t*>());
            return self;
        },
        "Add rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__isub__",
        [](py::object self, py::object rhs) {
            *(self.cast<rss_usage_t*>()) -= *(rhs.cast<rss_usage_t*>());
            return self;
        },
        "Subtract rss usage");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__add__",
        [](py::object self, py::object rhs) {
            rss_usage_t* _rss = new rss_usage_t(*(self.cast<rss_usage_t*>()));
            *_rss += *(rhs.cast<rss_usage_t*>());
            return _rss;
        },
        "Add rss usage", py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "__sub__",
        [](py::object self, py::object rhs) {
            rss_usage_t* _rss = new rss_usage_t(*(self.cast<rss_usage_t*>()));
            *_rss -= *(rhs.cast<rss_usage_t*>());
            return _rss;
        },
        "Subtract rss usage", py::return_value_policy::take_ownership);
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "current",
        [](py::object self, int64_t /*_units*/) {
            return std::get<0>(*self.cast<rss_usage_t*>()).get_display();
        },
        "Return the current rss usage", py::arg("units") = _pyunits.attr("megabyte"));
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "peak",
        [](py::object self, int64_t /*_units*/) {
            return std::get<1>(*self.cast<rss_usage_t*>()).get_display();
        },
        "Return the current rss usage", py::arg("units") = _pyunits.attr("megabyte"));
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "get_raw", [](rss_usage_t* self) { return self->get(); },
        "Return the rss usage data");
    //----------------------------------------------------------------------------------//
    rss_usage.def(
        "get",
        [](rss_usage_t* self) { return pytim::dict::construct(self->get_labeled()); },
        "Return the rss usage data");

    return rss_usage;
}
}  // namespace pyrss_usage
//
//======================================================================================//
