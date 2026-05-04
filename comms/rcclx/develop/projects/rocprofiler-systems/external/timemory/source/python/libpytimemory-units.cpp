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

#if !defined(TIMEMORY_PYUNITS_SOURCE)
#    define TIMEMORY_PYUNITS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/units.hpp"

//======================================================================================//
//
namespace pyunits
{
py::module
generate(py::module& _pymod)
{
    py::module units = _pymod.def_submodule("units", "units for timing and memory");

    units.attr("nsec")     = tim::units::nsec;
    units.attr("usec")     = tim::units::usec;
    units.attr("msec")     = tim::units::msec;
    units.attr("csec")     = tim::units::csec;
    units.attr("dsec")     = tim::units::dsec;
    units.attr("sec")      = tim::units::sec;
    units.attr("min")      = tim::units::minute;
    units.attr("hr")       = tim::units::hour;
    units.attr("byte")     = tim::units::byte;
    units.attr("kilobyte") = tim::units::kilobyte;
    units.attr("megabyte") = tim::units::megabyte;
    units.attr("gigabyte") = tim::units::gigabyte;
    units.attr("terabyte") = tim::units::terabyte;
    units.attr("petabyte") = tim::units::petabyte;

    return units;
}
}  // namespace pyunits
//
//======================================================================================//
