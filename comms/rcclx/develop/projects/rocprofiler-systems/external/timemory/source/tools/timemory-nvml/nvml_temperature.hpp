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

#include "nvml_backends.hpp"
#include "nvml_types.hpp"
#include "timemory/components/base/declaration.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/units.hpp"

#include <cassert>
#include <nvml.h>
#include <string>

namespace tim
{
namespace component
{
struct nvml_temperature : public base<nvml_temperature, unsigned int>
{
    using value_type = unsigned int;
    using base_type  = base<nvml_temperature, value_type>;

    static std::string label();
    static std::string description();
    static std::string display_unit() { return "Celsius"; }

    void sample(nvmlDevice_t _device);

    std::string get_display() const { return std::to_string(get()) + " deg C"; }
};
}  // namespace component
}  // namespace tim

TIMEMORY_SET_CLASS_VERSION(0, tim::component::nvml_temperature)
