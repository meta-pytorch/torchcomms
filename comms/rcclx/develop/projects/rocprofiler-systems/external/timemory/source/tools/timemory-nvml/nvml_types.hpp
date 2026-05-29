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

#include "timemory/components/base/declaration.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/units.hpp"

TIMEMORY_DECLARE_COMPONENT(nvml_processes)
TIMEMORY_DECLARE_COMPONENT(nvml_temperature)
TIMEMORY_DECLARE_COMPONENT(nvml_fan_speed)
TIMEMORY_DECLARE_COMPONENT(nvml_memory_info)
TIMEMORY_DECLARE_COMPONENT(nvml_utilization_rate)

TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::nvml_processes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::nvml_temperature, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::nvml_fan_speed, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::nvml_memory_info, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::nvml_utilization_rate,
                               false_type)

TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::nvml_processes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::nvml_temperature, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::nvml_fan_speed, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::nvml_memory_info, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::nvml_utilization_rate, false_type)
