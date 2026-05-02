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

#include "timemory/backends/device.hpp"
#include "timemory/backends/gpu.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/data_tracker/components.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/macros.hpp"

namespace device   = tim::device;
namespace mpl      = tim::mpl;
namespace comp     = tim::component;
namespace gpu      = tim::gpu;
namespace concepts = tim::concepts;

TIMEMORY_DECLARE_COMPONENT(gpu_device_timer)
TIMEMORY_DECLARE_COMPONENT(gpu_op_tracker)
TIMEMORY_SET_COMPONENT_API(component::gpu_device_timer, device::gpu, category::external,
                           os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::gpu_op_tracker, device::gpu, category::external,
                           os::agnostic)

#if !defined(TIMEMORY_USE_GPU)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_device_timer, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_op_tracker, false_type)
#endif

struct gpu_data_tag : concepts::api
{};
using gpu_device_timer_data = comp::data_tracker<double, gpu_data_tag>;
using gpu_device_op_data    = comp::data_tracker<int64_t, gpu_data_tag>;

TIMEMORY_STATISTICS_TYPE(gpu_device_timer_data, double)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, gpu_device_timer_data, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, gpu_device_timer_data, true_type)
TIMEMORY_STATISTICS_TYPE(gpu_device_op_data, int64_t)

#define CLOCK_DTYPE unsigned long long
