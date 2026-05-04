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

#include "nvml_utilization_rate.hpp"

using string_pair_type = std::pair<std::string, std::string>;

namespace tim
{
namespace component
{
std::string
nvml_utilization_rate::label()
{
    return "nvml_utilization_rate";
}

std::string
nvml_utilization_rate::description()
{
    return "Utilization information for a device. Each sample period may be between 1 "
           "second and 1/6 second, depending on the product";
}

std::vector<std::string>
nvml_utilization_rate::label_array()
{
    return std::vector<std::string>{ "gpu_utilization", "memory_utilization" };
}

std::vector<uint32_t>
nvml_utilization_rate::unit_array()
{
    return std::vector<uint32_t>{ 1, 1 };
}

std::vector<std::string>
nvml_utilization_rate::display_unit_array()
{
    return std::vector<std::string>{ "%", "%" };
}

void
nvml_utilization_rate::sample(nvmlDevice_t _device)
{
    nvmlUtilization_t _buffer{};
    nvmlDeviceGetUtilizationRates(_device, &_buffer);
    value = _buffer;
}

std::array<uint32_t, 2>
nvml_utilization_rate::get() const
{
    return load().data();
}

std::string
nvml_utilization_rate::get_display() const
{
    std::ostringstream os;
    os << get_value();
    return os.str();
}

}  // namespace component
}  // namespace tim
