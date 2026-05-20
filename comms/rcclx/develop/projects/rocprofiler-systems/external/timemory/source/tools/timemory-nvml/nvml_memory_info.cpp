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

#include "nvml_memory_info.hpp"

#include <timemory/settings.hpp>

namespace tim
{
namespace component
{
std::string
nvml_memory_info::label()
{
    return "nvml_memory_info";
}

std::string
nvml_memory_info::description()
{
    return "Free/Used/Total Memory on the GPU";
}

std::vector<std::string>
nvml_memory_info::label_array()
{
    return std::vector<std::string>{ "free", "total", "used" };
}

std::vector<unsigned long long>
nvml_memory_info::unit_array()
{
    return std::vector<unsigned long long>(3, std::get<1>(_memory_unit()));
}

std::vector<std::string>
nvml_memory_info::display_unit_array()
{
    return std::vector<std::string>(3, std::get<0>(_memory_unit()));
}

void
nvml_memory_info::sample(nvmlDevice_t _device)
{
    nvmlMemory_t _buffer;
    nvmlDeviceGetMemoryInfo(_device, &_buffer);
    set_value(_buffer);
}

std::array<unsigned long long, 3>
nvml_memory_info::get() const
{
    using namespace tim::component::operators;
    return load().data() / std::get<1>(_memory_unit());
}

std::string
nvml_memory_info::get_display() const
{
    std::ostringstream os;
    for(auto& itr : load().data())
        os << ", " << (itr / std::get<1>(_memory_unit())) << " "
           << std::get<0>(_memory_unit());
    return os.str().substr(2);
}

std::tuple<std::string, int64_t>
nvml_memory_info::_memory_unit()
{
    static auto _value = units::get_memory_unit(settings::memory_units());
    return _value;
}

}  // namespace component
}  // namespace tim
