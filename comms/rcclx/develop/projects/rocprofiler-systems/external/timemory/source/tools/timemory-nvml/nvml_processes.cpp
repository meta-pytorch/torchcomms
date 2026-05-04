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

#include "nvml_processes.hpp"

#include <timemory/settings.hpp>

using string_pair_type = std::pair<std::string, std::string>;

namespace tim
{
namespace component
{
std::string
nvml_processes::label()
{
    return "nvml_processes";
}

std::string
nvml_processes::description()
{
    return "Used GPU memory of running process";
}

std::vector<string_pair_type>
nvml_processes::label_array()
{
    return std::vector<string_pair_type>(
        load().size(), string_pair_type{ "process name", "used GPU memory" });
}

std::vector<nvml_processes::int_pair_type>
nvml_processes::unit_array()
{
    return std::vector<int_pair_type>(load().size(),
                                      int_pair_type{ 1, std::get<1>(_memory_unit()) });
}

std::vector<string_pair_type>
nvml_processes::display_unit_array()
{
    return std::vector<string_pair_type>(
        load().size(), string_pair_type{ "", std::get<0>(_memory_unit()) });
}

void
nvml_processes::sample(nvmlDevice_t _device)
{
    static thread_local std::vector<nvmlProcessInfo_t> _buffer{};

    unsigned int _n = 0;
    nvmlDeviceGetComputeRunningProcesses(_device, &_n, nullptr);
    _buffer.resize(_n);
    nvmlDeviceGetComputeRunningProcesses(_device, &_n, _buffer.data());
    value.resize(_n);
    for(size_t i = 0; i < _n; ++i)
        value.at(i) = _buffer.at(i);
}

nvml_processes::value_type
nvml_processes::get() const
{
    auto _value = load();
    for(auto& itr : _value)
        itr /= std::get<1>(_memory_unit());
    return _value;
}

std::string
nvml_processes::get_display() const
{
    std::ostringstream os;
    for(auto& itr : load())
        os << ", " << (itr / std::get<1>(_memory_unit())) << " "
           << std::get<0>(_memory_unit());
    if(os.str().length() > 0)
        return os.str().substr(2);
    else
        return std::string{};
}

std::tuple<std::string, int64_t>
nvml_processes::_memory_unit()
{
    static auto _value = units::get_memory_unit(settings::memory_units());
    return _value;
}

}  // namespace component
}  // namespace tim
