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
#include <tuple>
#include <utility>
#include <vector>

namespace tim
{
namespace component
{
struct nvml_processes
: public base<nvml_processes, std::vector<backends::nvml_process_data>>
{
    using string_pair_type = std::pair<std::string, std::string>;
    using int_pair_type    = std::pair<unsigned int, unsigned long long>;
    using value_pair_type  = backends::nvml_process_data;
    using value_type       = std::vector<value_pair_type>;
    using base_type        = base<nvml_processes, value_type>;

    static std::string label();
    static std::string description();

    std::vector<string_pair_type> label_array();
    std::vector<int_pair_type>    unit_array();
    std::vector<string_pair_type> display_unit_array();

    void sample(nvmlDevice_t _device);

    value_type  get() const;
    std::string get_display() const;

private:
    static std::tuple<std::string, int64_t> _memory_unit();
};
}  // namespace component
}  // namespace tim

TIMEMORY_SET_CLASS_VERSION(0, tim::component::nvml_processes)
TIMEMORY_SET_CLASS_VERSION(0, tim::backends::nvml_process_data)
