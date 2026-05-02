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

#include "timemory/ert/macros.hpp"
#define TIMEMORY_ERT_SOURCE_CXX

#include "timemory/ert/extern.hpp"

namespace tim
{
namespace ert
{
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE(class exec_data<component::ert_timer>)
//
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_CPU(
    class counter<device::cpu, float, component::ert_timer>)
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_CPU(
    class counter<device::cpu, double, component::ert_timer>)
//
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_CPU(
    struct configuration<device::cpu, float, component::ert_timer>)
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_CPU(
    struct configuration<device::cpu, double, component::ert_timer>)
//
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_CPU(
    struct executor<device::cpu, float, component::ert_timer>)
TIMEMORY_INSTANTIATE_ERT_EXTERN_TEMPLATE_CPU(
    struct executor<device::cpu, double, component::ert_timer>)
}  // namespace ert
}  // namespace tim
