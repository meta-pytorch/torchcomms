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

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/concepts.hpp"  // for concepts::is_component
#include "timemory/mpl/types.hpp"     // for derivation_types
#include "timemory/tpls/cereal/cereal.hpp"

#if !defined(TIMEMORY_COMPONENT_SOURCE) && !defined(TIMEMORY_USE_PRINTER_EXTERN)
#    if !defined(TIMEMORY_PRINTER_COMPONENT_HEADER_MODE)
#        define TIMEMORY_PRINTER_COMPONENT_HEADER_MODE 1
#    endif
#endif

namespace tim
{
namespace component
{
// forward decl
struct printer;
struct timestamp;
}  // namespace component

namespace trait
{
// a component's assemble and derive member functions only get called is the derivation
// types are specified
template <>
struct derivation_types<component::printer>
{
    using type = type_list<type_list<component::timestamp>>;
};
}  // namespace trait
}  // namespace tim

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::printer, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(default_runtime_enabled, component::printer, false_type)

TIMEMORY_PROPERTY_SPECIALIZATION(printer, TIMEMORY_PRINTER, "printer", "")
