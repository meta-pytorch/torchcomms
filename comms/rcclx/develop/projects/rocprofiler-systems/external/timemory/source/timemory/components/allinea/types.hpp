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

/**
 * \file timemory/components/allinea/types.hpp
 * \brief Declare the allinea component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

//--------------------------------------------------------------------------------------//
//
TIMEMORY_DECLARE_COMPONENT(allinea_map)
//
TIMEMORY_SET_COMPONENT_API(component::allinea_map, tpls::allinea, category::external,
                           os::supports_linux)
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_ALLINEA_MAP) || defined(TIMEMORY_COMPILER_INSTRUMENTATION)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::allinea, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::allinea_map, false_type)
#endif
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PROPERTY_SPECIALIZATION(allinea_map, TIMEMORY_ALLINEA_MAP, "allinea_map",
                                 "allinea", "forge")
//
//--------------------------------------------------------------------------------------//
