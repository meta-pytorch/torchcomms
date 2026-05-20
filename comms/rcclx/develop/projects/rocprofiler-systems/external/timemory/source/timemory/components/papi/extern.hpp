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
 * \file timemory/components/papi/extern.hpp
 * \brief Include the extern declarations for papi components
 */

#pragma once

#include "timemory/components/extern/common.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/papi/components.hpp"

TIMEMORY_DECLARE_EXTERN_COMPONENT(papi_vector, true, std::vector<long long>)
TIMEMORY_DECLARE_EXTERN_COMPONENT(papi_array_t, true,
                                  std::array<long long, TIMEMORY_PAPI_ARRAY_SIZE>)

// TIMEMORY_DECLARE_EXTERN_COMPONENT(papi_array8_t, true, std::array<long long, 8>)
// TIMEMORY_DECLARE_EXTERN_COMPONENT(papi_array16_t, true, std::array<long long, 16>)
// TIMEMORY_DECLARE_EXTERN_COMPONENT(papi_array32_t, true, std::array<long long, 32>)
