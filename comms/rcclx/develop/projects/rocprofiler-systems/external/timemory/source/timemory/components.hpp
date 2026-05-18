//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file timemory/components.hpp
 * \headerfile timemory/components.hpp "timemory/components.hpp"
 * These are core tools provided by timemory. These tools can be used individually
 * or bundled together in a component_tuple (C++) or component_list (C, Python)
 *
 */

#pragma once

// forward declare any types
#include "timemory/components/data_tracker/components.hpp"
#include "timemory/components/io/components.hpp"
#include "timemory/components/network/components.hpp"
#include "timemory/components/printer/printer.hpp"
#include "timemory/components/rusage/components.hpp"
#include "timemory/components/timestamp/timestamp.hpp"
#include "timemory/components/timing/components.hpp"
#include "timemory/components/trip_count/components.hpp"
#include "timemory/components/types.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

// allinea components
#if defined(TIMEMORY_USE_ALLINEA_MAP)
#    include "timemory/components/allinea/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// caliper components
#if defined(TIMEMORY_USE_CALIPER)
#    include "timemory/components/caliper/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// craypat
#if defined(TIMEMORY_USE_CRAYPAT)
#    include "timemory/components/craypat/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// cuda
#if defined(TIMEMORY_USE_CUDA)
#    include "timemory/components/cuda/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// GPU hardware counter components
#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/components/cupti/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// gotcha components
#if defined(TIMEMORY_USE_GOTCHA)
#    include "timemory/components/gotcha/components.hpp"
#    include "timemory/components/gotcha/memory_allocations.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// gperftools components
#if defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC) ||                                         \
    defined(TIMEMORY_USE_GPERFTOOLS_PROFILER) || defined(TIMEMORY_USE_GPERFTOOLS)
#    include "timemory/components/gperftools/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// hip
#if defined(TIMEMORY_USE_HIP)
#    include "timemory/components/hip/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// likwid
#if defined(TIMEMORY_USE_LIKWID)
#    include "timemory/components/likwid/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// OpenMP components
#if defined(TIMEMORY_USE_OMPT)
#    include "timemory/components/ompt/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// CPU/GPU hardware counter components
#if defined(TIMEMORY_USE_PAPI)
#    include "timemory/components/papi/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// Roofline components
#if defined(TIMEMORY_USE_CUPTI) || defined(TIMEMORY_USE_PAPI)
#    include "timemory/components/roofline/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// TAU component
#if defined(TIMEMORY_USE_TAU)
#    include "timemory/components/tau_marker/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// VTune components
#if defined(TIMEMORY_USE_VTUNE)
#    include "timemory/components/vtune/components.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
// perfetto components
#if defined(TIMEMORY_USE_PERFETTO)
#    include "timemory/components/perfetto/perfetto.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#include "timemory/components/user_bundle/components.hpp"
//
//--------------------------------------------------------------------------------------//
//
#include "timemory/backends.hpp"
