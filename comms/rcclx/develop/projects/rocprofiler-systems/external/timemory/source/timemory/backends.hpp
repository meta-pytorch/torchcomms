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

/** \file timemory/backends.hpp
 * \headerfile timemory/backends.hpp "timemory/backends.hpp"
 * Generic header for backends
 *
 */

#pragma once

#include "timemory/backends/types.hpp"
//
#include "timemory/backends/device.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/signals.hpp"
#include "timemory/backends/threading.hpp"
//
#include "timemory/components/rusage/backends.hpp"
#include "timemory/components/timing/backends.hpp"

#if defined(TIMEMORY_USE_CALIPER)
#    include "timemory/components/caliper/backends.hpp"
#endif

#if defined(TIMEMORY_USE_GOTCHA)
#    include "timemory/backends/gotcha.hpp"
#endif

#if defined(TIMEMORY_USE_MPI)
#    include "timemory/backends/mpi.hpp"
#endif

#if defined(TIMEMORY_USE_UPCXX)
#    include "timemory/backends/upcxx.hpp"
#endif

#if defined(TIMEMORY_USE_PAPI)
#    include "timemory/backends/papi.hpp"
#endif

#if defined(TIMEMORY_USE_GPERFTOOLS) || defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC) ||     \
    defined(TIMEMORY_USE_GPERFTOOLS_PROFILER)
#    include "timemory/backends/gperftools.hpp"
#endif

//--------------------------------------------------------------------------------------//
//
//      GPU backends
//
//--------------------------------------------------------------------------------------//

#include "timemory/backends/gpu.hpp"

#if defined(TIMEMORY_USE_CUDA)
#    include "timemory/backends/cuda.hpp"
#    include "timemory/components/cuda/backends.hpp"
#endif

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"
#endif

#if defined(TIMEMORY_USE_NVTX)
#    include "timemory/backends/nvtx.hpp"
#endif

#if defined(TIMEMORY_USE_HIP)
#    include "timemory/backends/hip.hpp"
#    include "timemory/components/hip/backends.hpp"
#endif
