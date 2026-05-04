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

#include "timemory/macros/os.hpp"

#if defined(TIMEMORY_USE_EXTERN)
#    if !defined(TIMEMORY_USE_TIMING_EXTERN)
#        define TIMEMORY_USE_TIMING_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_RUSAGE_EXTERN)
#        define TIMEMORY_USE_RUSAGE_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_IO_EXTERN) && !defined(TIMEMORY_MACOSX)
#        define TIMEMORY_USE_IO_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_NETWORK_EXTERN) && defined(TIMEMORY_LINUX)
#        define TIMEMORY_USE_NETWORK_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_TRIP_COUNT_EXTERN)
#        define TIMEMORY_USE_TRIP_COUNT_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_DATA_TRACKER_EXTERN)
#        define TIMEMORY_USE_DATA_TRACKER_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_TIMESTAMP_EXTERN)
#        define TIMEMORY_USE_TIMESTAMP_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
#        define TIMEMORY_USE_USER_BUNDLE_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_ALLINEA_MAP_EXTERN) && defined(TIMEMORY_USE_ALLINEA_MAP)
#        define TIMEMORY_USE_ALLINEA_MAP_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_CALIPER_EXTERN) && defined(TIMEMORY_USE_CALIPER)
#        define TIMEMORY_USE_CALIPER_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_CRAYPAT_EXTERN) && defined(TIMEMORY_USE_CRAYPAT)
#        define TIMEMORY_USE_CRAYPAT_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_CUDA_EXTERN) && defined(TIMEMORY_USE_CUDA)
#        define TIMEMORY_USE_CUDA_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_CUDA_EXTERN) && defined(TIMEMORY_USE_CUDA)
#        define TIMEMORY_USE_CUDA_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_CUPTI_EXTERN) && defined(TIMEMORY_USE_CUPTI)
#        define TIMEMORY_USE_CUPTI_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_HIP_EXTERN) && defined(TIMEMORY_USE_HIP)
#        define TIMEMORY_USE_HIP_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_GOTCHA_EXTERN) && defined(TIMEMORY_USE_GOTCHA)
#        define TIMEMORY_USE_GOTCHA_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_GPERFTOOLS_EXTERN) && defined(TIMEMORY_USE_GPERFTOOLS)
#        define TIMEMORY_USE_GPERFTOOLS_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_LIKWID_EXTERN) && defined(TIMEMORY_USE_LIKWID)
#        define TIMEMORY_USE_LIKWID_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_PAPI_EXTERN) && defined(TIMEMORY_USE_PAPI)
#        define TIMEMORY_USE_PAPI_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_TAU_EXTERN) && defined(TIMEMORY_USE_TAU)
#        define TIMEMORY_USE_TAU_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_VTUNE_EXTERN) && defined(TIMEMORY_USE_VTUNE)
#        define TIMEMORY_USE_VTUNE_EXTERN 1
#    endif
#    if !defined(TIMEMORY_USE_OMPT_EXTERN) && defined(TIMEMORY_USE_OMPT)
#        define TIMEMORY_USE_OMPT_EXTERN 1
#    endif
#endif

//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_ALLINEA_MAP_EXTERN)
#    include "timemory/components/allinea/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_CALIPER_EXTERN)
#    include "timemory/components/caliper/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_CRAYPAT_EXTERN)
#    include "timemory/components/craypat/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_CUDA_EXTERN)
#    include "timemory/components/cuda/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_CUPTI_EXTERN)
#    include "timemory/components/cupti/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_HIP_EXTERN)
#    include "timemory/components/hip/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_GOTCHA_EXTERN)
#    include "timemory/components/gotcha/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_GPERFTOOLS_EXTERN)
#    include "timemory/components/gperftools/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_LIKWID_EXTERN)
#    include "timemory/components/likwid/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_PAPI_EXTERN)
#    include "timemory/components/papi/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_PAPI_EXTERN) || defined(TIMEMORY_USE_CUPTI_EXTERN)
#    include "timemory/components/roofline/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_TAU_EXTERN)
#    include "timemory/components/tau_marker/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_VTUNE_EXTERN)
#    include "timemory/components/vtune/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_RUSAGE_EXTERN)
#    include "timemory/components/rusage/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_IO_EXTERN)
#    include "timemory/components/io/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_TIMING_EXTERN)
#    include "timemory/components/timing/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_NETWORK_EXTERN)
#    include "timemory/components/network/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_TRIP_COUNT_EXTERN)
#    include "timemory/components/trip_count/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_DATA_TRACKER_EXTERN)
#    include "timemory/components/data_tracker/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_OMPT_EXTERN)
#    include "timemory/components/ompt/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_TIMESTAMP_EXTERN)
#    include "timemory/components/timestamp/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
#    include "timemory/components/user_bundle/extern.hpp"
#endif
//
//--------------------------------------------------------------------------------------//
