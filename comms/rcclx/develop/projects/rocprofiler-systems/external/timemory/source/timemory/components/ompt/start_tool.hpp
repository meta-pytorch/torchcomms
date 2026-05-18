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

#include "timemory/defines.h"

#if defined(TIMEMORY_USE_OMPT)

#    include "timemory/components/base.hpp"
#    include "timemory/components/macros.hpp"
#    include "timemory/components/ompt/components.hpp"
#    include "timemory/components/ompt/tool.hpp"
#    include "timemory/components/ompt/types.hpp"

extern "C"
{
    // NOLINTNEXTLINE
    ompt_start_tool_result_t* ompt_start_tool(unsigned int omp_version,
                                              const char*  runtime_version)
    {
        TIMEMORY_PRINTF(stderr, "OpenMP version: %u, runtime version: %s\n", omp_version,
                        runtime_version);

        static auto ompt_initialize = [](ompt_function_lookup_t lookup,
                                         int                    initial_device_num,
                                         ompt_data_t*           tool_data) -> int {
            TIMEMORY_PRINTF(stderr, "OpenMP-tools configuring for initial device %i\n\n",
                            initial_device_num);
            auto _fini = tim::ompt::configure<TIMEMORY_OMPT_API_TAG>(
                lookup, initial_device_num, tool_data);
            if(_fini != nullptr && tim::manager::instance())
                tim::manager::instance()->add_cleanup("timemory-ompt", [_fini]() {
                    if(_fini)
                        (*_fini)();
                });
            return 1;  // success
        };

        static auto ompt_finalize = [](ompt_data_t* tool_data) {
            TIMEMORY_PRINTF(stderr, "OpenMP-tools finalized\n\n");
            tim::consume_parameters(tool_data);
        };

        static auto data =
            ompt_start_tool_result_t{ ompt_initialize, ompt_finalize, { 0 } };
        return (ompt_start_tool_result_t*) &data;
    }
}

#endif  // TIMEMORY_USE_OMPT
