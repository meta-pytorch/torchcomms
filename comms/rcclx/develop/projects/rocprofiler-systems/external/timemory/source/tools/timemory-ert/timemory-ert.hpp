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
//

#pragma once

#include "timemory/backends/gpu.hpp"
#include "timemory/components/timing/ert_timer.hpp"
#include "timemory/ert.hpp"
#include "timemory/mpl.hpp"

#include <cstdint>
#include <set>

#if !defined(TIMEMORY_USE_CUDA_HALF)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tim::gpu::fp16_t, false_type)
#endif

#if !defined(TIMEMORY_USE_GPU)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tim::device::gpu, false_type)
#endif

//--------------------------------------------------------------------------------------//
// make the namespace usage a little clearer
//
namespace ert    = tim::ert;
namespace device = tim::device;
namespace gpu    = tim::gpu;
namespace trait  = tim::trait;

using settings = tim::settings;

//--------------------------------------------------------------------------------------//
// timemory has a backends that will not call MPI_Init, cudaDeviceCount, etc.
// when that library/package is not available
//
namespace dmp = tim::dmp;

//--------------------------------------------------------------------------------------//
// some short-hand aliases
//
using counter_type   = tim::component::ert_timer;
using fp16_t         = tim::gpu::fp16_t;
using ert_data_t     = ert::exec_data<counter_type>;
using ert_data_ptr_t = std::shared_ptr<ert_data_t>;

//--------------------------------------------------------------------------------------//
//  this will invoke ERT with the specified settings
//
template <typename Tp, typename DeviceT>
void
run_ert(
    ert_data_ptr_t&, int64_t num_threads, int64_t min_size, int64_t max_data,
    int64_t num_streams = 0, int64_t block_size = 0, int64_t num_gpus = 0,
    std::enable_if_t<
        trait::is_available<Tp>::value && trait::is_available<DeviceT>::value, int> = 0);

template <typename Tp, typename DeviceT>
void
run_ert(
    ert_data_ptr_t&, int64_t, int64_t, int64_t, int64_t = 0, int64_t = 0, int64_t = 0,
    std::enable_if_t<
        !trait::is_available<Tp>::value || !trait::is_available<DeviceT>::value, int> = 0)
{}

extern template void
run_ert<float, device::cpu>(ert_data_ptr_t&, int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int);
extern template void
run_ert<double, device::cpu>(ert_data_ptr_t&, int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int);
extern template void
run_ert<gpu::fp16_t, device::gpu>(ert_data_ptr_t&, int64_t, int64_t, int64_t, int64_t,
                                  int64_t, int64_t, int);
extern template void
run_ert<float, device::gpu>(ert_data_ptr_t&, int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int);
extern template void
run_ert<double, device::gpu>(ert_data_ptr_t&, int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int);

//--------------------------------------------------------------------------------------//

void*
start_generic(const std::string&);
void
stop_generic(void*);
bool
get_verbose();

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT>
void
run_ert(ert_data_ptr_t& data, int64_t num_threads, int64_t min_size, int64_t max_data,
        int64_t num_streams, int64_t block_size, int64_t num_gpus,
        std::enable_if_t<
            trait::is_available<Tp>::value && trait::is_available<DeviceT>::value, int>)
{
    // create a label for this test
    auto dtype = tim::demangle<Tp>();
    auto htype = DeviceT::name();
    auto label = TIMEMORY_JOIN("_", __FUNCTION__, dtype, htype, num_threads, "threads",
                               min_size, "min-ws", max_data, "max-size");

    if(std::is_same<DeviceT, device::gpu>::value)
    {
        label = TIMEMORY_JOIN("_", label, num_gpus, "gpus", num_streams, "streams",
                              block_size, "thr-per-blk");
    }

    if(dmp::rank() == 0)
        printf("[timemory-ert]> Executing %s...", label.c_str());

    using ert_executor_type = ert::executor<DeviceT, Tp, counter_type>;
    using ert_config_type   = typename ert_executor_type::configuration_type;
    using ert_counter_type  = ert::counter<DeviceT, Tp, counter_type>;

    //
    // simple modifications to override method number of threads, number of streams,
    // block size, minimum working set size, and max data size
    //
    ert_config_type::get_num_threads()      = [=]() { return num_threads; };
    ert_config_type::get_num_streams()      = [=]() { return num_streams; };
    ert_config_type::get_block_size()       = [=]() { return block_size; };
    ert_config_type::get_min_working_size() = [=]() { return min_size; };
    ert_config_type::get_max_data_size()    = [=]() { return max_data; };

    //
    // create a callback function that sets the device based on the thread-id
    //
    auto set_counter_device = [=](uint64_t tid, ert_counter_type&) {
        if(num_gpus > 0)
            gpu::set_device(tid % num_gpus);
    };

    //
    // create a configuration object -- this handles a lot of the setup
    //
    ert_config_type config;
    //
    // "construct" an ert::executor that executes the configuration and inserts
    // into the data object.
    //
    // NOTE: the ert::executor has callbacks that allows one to customize the
    //       the ERT execution
    //
    auto* _generic = start_generic(label);
    ert_executor_type{ config, data, set_counter_device };
    stop_generic(_generic);
    if(dmp::rank() == 0)
    {
        if(data && get_verbose())
            std::cout << "\n" << *(data) << std::endl;
        printf("\n");
    }
}
