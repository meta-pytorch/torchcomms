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

#include "gpu_common.hpp"
#include "gpu_device_timer.hpp"
#include "gpu_op_tracker.hpp"
#include "timemory/backends/device.hpp"
#include "timemory/backends/gpu.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/macros.hpp"
#include "timemory/timemory.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace device   = tim::device;
namespace mpl      = tim::mpl;
namespace comp     = tim::component;
namespace gpu      = tim::gpu;
namespace concepts = tim::concepts;

using default_device = device::default_device;
using tim::component_tuple;

namespace tim
{
namespace component
{
#if defined(TIMEMORY_USE_CUDA)
using gpu_marker = nvtx_marker;
using gpu_event  = cuda_event;
#else
using gpu_marker = roctx_marker;
using gpu_event  = hip_event;
#endif
}  // namespace component
}  // namespace tim

using gpu_device_timer_t = comp::gpu_device_timer::device_data;
using gpu_device_count_t = comp::gpu_op_tracker::device_data;

//--------------------------------------------------------------------------------------//
// saxpy calculation
//
TIMEMORY_GLOBAL_FUNCTION void
saxpy(int64_t n, float a, float* x, float* y)
{
    TIMEMORY_CODE(auto _htimer = device::handle<gpu_device_timer_t>::get());
    TIMEMORY_CODE(auto _hcount = device::handle<gpu_device_count_t>::get());

    auto range = device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        TIMEMORY_CODE(_hcount(2));
        y[i] = a * x[i] + y[i];
    }

    TIMEMORY_CODE(_htimer.stop());  // optional
}

//--------------------------------------------------------------------------------------//
// saxpy calculation
//
TIMEMORY_GLOBAL_FUNCTION void
saxpy_params(int64_t n, float a, float* x, float* y, gpu_device_timer_t* _timer,
             gpu_device_count_t* _count)
{
    TIMEMORY_CODE(auto _htimer = device::handle<gpu_device_timer_t>{ _timer });
    TIMEMORY_CODE(auto _hcount = device::handle<gpu_device_count_t>{ _count });

    auto range = device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        TIMEMORY_CODE(_hcount(2));
        y[i] = a * x[i] + y[i];
    }

    TIMEMORY_CODE(_htimer.stop());  // optional
}

//--------------------------------------------------------------------------------------//

void
run_saxpy(int nitr, int nstreams, int64_t num_threads, int64_t N);

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    using parser_t = tim::argparse::argument_parser;

    int           nitr     = 1000;
    int           nstream  = 1;
    int           npow     = 21;
    std::set<int> nthreads = { 32 };

    parser_t _parser{ "ex_kernel_instrument_v2" };
    _parser.add_argument({ "-n", "--num-iter" }, "Number of iterations")
        .dtype("int")
        .count(1)
        .action([&](parser_t& p) { nitr = p.get<int>("num-iter"); });
    _parser.add_argument({ "-s", "--num-streams" }, "Number of GPU streams")
        .dtype("int")
        .count(1)
        .action([&](parser_t& p) { nstream = p.get<int>("num-streams"); });
    _parser.add_argument({ "-t", "--num-threads" }, "Number of threads per block")
        .dtype("int")
        .action([&](parser_t& p) { nthreads = p.get<std::set<int>>("num-threads"); });
    _parser
        .add_argument({ "-p", "--num-pow" },
                      "Data size (powers of 2, e.g. '20' in 1 << 20)")
        .dtype("int")
        .action([&](parser_t& p) { npow = p.get<int>("num-pow"); });

    tim::timemory_init(argc, argv);
    tim::timemory_argparse(&argc, &argv, &_parser);

    for(auto bitr : nthreads)
        run_saxpy(nitr, nstream, bitr, 50 * (1 << npow));

    tim::timemory_finalize();
    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//

void
run_saxpy(int nitr, int nstreams, int64_t num_threads, int64_t N)
{
    using params_t = device::params<default_device>;
    using stream_t = default_device::stream_t;
    using bundle_t = component_tuple<comp::wall_clock, comp::cpu_clock, comp::cpu_util,
                                     comp::gpu_event, comp::gpu_marker,
                                     comp::gpu_device_timer, comp::gpu_op_tracker>;

    params_t params(params_t::compute(N, num_threads), num_threads);
    bundle_t _bundle{ __FUNCTION__ };
    _bundle.store(comp::gpu_event::explicit_streams_only{}, true);
    _bundle.start(device::gpu{}, params);

    float*                x         = device::cpu::alloc<float>(N);
    float*                y         = device::cpu::alloc<float>(N);
    float                 data_size = (3.0 * N * sizeof(float)) / tim::units::gigabyte;
    std::vector<stream_t> streams(std::max<int>(nstreams, 1), gpu::default_stream_v);
    for(auto& itr : streams)
    {
        gpu::stream_create(itr);
    }
    stream_t stream = streams.at(0);

    auto sync_streams = [&streams]() {
        for(auto& itr : streams)
            gpu::stream_sync(itr);
    };

    for(int i = 0; i < N; ++i)
    {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    float* d_x = device::gpu::alloc<float>(N);
    float* d_y = device::gpu::alloc<float>(N);
    TIMEMORY_GPU_RUNTIME_API_CALL(gpu::memcpy(d_x, x, N, gpu::host_to_device_v, stream));
    TIMEMORY_GPU_RUNTIME_API_CALL(gpu::memcpy(d_y, y, N, gpu::host_to_device_v, stream));

    sync_streams();
    for(int i = 0; i < 3 * nitr / 4; ++i)
    {
        auto& itr     = streams.at(i % streams.size());
        params.stream = itr;
        std::cout << TIMEMORY_JOIN(" ", "[saxpy] launching on", default_device::name(),
                                   "with parameters:", params)
                  << std::endl;

        // mark the start of a gpu event on the stream
        TIMEMORY_CODE(_bundle.mark_begin(mpl::piecewise_select<comp::gpu_event>{}, itr));
        // increment the total kernel counter in gpu_device_timer
        TIMEMORY_CODE(_bundle.mark(
            mpl::piecewise_select<comp::gpu_device_timer, comp::gpu_op_tracker>{}));

        saxpy<<<params.grid, params.block, 0, itr>>>(N, 1.0, d_x, d_y);

        // mark the end of a gpu event on the stream
        TIMEMORY_CODE(_bundle.mark_end(mpl::piecewise_select<comp::gpu_event>{}, itr));
    }

    tim::component_tuple<comp::gpu_device_timer, comp::gpu_op_tracker> _pbundle{
        "saxpy_params"
    };
    _pbundle.start(device::gpu{}, params);

    for(int i = 3 * nitr / 4; i < nitr; ++i)
    {
        auto  _get_device_data = [](auto* _v) { return _v->get_device_data(); };
        auto& itr              = streams.at(i % streams.size());
        params.stream          = itr;
        std::cout << TIMEMORY_JOIN(" ", "[saxpy_params] launching on",
                                   default_device::name(), "with parameters:", params)
                  << std::endl;

        // mark the start of a gpu event on the stream
        TIMEMORY_CODE(_bundle.mark_begin(mpl::piecewise_select<comp::gpu_event>{}, itr));
        // increment the total kernel counter in gpu_device_timer
        TIMEMORY_CODE(_pbundle.mark());

        saxpy_params<<<params.grid, params.block, 0, itr>>>(
            N, 1.0, d_x, d_y, _pbundle.get<comp::gpu_device_timer>(_get_device_data),
            _pbundle.get<comp::gpu_op_tracker>(_get_device_data));

        // mark the end of a gpu event on the stream
        TIMEMORY_CODE(_bundle.mark_end(mpl::piecewise_select<comp::gpu_event>{}, itr));
    }

    _pbundle.stop();
    sync_streams();

    TIMEMORY_GPU_RUNTIME_API_CALL(gpu::memcpy(y, d_y, N, gpu::device_to_host_v, stream));
    gpu::device_sync();
    _bundle.stop();
    gpu::free(d_x);
    gpu::free(d_y);

    float maxError = 0.0;
    float sumError = 0.0;
    for(int64_t i = 0; i < N; i++)
    {
        maxError = std::max<float>(maxError, std::abs(y[i] - 2.0f));
        sumError += (y[i] > 2.0f) ? (y[i] - 2.0f) : (2.0f - y[i]);
    }

    device::cpu::free(x);
    device::cpu::free(y);

    auto ce = _bundle.get<comp::gpu_event>();
    auto rc = _bundle.get<comp::wall_clock>();

    printf("Max error: %8.4e\n", (double) maxError);
    printf("Sum error: %8.4e\n", (double) sumError);
    printf("Total amount of data (GB): %f\n", (double) data_size);
    if(ce)
    {
        printf("Effective Bandwidth (GB/s): %f\n", (double) (data_size / ce->get()));
        printf("Kernel Runtime (sec): %16.12e\n", (double) ce->get());
    }
    if(rc)
        printf("Wall-clock time (sec): %16.12e\n", (double) rc->get());
    if(ce)
        std::cout << __FUNCTION__ << " gpu event : " << *ce << std::endl;
    if(rc)
        std::cout << __FUNCTION__ << " real clock: " << *rc << std::endl;
    std::cout << _bundle << std::endl;
    std::cout << std::endl;

    for(auto& itr : streams)
        gpu::stream_destroy(itr);
}
