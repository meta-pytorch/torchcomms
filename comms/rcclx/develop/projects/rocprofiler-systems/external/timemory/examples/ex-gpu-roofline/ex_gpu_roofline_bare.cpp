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

#if !defined(DISABLE_TIMEMORY)
#    define DISABLE_TIMEMORY
#endif

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/device.hpp"
#include "timemory/backends/mpi.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/testing.hpp"
#include "timemory/utility/utility.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>

using device_t       = tim::device::gpu;
using params_t       = tim::device::params<device_t>;
using stream_t       = tim::cuda::stream_t;
using default_device = device_t;

//--------------------------------------------------------------------------------------//
// amypx calculation
//
template <typename Tp>
TIMEMORY_GLOBAL_FUNCTION void
amypx(int64_t n, Tp* x, Tp* y, int64_t nitr)
{
    auto range = tim::device::grid_strided_range<default_device, 0, int32_t>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
        y[i] = static_cast<Tp>(2.0) * y[i] + x[i];
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
exec_amypx(int64_t data_size, int64_t nitr, params_t params,
           std::vector<stream_t>& streams)
{
    Tp* y    = tim::cuda::malloc<Tp>(data_size * streams.size());
    Tp* x    = tim::cuda::malloc<Tp>(data_size * streams.size());
    Tp  zero = 0.0;
    Tp  one  = 1.0;

    for(uint64_t i = 0; i < streams.size(); ++i)
    {
        printf("Launching stream %lu with data size = %lu and %lu iterations\n",
               (unsigned long) i, (unsigned long) data_size, (unsigned long) nitr);

        stream_t stream = streams.at(i);
        auto     _y     = y + data_size * i;
        auto     _x     = x + data_size * i;
        params.stream   = stream;
        tim::device::launch(data_size, stream, params, amypx<Tp>, data_size, _x, _y, 1);
        tim::cuda::stream_sync(stream);
        tim::device::launch(data_size, stream, params,
                            tim::ert::initialize_buffer<device_t, Tp, int64_t>, _y, zero,
                            data_size);
        tim::device::launch(data_size, stream, params,
                            tim::ert::initialize_buffer<device_t, Tp, int64_t>, _x, one,
                            data_size);
        tim::cuda::stream_sync(stream);

        tim::device::launch(data_size, stream, params, amypx<Tp>, data_size, _x, _y,
                            nitr);
    }

    for(auto itr : streams)
        tim::cuda::stream_sync(itr);

    tim::cuda::free(x);
    tim::cuda::free(y);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::mpi::initialize(argc, argv);
    tim::timemory_init(argc, argv);

    tim::component::wall_clock wc;
    wc.start();

    int64_t num_threads =
        tim::threading::affinity::hw_physicalcpu();      // default number of threads
    int64_t num_streams   = 1;                           // default number of streams
    int64_t working_size  = 500 * tim::units::megabyte;  // default working set size
    int64_t memory_factor = 10;                          // default multiple of 500 MB
    int64_t iterations    = 1000;
    int64_t block_size    = 1024;
    int64_t grid_size     = 0;
    int64_t data_size     = 10000000;

    auto usage = [&]() {
        using lli = long long int;
        printf("\n%s [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] [%s = %lli] [%s = "
               "%lli] "
               "[%s = %lli] [%s = %lli]\n\n",
               argv[0], "num_threads", (lli) num_threads, "num_streams",
               (lli) num_streams, "working_size", (lli) working_size, "memory_factor",
               (lli) memory_factor, "iterations", (lli) iterations, "block_size",
               (lli) block_size, "grid_size", (lli) grid_size, "data_size",
               (lli) data_size);
    };

    for(auto i = 0; i < argc; ++i)
        if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            usage();
            exit(EXIT_SUCCESS);
        }

    int curr_arg = 1;
    if(argc > curr_arg)
        num_threads = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        num_streams = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        working_size = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        memory_factor = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        iterations = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        block_size = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        grid_size = atol(argv[curr_arg++]);
    if(argc > curr_arg)
        data_size = atol(argv[curr_arg++]);

    usage();

    std::vector<stream_t> streams(num_streams);
    for(auto& itr : streams)
        tim::cuda::stream_create(itr);

    params_t params(grid_size, block_size, 0, 0);

    //
    // run amypx calculations
    //
    for(int j = 1; j <= 10; ++j)
    {
        auto i = (j == 0) ? 1 : j;
        printf("Executing fp32 routines...\n");
        exec_amypx<float>(data_size * (4 * i), iterations * (2 * i), params, streams);
        exec_amypx<float>(data_size * (1 * i), iterations * (1 * i), params, streams);
        exec_amypx<float>(data_size * (2 * i), iterations * (4 * i), params, streams);

        printf("Executing fp64 routines...\n");
        exec_amypx<double>(data_size * (4 * i), iterations * (2 * i), params, streams);
        exec_amypx<double>(data_size * (1 * i), iterations * (1 * i), params, streams);
        exec_amypx<double>(data_size * (2 * i), iterations * (4 * i), params, streams);
    }

    wc.stop();

    printf("\nTotal time: %f seconds\n", wc.get());

    tim::timemory_finalize();
    tim::mpi::finalize();
}

//--------------------------------------------------------------------------------------//
