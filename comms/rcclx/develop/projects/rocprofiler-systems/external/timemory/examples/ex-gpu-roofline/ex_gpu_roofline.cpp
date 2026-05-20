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

#if ROOFLINE_FP_BYTES == 2
#    define TIMEMORY_USE_CUDA_HALF
#endif

#include "timemory/ert.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/utility/testing.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>

using namespace tim::component;

using device_t       = tim::device::gpu;
using params_t       = tim::device::params<device_t>;
using stream_t       = tim::cuda::stream_t;
using default_device = device_t;

#if defined(TIMEMORY_USE_CUDA_HALF)
using fp16_t = tim::cuda::fp16_t;
#endif

using simple_timer_t = tim::auto_tuple_t<wall_clock>;

//--------------------------------------------------------------------------------------//

#if ROOFLINE_FP_BYTES == 8

using gpu_roofline_t = gpu_roofline_dp_flops;

#elif ROOFLINE_FP_BYTES == 4

using gpu_roofline_t = gpu_roofline_sp_flops;

#elif(ROOFLINE_FP_BYTES == 2) && defined(TIMEMORY_USE_CUDA_HALF)

using gpu_roofline_t = gpu_roofline_hp_flops;

#else

using gpu_roofline_t = gpu_roofline_flops;

#endif

//--------------------------------------------------------------------------------------//

using auto_tuple_t = tim::auto_tuple<wall_clock, cpu_clock, cpu_util, gpu_roofline_t>;

//--------------------------------------------------------------------------------------//
// amypx calculation
//
template <typename Tp>
TIMEMORY_GLOBAL_FUNCTION void
amypx(int64_t n, Tp* x, Tp* y, int64_t nitr)
{
    auto range = tim::device::grid_strided_range<default_device, 0, int32_t>(n);
    for(int64_t j = 0; j < (nitr % 10 + 1); ++j)
    {
        for(int i = range.begin(); i < range.end(); i += range.stride())
            y[i] = static_cast<Tp>(2.0) * y[i] + x[i];
    }
}

//--------------------------------------------------------------------------------------//
// amypx calculation
//
#if defined(TIMEMORY_USE_CUDA_HALF)
template <>
TIMEMORY_GLOBAL_FUNCTION void
amypx(int64_t n, fp16_t* x, fp16_t* y, int64_t nitr)
{
    auto range = tim::device::grid_strided_range<default_device, 0, int32_t>(n);
    for(int64_t j = 0; j < nitr; ++j)
    {
        for(int i = range.begin(); i < range.end(); i += range.stride())
        {
            fp16_t a = { 2.0, 2.0 };
            y[i]     = a * y[i] + x[i];
        }
    }
}
#endif

//--------------------------------------------------------------------------------------//

template <typename Tp>
void customize_roofline(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
exec_amypx(int64_t data_size, int64_t nitr, params_t params,
           std::vector<stream_t>& streams)
{
    auto label = TIMEMORY_JOIN("_", data_size, nitr, tim::demangle(typeid(Tp).name()));

    Tp* y    = tim::cuda::malloc<Tp>(data_size * streams.size());
    Tp* x    = tim::cuda::malloc<Tp>(data_size * streams.size());
    Tp  zero = 0.0;
    Tp  one  = 1.0;

    for(uint64_t i = 0; i < streams.size(); ++i)
    {
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

        TIMEMORY_BLANK_CALIPER(0, auto_tuple_t, "amypx_", label);
        tim::device::launch(data_size, stream, params, amypx<Tp>, data_size, _x, _y,
                            nitr);
        tim::cuda::stream_sync(stream);
        TIMEMORY_CALIPER_APPLY(0, stop);
    }

    tim::cuda::free(x);
    tim::cuda::free(y);
}

//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_CUDA_HALF)
template <>
void
exec_amypx<fp16_t>(int64_t data_size, int64_t nitr, params_t params,
                   std::vector<stream_t>& streams)
{
    using Tp   = fp16_t;
    auto label = TIMEMORY_JOIN("_", data_size, nitr, tim::demangle(typeid(Tp).name()));
    // while(label.find("__") != std::string::npos)
    //     label.erase(label.find("__"), 1);

    Tp* y    = tim::cuda::malloc<Tp>(data_size * streams.size());
    Tp* x    = tim::cuda::malloc<Tp>(data_size * streams.size());
    Tp  zero = { 0.0, 0.0 };
    Tp  one  = { 1.0, 1.0 };

    for(uint64_t i = 0; i < streams.size(); ++i)
    {
        stream_t stream = streams.at(i);
        auto     _y     = y + data_size * i;
        auto     _x     = x + data_size * i;
        tim::device::launch(data_size, stream, params, amypx<Tp>, data_size, _x, _y, 1);
        tim::cuda::stream_sync(stream);
        tim::device::launch(data_size, stream, params,
                            tim::ert::initialize_buffer<device_t, Tp, int64_t>, _y, zero,
                            data_size);
        tim::device::launch(data_size, stream, params,
                            tim::ert::initialize_buffer<device_t, Tp, int64_t>, _x, one,
                            data_size);
        tim::cuda::stream_sync(stream);

        TIMEMORY_BLANK_CALIPER(0, auto_tuple_t, "amypx_", label);
        tim::device::launch(data_size, stream, params, amypx<Tp>, data_size, _x, _y,
                            nitr);
        tim::cuda::stream_sync(stream);
        TIMEMORY_CALIPER_APPLY(0, stop);
    }

    tim::cuda::free(x);
    tim::cuda::free(y);
}
#endif
//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::settings::json_output()          = true;
    tim::settings::instruction_roofline() = true;
    tim::timemory_init(argc, argv);
    tim::cuda::device_query();
    tim::cuda::set_device(0);
    tim::signals::enable_signal_detection();

    int64_t num_threads   = 1;                           // default number of threads
    int64_t num_streams   = 8;                           // default number of streams
    int64_t working_size  = 100 * tim::units::megabyte;  // default working set size
    int64_t memory_factor = 5;                           // default multiple of 100 MB
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

#if defined(TIMEMORY_USE_CUDA_HALF)
    customize_roofline<fp16_t>(num_threads, working_size, memory_factor, num_streams,
                               grid_size, block_size);
#endif

    customize_roofline<float>(num_threads, working_size, memory_factor, num_streams,
                              grid_size, block_size);
    customize_roofline<double>(num_threads, working_size, memory_factor, num_streams,
                               grid_size, block_size);

    // ensure cpu version also runs the same number of threads
    tim::settings::ert_num_threads_cpu() = num_threads;

    params_t params(grid_size, block_size, 0, 0);

    wall_clock total;
    total.start();

    //
    // overall timing
    //
    auto _main = TIMEMORY_BLANK_HANDLE(auto_tuple_t, argv[0]);
    //_main.report_at_exit(true);

    //
    // run amypx calculations
    //

#if defined(TIMEMORY_USE_CUDA_HALF)
    {
        printf("Executing fp16 routines...\n");
        simple_timer_t routine("fp16", tim::scope::tree{}, true);
        exec_amypx<fp16_t>(data_size / 2, iterations * 2, params, streams);
        exec_amypx<fp16_t>(data_size * 1, iterations / 1, params, streams);
        exec_amypx<fp16_t>(data_size * 2, iterations / 2, params, streams);
    }
#endif

    {
        printf("Executing fp32 routines...\n");
        simple_timer_t routine("fp32", tim::scope::tree{}, true);
        exec_amypx<float>(data_size / 2, iterations * 2, params, streams);
        exec_amypx<float>(data_size * 1, iterations / 1, params, streams);
        exec_amypx<float>(data_size * 2, iterations / 2, params, streams);
    }

    {
        printf("Executing fp64 routines...\n");
        simple_timer_t routine("fp64", tim::scope::tree{}, true);
        exec_amypx<double>(data_size / 2, iterations * 2, params, streams);
        exec_amypx<double>(data_size * 1, iterations / 1, params, streams);
        exec_amypx<double>(data_size * 2, iterations / 2, params, streams);
    }

    //
    // stop the overall timing
    //
    printf("Stopping main timer...\n");
    _main.stop();

    //
    // overall timing
    //
    printf("Stopping total timer...\n");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    total.stop();

    std::cout << "Total time: " << total << std::endl;

    printf("Finalizing timemory...\n");
    tim::timemory_finalize();
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
customize_roofline(int64_t num_threads, int64_t working_size, int64_t memory_factor,
                   int64_t num_streams, int64_t grid_size, int64_t block_size)
{
    using namespace tim;
    using counter_t         = component::wall_clock;
    using ert_data_t        = ert::exec_data<counter_t>;
    using ert_params_t      = ert::exec_params;
    using ert_data_ptr_t    = std::shared_ptr<ert_data_t>;
    using ert_executor_type = ert::executor<device_t, Tp, counter_t>;
    using ert_config_type   = typename ert_executor_type::configuration_type;
    using ert_counter_type  = typename ert_executor_type::counter_type;

    //
    // simple modifications to override method number of threads, number of streams,
    // block size, and grid size
    //
    ert_config_type::configure(num_threads, 64, num_streams, block_size, grid_size);

    //
    // fully customize the roofline
    //
    if(tim::get_env<bool>("CUSTOMIZE_ROOFLINE", false))
    {
        // overload the finalization function that runs ERT calculations
        ert_config_type::get_executor() = [=](ert_data_ptr_t data) {
            auto lm_size = 100 * units::megabyte;
            // create the execution parameters
            ert_params_t params(working_size, memory_factor * lm_size, num_threads,
                                num_streams, grid_size, block_size);
            // create the operation _counter
            ert_counter_type _counter(params, data, 64);
            std::cout << _counter << std::endl;
            return _counter;
        };

        // does the execution of ERT
        auto callback = [=](ert_counter_type& _counter) {
            // these are the kernel functions we want to calculate the peaks with
            auto store_func = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b) { a = b; };
            auto add_func   = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
                a = b + c;
            };
            auto fma_func = [] TIMEMORY_LAMBDA(Tp & a, const Tp& b, const Tp& c) {
                a = a * b + c;
            };

            // set bytes per element
            _counter.bytes_per_element = sizeof(Tp);
            // set number of memory accesses per element from two functions
            _counter.memory_accesses_per_element = 2;

            // set the label
            _counter.label = "scalar_add";
            // run the operation _counter kernels
            tim::ert::ops_main<1>(_counter, add_func, store_func);

            // set the label
            _counter.label = "vector_fma";
            tim::ert::ops_main<2, 4, 8, 16, 32, 64, 128, 256, 512>(_counter, fma_func,
                                                                   store_func);
        };

        // set the callback
        gpu_roofline_t::set_executor_callback<Tp>(callback);
    }
}

//--------------------------------------------------------------------------------------//
