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

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/backends/device.hpp"
#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;

//--------------------------------------------------------------------------------------//

static constexpr int     nitr     = 10;
static constexpr int     nstreams = 4;
static constexpr int64_t N        = 50 * (1 << 22);
using default_device              = tim::device::default_device;

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}
//--------------------------------------------------------------------------------------//
// saxpy calculation
//
TIMEMORY_GLOBAL_FUNCTION void
saxpy(int64_t n, float a, float* x, float* y)
{
    auto range = tim::device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        y[i] = a * x[i] + y[i];
    }
}
}  // namespace details

//--------------------------------------------------------------------------------------//

class cuda_tests : public ::testing::Test
{
protected:
    static void startup()
    {
        tim::cuda::device_query();
        prof = new tim::component_tuple<cuda_profiler, nvtx_marker>{ "cuda_tests" };
        prof->start();
    }

    static void shutdown()
    {
        prof->stop();
        delete prof;
    }

    TIMEMORY_TEST_SUITE_SETUP(startup())
    TIMEMORY_TEST_SUITE_TEARDOWN(shutdown())
    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN

    static tim::component_tuple<cuda_profiler, nvtx_marker>* prof;
};

tim::component_tuple<cuda_profiler, nvtx_marker>* cuda_tests::prof = nullptr;

//--------------------------------------------------------------------------------------//

TEST_F(cuda_tests, saxpy)
{
    using params_t = tim::device::params<default_device>;
    using stream_t = default_device::stream_t;
    using tuple_t =
        tim::component_tuple_t<wall_clock, cpu_clock, cpu_util, cuda_event, nvtx_marker>;

    tuple_t tot(details::get_test_name() + " total");
    tot.start();

    float*   x         = tim::device::cpu::alloc<float>(N);
    float*   y         = tim::device::cpu::alloc<float>(N);
    float    data_size = (3.0 * N * sizeof(float)) / tim::units::gigabyte;
    stream_t stream    = 0;
    params_t params(params_t::compute(N, 512), 512);

    std::cout << "\n"
              << details::get_test_name() << " launching on " << default_device::name()
              << " with parameters: " << params << "\n"
              << std::endl;

    for(int i = 0; i < N; ++i)
    {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    tuple_t dev(details::get_test_name() + " iterations + memory");
    dev.start();

    float* d_x = tim::device::gpu::alloc<float>(N);
    float* d_y = tim::device::gpu::alloc<float>(N);
    tim::cuda::memcpy(d_x, x, N, tim::cuda::host_to_device_v, stream);
    tim::cuda::memcpy(d_y, y, N, tim::cuda::host_to_device_v, stream);

    tuple_t bw(details::get_test_name() + " iterations");
    for(int i = 0; i < nitr; ++i)
    {
        bw.start();
        dev.mark_begin();
        tim::device::launch(params, details::saxpy, N, 1.0, d_x, d_y);
        dev.mark_end();
        bw.stop();
    }

    tim::cuda::memcpy(y, d_y, N, tim::cuda::device_to_host_v, stream);
    tim::cuda::device_sync();
    tim::cuda::free(d_x);
    tim::cuda::free(d_y);

    dev.stop();

    std::this_thread::sleep_for(std::chrono::seconds(1));
    float maxError = 0.0;
    float sumError = 0.0;
    for(int64_t i = 0; i < N; i++)
    {
        maxError = std::max<float>(maxError, std::abs(y[i] - 2.0f));
        sumError += (y[i] > 2.0f) ? (y[i] - 2.0f) : (2.0f - y[i]);
    }

    tim::device::cpu::free(x);
    tim::device::cpu::free(y);
    tot.stop();
    tim::cuda::device_reset();

#if defined(TIMEMORY_USE_CUDA)
    auto ce = *bw.get<cuda_event>();
#else
    auto ce = *bw.get<wall_clock>();
#endif
    auto rc = *bw.get<wall_clock>();

    printf("Max error: %8.4e\n", (double) maxError);
    printf("Sum error: %8.4e\n", (double) sumError);
    printf("Total amount of data (GB): %f\n", (double) data_size);
    printf("Effective Bandwidth (GB/s): %f\n", (double) (data_size / ce.get()));
    printf("Kernel Runtime (sec): %16.12e\n", (double) ce.get());
    printf("Wall-clock time (sec): %16.12e\n", (double) rc.get());
    std::cout << details::get_test_name() << " cuda event: " << ce << std::endl;
    std::cout << details::get_test_name() << " real clock: " << rc << std::endl;
    std::cout << tot << std::endl;
    std::cout << dev << std::endl;
    std::cout << bw << std::endl;
    std::cout << std::endl;
    ASSERT_NEAR(ce.get(), rc.get(), 2.0e-3);
}

//--------------------------------------------------------------------------------------//

TEST_F(cuda_tests, saxpy_streams)
{
    using params_t = tim::device::params<default_device>;
    using stream_t = default_device::stream_t;
    using tuple_t =
        tim::component_tuple_t<wall_clock, cpu_clock, cpu_util, cuda_event, nvtx_marker>;

    tuple_t tot(details::get_test_name() + " total");
    tot.start();

    std::vector<stream_t> streams(nstreams);
    for(auto& itr : streams)
        tim::cuda::stream_create(itr);

    auto sync_streams = [&]() {
        for(auto& itr : streams)
            tim::cuda::stream_sync(itr);
    };

    float*   x         = tim::device::cpu::alloc<float>(N);
    float*   y         = tim::device::cpu::alloc<float>(N);
    float    data_size = (3.0 * N * sizeof(float)) / tim::units::gigabyte;
    params_t params(params_t::compute(N, 512), 512);

    std::cout << "\n"
              << details::get_test_name() << " launching on " << default_device::name()
              << " with parameters: " << params << "\n"
              << std::endl;

    for(int i = 0; i < N; ++i)
    {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    TIMEMORY_BLANK_CALIPER(mem, tuple_t, "memory");
    auto& mem = TIMEMORY_CALIPER_REFERENCE(mem);
    mem.start();

    TIMEMORY_BLANK_CALIPER(dev, tuple_t, "iterations + memory");
    auto& dev = TIMEMORY_CALIPER_REFERENCE(dev);
    dev.start();

    float* d_x = tim::device::gpu::alloc<float>(N);
    float* d_y = tim::device::gpu::alloc<float>(N);

    TIMEMORY_CALIPER_APPLY(mem, mark_begin, streams[0]);
    tim::cuda::memcpy(d_x, x, N, tim::cuda::host_to_device_v, streams[0]);
    TIMEMORY_CALIPER_APPLY(mem, mark_end, streams[0]);

    TIMEMORY_CALIPER_APPLY(mem, mark_begin, streams[1]);
    tim::cuda::memcpy(d_y, y, N, tim::cuda::host_to_device_v, streams[1]);
    TIMEMORY_CALIPER_APPLY(mem, mark_end, streams[1]);

    sync_streams();
    tuple_t bw("iterations");

    for(int i = 0; i < nitr; ++i)
    {
        bw.start();
        params.stream = streams[i % streams.size()];
        TIMEMORY_CALIPER_APPLY(dev, mark_begin, params.stream);
        tim::device::launch(params, details::saxpy, N, 1.0, d_x, d_y);
        TIMEMORY_CALIPER_APPLY(dev, mark_end, params.stream);
        bw.stop();
    }

    TIMEMORY_CALIPER_APPLY(mem, mark_begin, streams[0]);
    tim::cuda::memcpy(y, d_y, N, tim::cuda::device_to_host_v, streams[0]);
    TIMEMORY_CALIPER_APPLY(mem, mark_end, streams[0]);

    sync_streams();
    tim::cuda::device_sync();
    tim::cuda::free(d_x);
    tim::cuda::free(d_y);

    mem.stop();
    dev.stop();

    std::this_thread::sleep_for(std::chrono::seconds(1));
    float maxError = 0.0;
    float sumError = 0.0;
    for(int64_t i = 0; i < N; i++)
    {
        maxError = std::max<float>(maxError, std::abs(y[i] - 2.0f));
        sumError += (y[i] > 2.0f) ? (y[i] - 2.0f) : (2.0f - y[i]);
    }

    tim::device::cpu::free(x);
    tim::device::cpu::free(y);
    tot.stop();
    tim::cuda::device_reset();

#if defined(TIMEMORY_USE_CUDA)
    auto ce = *bw.get<cuda_event>();
#else
    auto ce = *bw.get<wall_clock>();
#endif
    auto rc = *bw.get<wall_clock>();

    printf("Max error: %8.4e\n", (double) maxError);
    printf("Sum error: %8.4e\n", (double) sumError);
    printf("Total amount of data (GB): %f\n", (double) data_size);
    printf("Effective Bandwidth (GB/s): %f\n", (double) (data_size / ce.get()));
    printf("Kernel Runtime (sec): %16.12e\n", (double) ce.get());
    printf("Wall-clock time (sec): %16.12e\n", (double) rc.get());
    std::cout << details::get_test_name() << " cuda event: " << ce << std::endl;
    std::cout << details::get_test_name() << " real clock: " << rc << std::endl;
    std::cout << tot << std::endl;
    std::cout << mem << std::endl;
    std::cout << dev << std::endl;
    std::cout << bw << std::endl;
    std::cout << std::endl;
    ASSERT_NEAR(ce.get(), rc.get(), 2.0e-3);
}

//--------------------------------------------------------------------------------------//
