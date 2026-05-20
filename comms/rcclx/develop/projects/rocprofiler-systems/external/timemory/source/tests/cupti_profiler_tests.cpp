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

#if defined(DEBUG)
#    undef DEBUG
#endif

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
//
#include "timemory/components/cupti/cupti_profiler.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;

//--------------------------------------------------------------------------------------//

using default_device = tim::device::default_device;
using params_t       = tim::device::params<default_device>;

static const auto num_data = 96;
static auto       num_iter = 10;
static const auto num_blck = 32;
static const auto num_grid = 32;
// static const auto epsilon  = 10 * std::numeric_limits<float>::epsilon();

void
record_kernel(const char* kernel_name, int nreplays, void*)
{
    printf("replaying kernel: %s. # replays = %i\n", kernel_name, nreplays);
}

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
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
TIMEMORY_DEVICE_FUNCTION inline void
add_func(Tp& a, const Tp& b, const Tp& c)
{
    a = b + c;
}
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
TIMEMORY_DEVICE_FUNCTION inline void
fma_func(Tp& a, const Tp& b, const Tp& c)
{
    a = a * b + c;
}
//--------------------------------------------------------------------------------------//
//  print an array to a string
//
template <typename Tp>
std::string
array_to_string(const Tp& arr, const std::string& delimiter = ", ",
                const int& _width = 16, const int& _break = 8,
                const std::string& _break_delim = "\t")
{
    auto size      = std::distance(arr.begin(), arr.end());
    using int_type = decltype(size);
    std::stringstream ss;
    for(int_type i = 0; i < size; ++i)
    {
        ss << std::setw(_width) << arr.at(i);
        if(i + 1 < size)
            ss << delimiter;
        if((i + 1) % _break == 0 && (i + 1) < size)
            ss << "\n" << _break_delim;
    }
    return ss.str();
}
//======================================================================================//
namespace impl
{
template <typename T>
TIMEMORY_GLOBAL_FUNCTION void
KERNEL_A(T* begin, int n)
{
    auto range = tim::device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        if(i < n)
            *(begin + i) += 2.0f * n;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T>
TIMEMORY_GLOBAL_FUNCTION void
KERNEL_B(T* begin, int n)
{
    auto range = tim::device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        if(i < n / 2)
            *(begin + i) *= 2.0f;
        else if(i >= n / 2 && i < n)
            *(begin + i) += 3.0f;
    }
}
}  // namespace impl
//--------------------------------------------------------------------------------------//

template <typename T>
void
KERNEL_A(T* arr, int size, tim::cuda::stream_t stream = 0)
{
    // this kernel is designed for size / (64.0 * 2.0) operations
    // of 128
    params_t params(num_grid, num_blck, 0, stream);
    tim::device::launch(params, impl::KERNEL_A<T>, arr, size);
    // impl::KERNEL_A<T><<<num_grid, num_blck, 0, stream>>>(arr, size);
}

//--------------------------------------------------------------------------------------//
template <typename T>
void
KERNEL_B(T* arr, int size, tim::cuda::stream_t stream = 0)
{
    // this kernel is designed for (64.0 / 2.0) * (size / (64.0 * 2.0)) operations
    params_t params(num_blck, num_grid, 0, stream);
    tim::device::launch(params, impl::KERNEL_B<T>, arr, size / 2);
    // impl::KERNEL_B<T><<<num_blck, num_grid, 0, stream>>>(arr, size / 2);
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class cupti_profiler_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(cupti_profiler_tests, available)
{
    tim::settings::cupti_metrics() = "smsp__warps_launched.avg+";
    cupti_profiler::configure();

    auto chips = cupti_profiler::ListSupportedChips();
    for(const auto& itr : chips)
    {
        auto chip_metrics = cupti_profiler::ListMetrics(itr.c_str(), false);
        if(chip_metrics.size() > 0)
        {
            std::cout << "CHIP: " << itr << "\n";
            int n = 0;
            int t = chip_metrics.size();
            int w = 0;
            for(const auto& citr : chip_metrics)
                w = std::max<int>(citr.length(), w);
            w += 4;
            int ndiv = 240 / w;
            for(const auto& citr : chip_metrics)
            {
                if(n % ndiv == 0)
                    std::cout << "    ";
                std::cout << std::setw(w) << std::left << citr;
                if(n + 1 < t)
                    std::cout << ", ";
                if(n % ndiv == (ndiv - 1))
                    std::cout << '\n';
                ++n;
            }
        }
    }

    cupti_profiler::finalize();
}

//--------------------------------------------------------------------------------------//

TEST_F(cupti_profiler_tests, general)
{
    using tuple_t = tim::component_tuple_t<wall_clock, cupti_profiler>;

    tim::settings::cupti_metrics() =
        "smsp__warps_launched.avg,smsp__warps_launched.max,smsp__warps_launched.sum,smsp_"
        "_warps_launched_total.sum,smsp__warps_launched_total.max";
    // cupti_profiler::configure();

    tuple_t timer(details::get_test_name(), true);
    timer.start();

    num_iter *= 2;
    uint64_t                         nstream = 1;
    std::vector<tim::cuda::stream_t> streams(nstream);
    for(auto& itr : streams)
        tim::cuda::stream_create(itr);

    std::vector<float> cpu_data(num_data, 0);
    float*             data = tim::device::gpu::alloc<float>(num_data);
    for(uint64_t i = 0; i < nstream; ++i)
    {
        auto _off      = i * (num_data / nstream);
        auto _data     = data + _off;
        auto _cpu_data = cpu_data.data() + _off;
        auto _ndata    = (num_data / nstream);
        if(i + 1 == nstream)
            _ndata += num_data % nstream;
        tim::cuda::memcpy(_data, _cpu_data, _ndata, tim::cuda::host_to_device_v,
                          streams.at(i));
    }

    for(auto& itr : streams)
        tim::cuda::stream_sync(itr);

    // tuple_t async_timer(details::get_test_name() + "_no_subtimers", true);
    // async_timer.start();
    for(int i = 0; i < num_iter; ++i)
    {
        printf("[%s]> iteration %i...\n", details::get_test_name().c_str(), i);
        auto _stream = streams.at(i % nstream);
        details::KERNEL_A(data, num_data, _stream);
        details::KERNEL_B(data, num_data, _stream);
    }
    // async_timer.stop();
    timer.stop();

    tim::device::gpu::free(data);
    tim::cuda::device_sync();
    num_iter /= 2;
}

//--------------------------------------------------------------------------------------//

TEST_F(cupti_profiler_tests, nested)
{
    using tuple_t = tim::component_tuple_t<wall_clock, cupti_profiler>;

    tim::settings::cupti_metrics() =
        "smsp__warps_launched.avg,smsp__warps_launched.max,smsp__warps_launched.sum,smsp_"
        "_warps_launched_total.sum,smsp__warps_launched_total.max";

    tuple_t timer(details::get_test_name());
    timer.start();

    num_iter *= 2;
    uint64_t                         nstream = 1;
    std::vector<tim::cuda::stream_t> streams(nstream);
    for(auto& itr : streams)
        tim::cuda::stream_create(itr);

    std::vector<float> cpu_data(num_data, 0);
    float*             data = tim::device::gpu::alloc<float>(num_data);
    for(uint64_t i = 0; i < nstream; ++i)
    {
        auto _off      = i * (num_data / nstream);
        auto _data     = data + _off;
        auto _cpu_data = cpu_data.data() + _off;
        auto _ndata    = (num_data / nstream);
        if(i + 1 == nstream)
            _ndata += num_data % nstream;
        tim::cuda::memcpy(_data, _cpu_data, _ndata, tim::cuda::host_to_device_v,
                          streams.at(i));
    }

    for(auto& itr : streams)
        tim::cuda::stream_sync(itr);

    for(int i = 0; i < num_iter; ++i)
    {
        tuple_t subtimer(details::get_test_name() + "_subtimer");
        subtimer.start();
        printf("[%s]> iteration %i...\n", details::get_test_name().c_str(), i);
        auto _stream = streams.at(i % nstream);
        details::KERNEL_A(data, num_data, _stream);
        details::KERNEL_B(data, num_data, _stream);
        subtimer.stop();
    }
    timer.stop();

    tim::device::gpu::free(data);
    tim::cuda::device_sync();
    num_iter /= 2;
}

//--------------------------------------------------------------------------------------//
