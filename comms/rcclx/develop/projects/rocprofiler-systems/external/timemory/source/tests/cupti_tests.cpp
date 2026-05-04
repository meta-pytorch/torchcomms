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
static const auto num_blck = 64;
static const auto num_grid = 2;
static const auto epsilon  = 10 * std::numeric_limits<float>::epsilon();

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
        {
            T& _v = begin[i];
            _v += (static_cast<T>(2.0) * n);
        }
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
        if(i < n)
        {
            T& _v = begin[i];
            if(i < n / 2)
                _v *= static_cast<T>(2.0);
            else if(i >= n / 2)
                _v += static_cast<T>(3.0);
        }
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

class cupti_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, activity)
{
    using tuple_t            = tim::component_tuple_t<wall_clock, cupti_activity>;
    tim::settings::verbose() = 2;
    tim::settings::debug()   = true;

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

    int64_t sleep_msec = 700;

    tuple_t async_timer(details::get_test_name() + "_no_subtimers", true);
    async_timer.start();
    for(int i = 0; i < num_iter; ++i)
    {
        printf("[%s]> iteration %i...\n", __FUNCTION__, i);
        auto _stream = streams.at(i % nstream);
        details::KERNEL_A(data, num_data, _stream);
        details::KERNEL_B(data, num_data, _stream);
    }
    async_timer.stop();

    std::vector<tuple_t> subtimers;
    for(int i = 0; i < num_iter; ++i)
    {
        printf("[%s]> iteration %i...\n", __FUNCTION__, i);
        tuple_t subtimer(details::get_test_name() + "_itr", true);
        subtimer.start();
        auto _stream = streams.at(i % nstream);
        details::KERNEL_A(data, num_data, _stream);
        details::KERNEL_B(data, num_data, _stream);
        subtimer.stop();
        subtimers.push_back(subtimer);
    }
    // add in a specific amount of sleep to validate that not recording time on CPU
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
    timer.stop();

    int     kwidth = 60;
    tuple_t subtot(details::get_test_name() + "_itr_subtotal");
    for(const auto& itr : subtimers)
    {
        std::cout << itr << std::endl;
        // secondaries (individual kernels)
        std::cout << "Individual kernels:\n";
        for(const auto& sitr : itr.get<cupti_activity>()->get_secondary())
        {
            std::cout << "    " << std::setw(kwidth) << sitr.first << " : "
                      << std::setw(12) << std::setprecision(8) << std::fixed
                      << sitr.second << "\n";
        }
        std::cout << "\n";
        subtot += itr;
    }
    std::cout << subtot << std::endl;
    std::cout << timer << std::endl;

    // secondaries (individual kernels)
    std::cout << "Individual kernels:\n";
    for(const auto& itr : timer.get<cupti_activity>()->get_secondary())
    {
        std::cout << "    " << std::setw(kwidth) << itr.first << " : " << std::setw(12)
                  << std::setprecision(8) << std::fixed << itr.second << "\n";
    }
    std::cout << "\n";

    auto& rc = *timer.get<wall_clock>();
    auto& ca = *timer.get<cupti_activity>();

    double rc_msec = (rc.get() / cupti_activity::get_unit()) * tim::units::msec;
    double ca_msec = (ca.get() / cupti_activity::get_unit()) * tim::units::msec;

    double expected_diff = sleep_msec;
    double expected_tol  = 0.05 * expected_diff;
    double real_diff     = rc_msec - ca_msec;

    tim::device::gpu::free(data);
    tim::cuda::device_sync();
    cupti_activity::global_finalize();
    num_iter /= 2;

    EXPECT_NEAR(real_diff, expected_diff, expected_tol)
        << "real_clock: " << rc << ", cupti_activity: " << ca << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, available)
{
    CUdevice device;
    TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&device, 0));

    auto event_names  = tim::cupti::available_events(device);
    auto metric_names = tim::cupti::available_metrics(device);
    std::sort(event_names.begin(), event_names.end());
    std::sort(metric_names.begin(), metric_names.end());

    using size_type = decltype(event_names.size());
    size_type wevt  = 10;
    size_type wmet  = 10;
    for(const auto& itr : event_names)
        wevt = std::max(itr.size(), wevt);
    for(const auto& itr : metric_names)
        wmet = std::max(itr.size(), wmet);

    std::cout << "Event names: \n\t"
              << details::array_to_string(event_names, ", ", wevt, 200 / wevt)
              << std::endl;
    std::cout << "Metric names: \n\t"
              << details::array_to_string(metric_names, ", ", wmet, 200 / wmet)
              << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, kernels)
{
    cupti_event::get_device_initializer() = []() { return 0; };
    cupti_event::get_event_initializer()  = []() {
        return std::vector<std::string>(
            { "active_warps", "active_cycles", "global_load", "global_store" });
    };
    cupti_counters::get_metric_initializer() = []() {
        return std::vector<std::string>(
            { "inst_per_warp", "branch_efficiency", "warp_execution_efficiency",
              "flop_count_sp", "flop_count_sp_add", "flop_count_sp_mul",
              "flop_count_sp_fma", "flop_sp_efficiency", "flop_count_dp",
              "flop_count_dp_add", "flop_count_dp_mul", "flop_count_dp_fma",
              "flop_dp_efficiency", "gld_efficiency", "gst_efficiency", "ldst_executed",
              "ldst_issued" });
    };

    std::vector<float> cpu_data(num_data, 0);
    float*             data = tim::device::gpu::alloc<float>(num_data);
    tim::cuda::memcpy(data, cpu_data.data(), num_data, tim::cuda::host_to_device_v, 0);

    using tuple_t = tim::component_tuple_t<wall_clock, cupti_counters>;
    tuple_t timer(details::get_test_name(), true);

    timer.start();
    for(int i = 0; i < num_iter; ++i)
    {
        printf("[%s]> iteration %i...\n", __FUNCTION__, i);
        details::KERNEL_A(data, num_data);
        details::KERNEL_B(data, num_data);
    }
    timer.stop();
    std::cout << timer << std::endl;
    int kwidth = 40;
    // secondaries (individual kernels)
    std::cout << "Individual kernels:\n";
    for(const auto& itr : timer.get<cupti_counters>()->get_secondary())
    {
        std::stringstream ss_beg, ss_data;
        ss_beg << "    " << std::setw(kwidth) << itr.first;
        for(const auto& sitr : itr.second)
        {
            ss_data << " : " << std::setw(12) << std::setprecision(8) << std::fixed
                    << sitr << "\n";
            ss_data << std::setw(ss_beg.str().length() + 3);
        }
        std::cout << ss_beg.str() << ss_data.str();
    }
    std::cout << "\n";

    tim::device::gpu::free(data);
    tim::cuda::device_sync();
    tim::cuda::device_reset();
    printf("\n");

    auto cupti_data   = timer.get<cupti_counters>()->get();
    auto cupti_labels = timer.get<cupti_counters>()->label_array();
    std::cout << "CUPTI: data size = " << cupti_data.size()
              << ", label size = " << cupti_labels.size() << "\n"
              << std::endl;

    double num_active = num_blck * num_grid;
    double ratio      = num_data / static_cast<double>(num_active);

    auto A_glob_load  = (2.0 * num_grid) * ratio;
    auto B_glob_load  = (1.0 * num_blck / num_grid) * ratio;
    auto A_glob_store = (2.0 * num_grid) * ratio;
    auto B_glob_store = (1.0 * num_blck / num_grid) * ratio;
    // auto A_warp_eff      = (num_blck % 32 == 0) ? 1.0 : ((num_blck % 32) / 32.0);
    // auto B_warp_eff      = (num_grid % 32 == 0) ? 1.0 : ((num_grid % 32) / 32.0);
    auto A_flop_count_sp = 2.00 * num_data;
    auto B_flop_count_sp = 0.75 * num_data;
    // these are inherent to kernel design
    auto A_glob_load_eff  = 100.0;
    auto B_glob_load_eff  = 25.0;
    auto A_glob_store_eff = 100.0;
    auto B_glob_store_eff = 25.0;

    std::map<std::string, double> cupti_map;
    for(uint64_t i = 0; i < cupti_data.size(); ++i)
    {
        std::cout << "    " << std::setw(28) << cupti_labels[i] << " : " << std::setw(12)
                  << std::setprecision(6) << cupti_data[i] << std::endl;
        cupti_map[cupti_labels[i]] = cupti_data[i];
    }

    auto global_load  = num_iter * (A_glob_load + B_glob_load);
    auto global_store = num_iter * (A_glob_store + B_glob_store);
    // auto warp_eff         = 0.5 * (A_warp_eff + B_warp_eff) * 100.0;
    auto global_load_eff  = 0.5 * (A_glob_load_eff + B_glob_load_eff);
    auto global_store_eff = 0.5 * (A_glob_store_eff + B_glob_store_eff);
    auto flop_count_sp    = num_iter * (A_flop_count_sp + B_flop_count_sp);

    printf("A flop = %f\n", A_flop_count_sp);
    printf("B flop = %f\n", B_flop_count_sp);

    EXPECT_NEAR(cupti_map["global_load"], global_load, epsilon);
    EXPECT_NEAR(cupti_map["global_store"], global_store, epsilon);
    EXPECT_NEAR(cupti_map["flop_count_sp"], flop_count_sp, epsilon);
    EXPECT_NEAR(cupti_map["gld_efficiency"], global_load_eff, epsilon);
    EXPECT_NEAR(cupti_map["gst_efficiency"], global_store_eff, epsilon);
    // EXPECT_NEAR(cupti_map["warp_execution_efficiency"], warp_eff, epsilon);
}

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, streams)
{
    num_iter *= 2;
    TIMEMORY_CONFIGURE(
        cupti_event, 0,
        { "active_warps", "active_cycles", "global_load", "global_store" },
        { "inst_per_warp", "branch_efficiency", "warp_execution_efficiency",
          "flop_count_sp", "flop_count_sp_add", "flop_count_sp_mul", "flop_count_sp_fma",
          "flop_sp_efficiency", "flop_count_dp", "flop_count_dp_add", "flop_count_dp_mul",
          "flop_count_dp_fma", "flop_dp_efficiency", "gld_efficiency", "gst_efficiency",
          "ldst_executed", "ldst_issued" });

    // must initialize storage before creating the stream
    using tuple_t = tim::component_tuple_t<wall_clock, cupti_event>;

    tim::cuda::stream_t stream;
    tim::cuda::stream_create(stream);

    std::vector<float> cpu_data(num_data, 0);
    float*             data = tim::device::gpu::alloc<float>(num_data);
    tim::cuda::memcpy(data, cpu_data.data(), num_data, tim::cuda::host_to_device_v, 0);

    tuple_t timer(details::get_test_name(), true);
    timer.start();
    std::vector<tuple_t> subtimers;
    for(int i = 0; i < num_iter; ++i)
    {
        printf("[%s]> iteration %i...\n", __FUNCTION__, i);
        tuple_t subtimer(details::get_test_name() + "_itr", true);
        subtimer.start();
        details::KERNEL_A(data, num_data, stream);
        details::KERNEL_B(data, num_data, stream);
        subtimer.stop();
        std::cout << subtimer << std::endl;
        subtimers.push_back(subtimer);
    }
    timer.stop();
    tim::device::gpu::free(data);
    printf("\n");

    tuple_t subtot(details::get_test_name() + "_itr_subtotal");
    for(const auto& itr : subtimers)
        subtot += itr;
    std::cout << subtot << std::endl;
    std::cout << timer << std::endl;
    printf("\n");

    auto cupti_data   = timer.get<cupti_event>()->get();
    auto cupti_labels = timer.get<cupti_event>()->label_array();
    std::cout << "CUPTI: data size = " << cupti_data.size()
              << ", label size = " << cupti_labels.size() << "\n"
              << std::endl;

    double num_active = num_blck * num_grid;
    double ratio      = num_data / static_cast<double>(num_active);

    auto A_glob_load  = (2.0 * num_grid) * ratio;
    auto B_glob_load  = (1.0 * num_blck / num_grid) * ratio;
    auto A_glob_store = (2.0 * num_grid) * ratio;
    auto B_glob_store = (1.0 * num_blck / num_grid) * ratio;
    // auto A_warp_eff      = (num_blck % 32 == 0) ? 1.0 : ((num_blck % 32) / 32.0);
    // auto B_warp_eff      = (num_grid % 32 == 0) ? 1.0 : ((num_grid % 32) / 32.0);
    auto A_flop_count_sp = 2.00 * num_data;
    auto B_flop_count_sp = 0.75 * num_data;
    // these are inherent to kernel design
    auto A_glob_load_eff  = 100.0;
    auto B_glob_load_eff  = 25.0;
    auto A_glob_store_eff = 100.0;
    auto B_glob_store_eff = 25.0;

    std::map<std::string, double> cupti_map;
    for(uint64_t i = 0; i < cupti_data.size(); ++i)
    {
        std::cout << "    " << std::setw(28) << cupti_labels[i] << " : " << std::setw(12)
                  << std::setprecision(6) << cupti_data[i] << std::endl;
        cupti_map[cupti_labels[i]] = cupti_data[i];
    }

    auto global_load  = num_iter * (A_glob_load + B_glob_load);
    auto global_store = num_iter * (A_glob_store + B_glob_store);
    // auto warp_eff         = 0.5 * (A_warp_eff + B_warp_eff) * 100.0;
    auto global_load_eff  = 0.5 * (A_glob_load_eff + B_glob_load_eff);
    auto global_store_eff = 0.5 * (A_glob_store_eff + B_glob_store_eff);
    auto flop_count_sp    = num_iter * (A_flop_count_sp + B_flop_count_sp);

    printf("A flop = %f\n", A_flop_count_sp);
    printf("B flop = %f\n", B_flop_count_sp);

    EXPECT_NEAR(cupti_map["global_load"], global_load, epsilon);
    EXPECT_NEAR(cupti_map["global_store"], global_store, epsilon);
    EXPECT_NEAR(cupti_map["flop_count_sp"], flop_count_sp, epsilon);
    EXPECT_NEAR(cupti_map["gld_efficiency"], global_load_eff, epsilon);
    EXPECT_NEAR(cupti_map["gst_efficiency"], global_store_eff, epsilon);
    // EXPECT_NEAR(cupti_map["warp_execution_efficiency"], warp_eff, epsilon);
}

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, roofline_activity)
{
    using roofline_t = gpu_roofline<float>;
    using tuple_t    = tim::component_tuple_t<wall_clock, roofline_t>;

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

    int64_t sleep_msec = 700;

    roofline_t::configure(roofline_t::MODE::ACTIVITY);
    tuple_t timer(details::get_test_name(), true);
    timer.start();
    std::vector<tuple_t> subtimers;
    for(int i = 0; i < num_iter; ++i)
    {
        printf("[%s]> iteration %i...\n", __FUNCTION__, i);
        tuple_t subtimer(details::get_test_name() + "_itr", true);
        subtimer.start();
        auto _stream = streams.at(i % nstream);
        details::KERNEL_A(data, num_data, _stream);
        details::KERNEL_B(data, num_data, _stream);
        subtimer.stop();
        subtimers.push_back(subtimer);
    }
    // add in a specific amount of sleep to validate that not recording time on CPU
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
    timer.stop();

    tuple_t subtot(details::get_test_name() + "_itr_subtotal");
    for(const auto& itr : subtimers)
    {
        std::cout << itr << std::endl;
        subtot += itr;
    }
    std::cout << subtot << std::endl;
    std::cout << timer << std::endl;

    tim::device::gpu::free(data);
    tim::cuda::device_sync();
    num_iter /= 2;
}

//--------------------------------------------------------------------------------------//

TEST_F(cupti_tests, roofline_counters)
{
    using roofline_t = gpu_roofline<float>;
    using tuple_t    = tim::component_tuple_t<wall_clock, roofline_t>;

    tim::settings::ert_max_data_size_gpu()    = 100000000;
    tim::settings::ert_min_working_size_gpu() = 5000000;

    roofline_t::configure(roofline_t::MODE::COUNTERS);
    num_iter *= 2;

    std::vector<float> cpu_data(num_data, 0);
    float*             data = tim::device::gpu::alloc<float>(num_data);
    tim::cuda::memcpy(data, cpu_data.data(), num_data, tim::cuda::host_to_device_v);

    tuple_t timer(details::get_test_name(), true);
    timer.start();
    std::vector<tuple_t> subtimers;
    for(int i = 0; i < num_iter; ++i)
    {
        printf("[%s]> iteration %i...\n", __FUNCTION__, i);
        tuple_t subtimer(details::get_test_name() + "_itr", true);
        subtimer.start();
        details::KERNEL_A(data, num_data);
        details::KERNEL_B(data, num_data);
        subtimer.stop();
        std::cout << subtimer << std::endl;
    }
    timer.stop();

    tuple_t subtot(details::get_test_name() + "_itr_subtotal");
    for(const auto& itr : subtimers)
    {
        std::cout << itr << std::endl;
        subtot += itr;
    }
    std::cout << subtot << std::endl;
    std::cout << timer << std::endl;

    tim::device::gpu::free(data);
    tim::cuda::device_sync();
    num_iter /= 2;
    printf("\n");
}

//--------------------------------------------------------------------------------------//

#pragma nv_diag_suppress 177

namespace
{
auto library_init = (tim::set_env("TIMEMORY_CUPTI_ACTIVITY_LEVEL", "2", 1), true);
}  // namespace
