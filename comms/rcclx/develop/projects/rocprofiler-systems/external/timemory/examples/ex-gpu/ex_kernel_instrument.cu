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

#include "timemory/backends/device.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/macros.hpp"
#include "timemory/timemory.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace device = tim::device;
namespace mpl    = tim::mpl;
namespace comp   = tim::component;
namespace gpu    = tim::gpu;

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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ long long int
atomicAdd(long long int* address, long long int val);
#endif

#if !defined(TIMEMORY_USE_GPU)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_device_timer, false_type)
#endif

TIMEMORY_DECLARE_COMPONENT(gpu_device_timer)
TIMEMORY_SET_COMPONENT_API(component::gpu_device_timer, device::gpu, category::external,
                           os::agnostic)
TIMEMORY_STATISTICS_TYPE(component::gpu_device_timer, std::vector<float>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::gpu_device_timer, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::gpu_device_timer, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::gpu_device_timer, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(echo_enabled, component::gpu_device_timer, false_type)

namespace tim
{
namespace component
{
struct gpu_device_timer : base<gpu_device_timer, std::vector<long long int>>
{
    using value_type = std::vector<long long int>;
    using base_type  = base<gpu_device_timer, value_type>;

    gpu_device_timer() = default;
    ~gpu_device_timer();
    gpu_device_timer(gpu_device_timer&&) = default;
    gpu_device_timer& operator=(gpu_device_timer&&) = default;
    gpu_device_timer(const gpu_device_timer& rhs);
    gpu_device_timer& operator=(const gpu_device_timer& rhs);

    static std::string              label();
    static std::string              description();
    static std::vector<std::string> label_array();
    static std::vector<int64_t>     unit_array();
    static std::vector<std::string> display_unit_array();

    std::vector<float> get() const;
    std::string        get_display() const;

    void allocate(device::gpu = {}, size_t nthreads = 2048);
    void deallocate();
    void start(device::gpu, size_t nthreads = 2048);
    void start();
    void stop();

    struct device_data
    {
        TIMEMORY_DEVICE_FUNCTION int  get_index();
        TIMEMORY_DEVICE_FUNCTION void start();
        TIMEMORY_DEVICE_FUNCTION void stop();

        long long int m_buff = 0;
        unsigned int* m_data = nullptr;
    };

    auto get_device_data() const { return device_data{ 0, m_data }; }

private:
    static size_t& max_threads();

    bool          m_copy       = false;
    int           m_device_num = gpu::get_device();
    size_t        m_threads    = 0;
    unsigned int* m_data       = nullptr;
};
}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//
// saxpy calculation
//
TIMEMORY_GLOBAL_FUNCTION void
saxpy_inst(int64_t n, float a, float* x, float* y,
           comp::gpu_device_timer::device_data _timer)
{
    _timer.start();
    auto range = device::grid_strided_range<default_device, 0>(n);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        y[i] = a * x[i] + y[i];
    }
    _timer.stop();
}

//--------------------------------------------------------------------------------------//

void
run_saxpy(int nitr, int nstreams, int64_t block_size, int64_t N)
{
    using params_t = device::params<default_device>;
    using stream_t = default_device::stream_t;
    using tuple_t  = component_tuple<comp::wall_clock, comp::cpu_clock, comp::cpu_util,
                                    comp::gpu_event, comp::gpu_marker>;

    mpl::append_type_t<comp::gpu_device_timer, tuple_t> tot{ __FUNCTION__ };
    tot.start(device::gpu{}, block_size);

    float*                x         = device::cpu::alloc<float>(N);
    float*                y         = device::cpu::alloc<float>(N);
    float                 data_size = (3.0 * N * sizeof(float)) / tim::units::gigabyte;
    std::vector<stream_t> streams(std::max<int>(nstreams, 1));
    for(auto& itr : streams)
        gpu::stream_create(itr);
    stream_t stream = streams.at(0);
    params_t params(params_t::compute(N, block_size), block_size);

    auto sync_streams = [&streams]() {
        for(auto& itr : streams)
            gpu::stream_sync(itr);
    };

    std::cout << "\n"
              << __FUNCTION__ << " launching on " << default_device::name()
              << " with parameters: " << params << "\n"
              << std::endl;

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
    for(int i = 0; i < nitr; ++i)
    {
        params.stream = streams.at(i % streams.size());
        device::launch(params, saxpy_inst, N, 1.0, d_x, d_y,
                       tot.get<comp::gpu_device_timer>()->get_device_data());
    }
    sync_streams();

    TIMEMORY_GPU_RUNTIME_API_CALL(gpu::memcpy(y, d_y, N, gpu::device_to_host_v, stream));
    gpu::device_sync();
    tot.stop();
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

    auto ce = tot.get<comp::gpu_event>();
    auto rc = tot.get<comp::wall_clock>();

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
    std::cout << tot << std::endl;
    std::cout << std::endl;

    for(auto& itr : streams)
        gpu::stream_destroy(itr);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::timemory_init(argc, argv);
    tim::timemory_argparse(&argc, &argv);

    for(auto blocks : { 32, 64, 128, 256, 512 })
        run_saxpy(10, 4, blocks, 50 * (1 << 20));

    tim::timemory_finalize();
    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//

tim::component::gpu_device_timer::gpu_device_timer(const gpu_device_timer& rhs)
: base_type{ rhs }
, m_copy{ true }
, m_device_num{ rhs.m_device_num }
, m_threads{ rhs.m_threads }
, m_data{ rhs.m_data }
{}

tim::component::gpu_device_timer&
tim::component::gpu_device_timer::operator=(const gpu_device_timer& rhs)
{
    if(this != &rhs)
    {
        base_type::operator=(rhs);
        m_copy             = true;
        m_device_num       = rhs.m_device_num;
        m_threads          = rhs.m_threads;
        m_data             = rhs.m_data;
    }
    return *this;
}

tim::component::gpu_device_timer::~gpu_device_timer() { deallocate(); }

std::string
tim::component::gpu_device_timer::label()
{
    return "gpu_device_timer";
}
std::string
tim::component::gpu_device_timer::description()
{
    return "instruments clock64() within a GPU kernel";
}

std::vector<std::string>
tim::component::gpu_device_timer::label_array()
{
    std::vector<std::string> _labels{};
    for(size_t i = 0; i < max_threads(); ++i)
        _labels.emplace_back(std::string{ "thr" } + std::to_string(i));
    return _labels;
}

std::vector<int64_t>
tim::component::gpu_device_timer::unit_array()
{
    return std::vector<int64_t>(max_threads(), unit());
}

std::vector<std::string>
tim::component::gpu_device_timer::display_unit_array()
{
    return std::vector<std::string>(max_threads(), display_unit());
}

std::vector<float>
tim::component::gpu_device_timer::get() const
{
    std::vector<float> _value{};
    auto               _factor     = static_cast<float>(units::msec) * unit();
    auto               _clock_rate = gpu::get_device_clock_rate(m_device_num);
    for(const auto& itr : get_value())
        _value.emplace_back(itr / static_cast<float>(_clock_rate) / _factor);
    return _value;
}

std::string
tim::component::gpu_device_timer::get_display() const
{
    auto               _data = get();
    std::ostringstream _oss{};
    _oss.setf(get_format_flags());
    for(size_t i = 0; i < _data.size(); ++i)
    {
        if(_data.at(i) > 0.0)
            _oss << ", " << std::setprecision(get_precision()) << std::setw(get_width())
                 << _data.at(i) << " thr" << i;
    }
    auto _str = _oss.str();
    if(_str.length() > 2)
    {
        _oss << " " << get_display_unit();
        return _str.substr(2);
    }
    return _str;
}

void
tim::component::gpu_device_timer::allocate(device::gpu, size_t nthreads)
{
    if(nthreads > m_threads)
    {
        deallocate();
        m_copy    = false;
        m_threads = nthreads;
        m_data    = gpu::malloc<unsigned int>(m_threads);
        TIMEMORY_GPU_RUNTIME_API_CALL(gpu::memset(m_data, 0, m_threads));
        max_threads() = std::max<size_t>(max_threads(), nthreads);
        gpu::check(gpu::get_last_error());
    }
}

void
tim::component::gpu_device_timer::deallocate()
{
    if(m_copy)
        return;
    m_threads = 0;
    if(m_data)
        gpu::free(m_data);
}

void
tim::component::gpu_device_timer::start(device::gpu, size_t nthreads)
{
    set_is_invalid(false);
    m_device_num = gpu::get_device();
    allocate(device::gpu{}, nthreads);
}

void
tim::component::gpu_device_timer::start()
{
    set_is_invalid(m_data == nullptr);
}

void
tim::component::gpu_device_timer::stop()
{
    if(m_data)
    {
        gpu::stream_sync(gpu::default_stream_v);
        std::vector<unsigned int> _host(m_threads);
        TIMEMORY_GPU_RUNTIME_API_CALL(
            gpu::memcpy(_host.data(), m_data, m_threads, gpu::device_to_host_v));
        auto _value = get_value();
        _value.resize(m_threads, 0);
        for(size_t i = 0; i < m_threads; ++i)
            _value[i] += _host[i];
        set_value(std::move(_value));
    }
}

TIMEMORY_DEVICE_FUNCTION
int
tim::component::gpu_device_timer::device_data::get_index()
{
    return (blockDim.x * blockDim.y * blockIdx.z) + (blockDim.x * blockIdx.y) +
           threadIdx.x;
}

TIMEMORY_DEVICE_FUNCTION
void
tim::component::gpu_device_timer::device_data::start()
{
    if(m_data)
        m_buff = clock64();
}

TIMEMORY_DEVICE_FUNCTION
void
tim::component::gpu_device_timer::device_data::stop()
{
    if(m_data)
        atomicAdd(&(m_data[get_index()]), static_cast<int>(clock64() - m_buff));
}

size_t&
tim::component::gpu_device_timer::max_threads()
{
    static size_t _value = 0;
    return _value;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ long long int
atomicAdd(long long int* address, long long int val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int  old            = *address_as_ull;
    unsigned long long int  assumed;
    do
    {
        assumed = old;
        old     = atomicCAS(address_as_ull, assumed, val + assumed);
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while(assumed != old);
    return old;
}
#endif
