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

struct gpu_data_tag
{};
using gpu_device_timer_data = comp::data_tracker<double, gpu_data_tag>;

TIMEMORY_STATISTICS_TYPE(gpu_device_timer_data, double)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, gpu_device_timer_data, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, gpu_device_timer_data, true_type)

#define CLOCK_DTYPE unsigned long long
namespace tim
{
namespace component
{
struct gpu_device_timer : base<gpu_device_timer, void>
{
    using value_type   = void;
    using base_type    = base<gpu_device_timer, value_type>;
    using tracker_type = tim::component_bundle<TIMEMORY_API, gpu_device_timer_data>;
    using stream_vec_t = std::vector<gpu::stream_t>;

    gpu_device_timer() = default;
    ~gpu_device_timer();
    gpu_device_timer(gpu_device_timer&&) = default;
    gpu_device_timer& operator=(gpu_device_timer&&) = default;
    gpu_device_timer(const gpu_device_timer& rhs);
    gpu_device_timer& operator=(const gpu_device_timer& rhs);

    static void        preinit();
    static std::string label();
    static std::string description();

    void allocate(device::gpu = {}, size_t nthreads = 2048);
    void deallocate();
    void set_prefix(const char* _prefix) { m_prefix = _prefix; }
    void start(device::gpu, size_t nthreads = 2048);
    void start();
    void stop();
    void mark();
    void mark_begin() { mark(); }
    void mark_begin(gpu::stream_t) { mark(); }

    struct device_data
    {
        TIMEMORY_DEVICE_FUNCTION int  get_index();
        TIMEMORY_DEVICE_FUNCTION void start();
        TIMEMORY_DEVICE_FUNCTION void stop();

        // clock64() return a long long int but older GPU archs only have
        // atomics for 32-bit values (and sometimes unsigned long long) but
        // the difference in the clocks should be much, much less than
        // numeric_limits<int>::max() so m_data is an array of ints.
        // m_incr is the counter and shouldn't exceed 1 << 32.
        long long int m_buff = 0;
        unsigned int* m_incr = nullptr;
        CLOCK_DTYPE*  m_data = nullptr;
    };

    auto get_device_data() const { return device_data{ 0, m_incr, m_data }; }

private:
    static size_t& max_threads();

    bool          m_copy       = false;
    int           m_device_num = gpu::get_device();
    size_t        m_count      = 0;
    size_t        m_threads    = 0;
    unsigned int* m_incr       = nullptr;
    CLOCK_DTYPE*  m_data       = nullptr;
    const char*   m_prefix     = nullptr;
    tracker_type  m_tracker    = {};
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
run_saxpy(int nitr, int nstreams, int64_t block_size, int64_t N);

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    using parser_t = tim::argparse::argument_parser;

    int           nitr    = 10;
    int           nstream = 2;
    int           npow    = 20;
    std::set<int> nblocks = { 32, 64, 128, 256, 512 };

    parser_t _parser{ "ex_kernel_instrument_v2" };
    _parser.add_argument({ "-n", "--num-iter" }, "Number of iterations")
        .dtype("int")
        .count(1)
        .action([&](parser_t& p) { nitr = p.get<int>("num-iter"); });
    _parser.add_argument({ "-s", "--num-streams" }, "Number of GPU streams")
        .dtype("int")
        .count(1)
        .action([&](parser_t& p) { nstream = p.get<int>("num-streams"); });
    _parser.add_argument({ "-b", "--num-blocks" }, "Thread-block sizes")
        .dtype("int")
        .action([&](parser_t& p) { nblocks = p.get<std::set<int>>("num-blocks"); });
    _parser
        .add_argument({ "-p", "--num-pow" },
                      "Data size (powers of 2, e.g. '20' in 1 << 20)")
        .dtype("int")
        .action([&](parser_t& p) { npow = p.get<int>("num-pow"); });

    tim::timemory_init(argc, argv);
    tim::timemory_argparse(&argc, &argv, &_parser);

    for(auto bitr : nblocks)
        run_saxpy(nitr, nstream, bitr, 50 * (1 << npow));

    tim::timemory_finalize();
    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//

tim::component::gpu_device_timer::gpu_device_timer(const gpu_device_timer& rhs)
: base_type{ rhs }
, m_copy{ true }
, m_device_num{ rhs.m_device_num }
, m_count{ rhs.m_count }
, m_threads{ rhs.m_threads }
, m_incr{ rhs.m_incr }
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
        m_count            = rhs.m_count;
        m_threads          = rhs.m_threads;
        m_incr             = rhs.m_incr;
        m_data             = rhs.m_data;
    }
    return *this;
}

tim::component::gpu_device_timer::~gpu_device_timer() { deallocate(); }

void
tim::component::gpu_device_timer::preinit()
{
    gpu_device_timer_data::label()       = label();
    gpu_device_timer_data::description() = description();
}

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

void
tim::component::gpu_device_timer::allocate(device::gpu, size_t nthreads)
{
    if(m_copy || nthreads > m_threads)
    {
        deallocate();
        m_copy    = false;
        m_count   = 0;
        m_threads = nthreads;
        m_incr    = gpu::malloc<unsigned int>(m_threads);
        m_data    = gpu::malloc<CLOCK_DTYPE>(m_threads);
        TIMEMORY_GPU_RUNTIME_API_CALL(gpu::memset(m_incr, 0, m_threads));
        TIMEMORY_GPU_RUNTIME_API_CALL(gpu::memset(m_data, 0, m_threads));
        max_threads() = std::max<size_t>(max_threads(), nthreads);
        gpu::check(gpu::get_last_error());
    }
}

void
tim::component::gpu_device_timer::deallocate()
{
    // only the instance that allocated should deallocate
    if(!m_copy)
    {
        m_count   = 0;
        m_threads = 0;
        if(m_incr)
            gpu::free(m_incr);
        if(m_data)
            gpu::free(m_data);
    }
}

void
tim::component::gpu_device_timer::start(device::gpu, size_t nthreads)
{
    set_is_invalid(false);
    m_device_num = gpu::get_device();
    allocate(device::gpu{}, nthreads);
    if(m_prefix)
    {
        m_tracker.rekey(m_prefix);
        m_tracker.start();
    }
}

void
tim::component::gpu_device_timer::start()
{
    set_is_invalid(m_data == nullptr);
    if(m_incr != nullptr && m_data != nullptr && m_prefix != nullptr)
    {
        m_tracker.rekey(m_prefix);
        m_tracker.start();
    }
}

void
tim::component::gpu_device_timer::stop()
{
    if(m_incr && m_data)
    {
        // int _clock_rate = 1;
        // cudaDeviceGetAttribute(&_clock_rate, cudaDevAttrClockRate, m_device_num);
        auto _clock_rate = gpu::get_device_clock_rate(m_device_num);
        auto _scope      = tim::scope::config{} + tim::scope::tree{};

        gpu::stream_sync(gpu::default_stream_v);
        std::vector<CLOCK_DTYPE>  _host(m_threads);
        std::vector<unsigned int> _incr(m_threads);

        TIMEMORY_GPU_RUNTIME_API_CALL(
            gpu::memcpy(_host.data(), m_data, m_threads, gpu::device_to_host_v));
        TIMEMORY_GPU_RUNTIME_API_CALL(
            gpu::memcpy(_incr.data(), m_incr, m_threads, gpu::device_to_host_v));

        std::vector<double> _values(m_threads, 0.0f);

        for(size_t i = 0; i < m_threads; ++i)
            _values.at(i) =
                (_host.at(i) / static_cast<double>(_clock_rate) / units::msec);

        tracker_type _child{ "", _scope };
        for(size_t i = 0; i < m_threads; ++i)
        {
            if(_incr.at(i) == 0)
                continue;

            double _value = _values.at(i);

            m_tracker.store(
                [](double lhs, double rhs) { return std::max<double>(lhs, rhs); },
                _value);

            if(settings::add_secondary())
            {
                // secondary data
                _child.rekey(TIMEMORY_JOIN("_", "thread", i));
                _child.start();
                _child.store(std::plus<double>{}, _value);
                auto* _data = _child.get<gpu_device_timer_data>();
                if(_data)
                    _data->set_laps(_data->get_laps() + _incr.at(i) - 1);
                _child.stop();
                _child.reset();
            }
        }

        auto* _data = m_tracker.get<gpu_device_timer_data>();
        if(_data)
            _data->set_laps(_data->get_laps() + m_count - 1);

        m_tracker.stop();
    }
}

void
tim::component::gpu_device_timer::mark()
{
    ++m_count;
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
    {
        __syncthreads();
        m_buff = clock64();
    }
}

TIMEMORY_DEVICE_FUNCTION
void
tim::component::gpu_device_timer::device_data::stop()
{
    if(m_data)
    {
        auto _time = clock64();
        __syncthreads();
        atomicAdd(&m_incr[get_index()], 1);
        atomicAdd(&m_data[get_index()], static_cast<CLOCK_DTYPE>(_time - m_buff));
    }
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

void
run_saxpy(int nitr, int nstreams, int64_t block_size, int64_t N)
{
    using params_t = device::params<default_device>;
    using stream_t = default_device::stream_t;
    using tuple_t =
        component_tuple<comp::wall_clock, comp::cpu_clock, comp::cpu_util,
                        comp::gpu_event, comp::gpu_marker, comp::gpu_device_timer>;

    tuple_t tot{ __FUNCTION__ };
    tot.store(comp::gpu_event::explicit_streams_only{}, true);
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
    for(auto& itr : streams)
        tot.mark_begin(mpl::piecewise_select<comp::gpu_event>{}, itr);
    for(int i = 0; i < nitr; ++i)
    {
        tot.mark(mpl::piecewise_select<comp::gpu_device_timer>{});
        params.stream = streams.at(i % streams.size());
        device::launch(params, saxpy_inst, N, 1.0, d_x, d_y,
                       tot.get<comp::gpu_device_timer>()->get_device_data());
    }
    for(auto& itr : streams)
        tot.mark_end(mpl::piecewise_select<comp::gpu_event>{}, itr);
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

TIMEMORY_INITIALIZE_STORAGE(comp::gpu_device_timer, gpu_device_timer_data)
