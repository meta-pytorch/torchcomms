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

#include "gpu_common.hpp"
#include "timemory/variadic.hpp"

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

    void allocate(device::gpu, device::params<device::gpu>);
    void allocate(device::gpu = {}, size_t nthreads = 2048);
    void deallocate();
    void set_prefix(const char* _prefix) { m_prefix = _prefix; }
    void start(device::gpu, device::params<device::gpu>);
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

    using device_handle = device::handle<device_data>;

    device_data* get_device_data() const { return m_device_data; }

    TIMEMORY_STATIC_ACCESSOR(bool, add_secondary,
                             tim::get_env("TIMEMORY_GPU_ADD_SECONDARY",
                                          settings::add_secondary()))

private:
    static size_t& max_threads();

    bool          m_copy        = false;
    int           m_device_num  = gpu::get_device();
    size_t        m_count       = 0;
    size_t        m_threads     = 0;
    unsigned int* m_incr        = nullptr;
    CLOCK_DTYPE*  m_data        = nullptr;
    const char*   m_prefix      = nullptr;
    device_data*  m_device_data = nullptr;
    tracker_type  m_tracker     = {};
};

// ************** IMPORTANT ************** //
//   Anything launching kernel or calling  //
//   function launching kernel MUST be in  //
//   the same translation unit as the      //
//   kernel collecting the data. Thus, the //
//   inlined functions below               //
// ************** IMPORTANT ************** //

inline gpu_device_timer::~gpu_device_timer() { deallocate(); }

inline void
gpu_device_timer::allocate(device::gpu, device::params<device::gpu> _params)
{
    allocate(device::gpu{}, _params.block);
}

inline void
gpu_device_timer::allocate(device::gpu, size_t nthreads)
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
        m_device_data = gpu::malloc<device_data>(1);
        auto _data    = device_data{ 0, m_incr, m_data };
        TIMEMORY_GPU_RUNTIME_API_CALL(
            gpu::memcpy(m_device_data, &_data, 1, gpu::host_to_device_v));
        device::set_handle<<<1, 1>>>(m_device_data);
    }
}

inline void
gpu_device_timer::deallocate()
{
    // only the instance that allocated should deallocate
    if(!m_copy)
    {
        gpu::device_sync();
        m_count   = 0;
        m_threads = 0;
        if(m_incr)
            gpu::free(m_incr);
        if(m_data)
            gpu::free(m_data);
        if(m_device_data)
            gpu::free(m_device_data);
        m_incr        = nullptr;
        m_data        = nullptr;
        m_device_data = nullptr;
        device::set_handle<<<1, 1>>>(m_device_data);
    }
}

inline void
gpu_device_timer::start(device::gpu, device::params<device::gpu> _params)
{
    start(device::gpu{}, _params.block);
}

inline void
gpu_device_timer::start(device::gpu, size_t nthreads)
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

inline void
gpu_device_timer::start()
{
    set_is_invalid(m_data == nullptr);
    if(m_incr != nullptr && m_data != nullptr && m_prefix != nullptr)
    {
        m_tracker.rekey(m_prefix);
        m_tracker.start();
    }
}

TIMEMORY_DEVICE_FUNCTION
inline int
gpu_device_timer::device_data::get_index()
{
    return threadIdx.x + (blockDim.y * threadIdx.y) +
           (blockDim.x * blockDim.y * threadIdx.z);
}

TIMEMORY_DEVICE_FUNCTION
inline void
gpu_device_timer::device_data::start()
{
    m_buff = clock64();
}

TIMEMORY_DEVICE_FUNCTION
inline void
gpu_device_timer::device_data::stop()
{
    __syncthreads();
    auto _time = clock64();
    if(_time > m_buff)
    {
        atomicAdd(&m_incr[get_index()], 1);
        atomicAdd(&m_data[get_index()], _time - m_buff);
    }
}

}  // namespace component
}  // namespace tim

#include "timemory/operations.hpp"
#include "timemory/storage.hpp"

TIMEMORY_DECLARE_EXTERN_COMPONENT(gpu_device_timer, false, void)
