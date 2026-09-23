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
#include "timemory/operations.hpp"
#include "timemory/variadic.hpp"

namespace tim
{
namespace component
{
struct gpu_op_tracker : base<gpu_op_tracker, void>
{
    using value_type   = void;
    using base_type    = base<gpu_op_tracker, value_type>;
    using data_type    = gpu_device_op_data;
    using tracker_type = tim::lightweight_tuple<data_type>;
    using stream_vec_t = std::vector<gpu::stream_t>;

    gpu_op_tracker() = default;
    ~gpu_op_tracker();
    gpu_op_tracker(gpu_op_tracker&&) = default;
    gpu_op_tracker& operator=(gpu_op_tracker&&) = default;
    gpu_op_tracker(const gpu_op_tracker& rhs);
    gpu_op_tracker& operator=(const gpu_op_tracker& rhs);

    static void        preinit();
    static std::string label();
    static std::string description();

    static void   set_max_blocks(size_t _v) { max_blocks() = _v; }
    static size_t get_max_blocks() { return max_blocks(); }

    void set_prefix(size_t _prefix) { m_prefix = _prefix; }

    void allocate(device::gpu, device::params<device::gpu>,
                  gpu::stream_t _stream = gpu::default_stream_v);
    void allocate(device::gpu = {}, size_t nblocks = max_blocks(),
                  gpu::stream_t _stream = gpu::default_stream_v);

    void deallocate();

    void start(device::gpu, device::params<device::gpu>,
               gpu::stream_t _stream = gpu::default_stream_v);
    void start(device::gpu, size_t nblocks = max_blocks(),
               gpu::stream_t _stream = gpu::default_stream_v);

    void start();
    void stop();
    void mark() { ++m_count; }

    template <typename... Args>
    void push(Args&&... args);

    template <typename... Args>
    void pop(Args&&... args);

    struct device_data
    {
        TIMEMORY_DEVICE_FUNCTION uint32_t get_index();
        TIMEMORY_DEVICE_FUNCTION void     start() {}
        TIMEMORY_DEVICE_FUNCTION void     stop() {}
        TIMEMORY_DEVICE_FUNCTION void     operator()(unsigned long long _v);

        uint32_t            size = 0;
        unsigned long long* data = nullptr;
    };

    using device_handle = device::handle<device_data>;

    device_data* get_device_data() const { return m_device_data; }

    TIMEMORY_STATIC_ACCESSOR(bool, add_secondary,
                             tim::get_env("TIMEMORY_GPU_ADD_SECONDARY",
                                          settings::add_secondary()))

private:
    static size_t& max_blocks();

    bool                m_copy        = false;
    int                 m_device_num  = gpu::get_device();
    size_t              m_count       = 0;
    size_t              m_threads     = 0;
    size_t              m_prefix      = 0;
    gpu::stream_t       m_stream      = gpu::default_stream_v;
    unsigned long long* m_data        = nullptr;
    device_data*        m_device_data = nullptr;
    tracker_type        m_tracker     = {};
};

// ************** IMPORTANT ************** //
//   Anything launching kernel or calling  //
//   function launching kernel MUST be in  //
//   the same translation unit as the      //
//   kernel collecting the data. Thus, the //
//   inlined functions below               //
// ************** IMPORTANT ************** //

inline gpu_op_tracker::~gpu_op_tracker() { deallocate(); }

template <typename... Args>
void
gpu_op_tracker::push(Args&&... args)
{
    tim::operation::push_node<data_type>{}(*m_tracker.get<data_type>(),
                                           std::forward<Args>(args)...);
}

template <typename... Args>
void
gpu_op_tracker::pop(Args&&... args)
{
    tim::operation::pop_node<data_type>{}(*m_tracker.get<data_type>(),
                                          std::forward<Args>(args)...);
}

inline void
gpu_op_tracker::allocate(device::gpu, device::params<device::gpu> _params,
                         gpu::stream_t _stream)
{
    allocate(device::gpu{}, _params.block, _stream);
}

inline void
gpu_op_tracker::allocate(device::gpu, size_t nblocks, gpu::stream_t _stream)
{
    if(!m_device_data || m_copy || nblocks > m_threads)
    {
        deallocate();
        m_copy        = false;
        m_threads     = nblocks;
        m_stream      = _stream;
        max_blocks()  = std::max<size_t>(max_blocks(), nblocks);
        m_data        = gpu::malloc<unsigned long long>(m_threads);
        m_device_data = gpu::malloc<device_data>(1);
        TIMEMORY_GPU_RUNTIME_API_CALL(gpu::memset(m_data, 0, m_threads, m_stream));
        auto _data = device_data{ static_cast<uint32_t>(m_threads), m_data };
        TIMEMORY_GPU_RUNTIME_API_CALL(
            gpu::memcpy(m_device_data, &_data, 1, gpu::host_to_device_v, m_stream));
        device::set_handle<<<1, 1, 0, m_stream>>>(m_device_data);
        gpu::stream_sync(m_stream);
    }
}

inline void
gpu_op_tracker::deallocate()
{
    // only the instance that allocated should deallocate
    if(!m_copy && m_device_data)
    {
        m_threads = 0;
        gpu::free(m_data);
        gpu::free(m_device_data);
        m_data        = nullptr;
        m_device_data = nullptr;
        device::set_handle<<<1, 1>>>(m_device_data);
    }
}

inline void
gpu_op_tracker::start(device::gpu, device::params<device::gpu> _params,
                      gpu::stream_t _stream)
{
    start(device::gpu{}, _params.block, _stream);
}

inline void
gpu_op_tracker::start(device::gpu, size_t nblocks, gpu::stream_t _stream)
{
    set_is_invalid(false);
    m_device_num = gpu::get_device();
    allocate(device::gpu{}, nblocks, _stream);
    if(m_prefix != 0)
    {
        m_tracker.rekey(m_prefix);
        m_tracker.start();
    }
}

inline void
gpu_op_tracker::start()
{
    set_is_invalid(m_data == nullptr);
    if(!get_is_invalid())
    {
        m_tracker.rekey(m_prefix);
        m_tracker.start();
    }
}

TIMEMORY_DEVICE_FUNCTION
inline uint32_t
gpu_op_tracker::device_data::get_index()
{
    return threadIdx.x + (blockDim.y * threadIdx.y) +
           (blockDim.x * blockDim.y * threadIdx.z);
}

TIMEMORY_DEVICE_FUNCTION
inline void
gpu_op_tracker::device_data::operator()(unsigned long long _v)
{
    atomicAdd(&data[get_index()], _v);
}
}  // namespace component
}  // namespace tim

#include "timemory/storage.hpp"

TIMEMORY_DECLARE_EXTERN_COMPONENT(gpu_op_tracker, false, void)
