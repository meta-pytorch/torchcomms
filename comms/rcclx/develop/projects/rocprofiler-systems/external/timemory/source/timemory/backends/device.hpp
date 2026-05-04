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

/** \file backends/device.hpp
 * \headerfile backends/device.hpp "timemory/backends/device.hpp"
 * Defines device-generic structures and routines
 *
 */

#pragma once

#include "timemory/backends/gpu.hpp"
#include "timemory/components/cuda/backends.hpp"
#include "timemory/components/hip/backends.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/macros/compiler.hpp"

#include <cstdint>
#include <type_traits>

namespace tim
{
namespace device
{
//--------------------------------------------------------------------------------------//

template <typename... ArgsT>
inline void
consume_parameters(ArgsT&&...)
{}

//--------------------------------------------------------------------------------------//
//
template <bool B, class T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

//--------------------------------------------------------------------------------------//

struct cpu
{
    using stream_t = ::tim::gpu::stream_t;
    using fp16_t   = float;

    static constexpr auto default_stream = ::tim::gpu::default_stream_v;

    template <typename Tp>
    static Tp* alloc(std::size_t nsize)
    {
        return new Tp[nsize];
    }

    template <typename Tp>
    static void free(Tp* ptr)
    {
        delete[] ptr;
    }

    static std::string name() { return "cpu"; }
};

//--------------------------------------------------------------------------------------//

struct gpu
{
#if(defined(TIMEMORY_USE_CUDA) || defined(TIMEMORY_USE_HIP))
    using stream_t = ::tim::gpu::stream_t;
    using fp16_t   = ::tim::gpu::fp16_t;

    static constexpr auto default_stream = ::tim::gpu::default_stream_v;

    template <typename Tp>
    static Tp* alloc(std::size_t nsize)
    {
        return ::tim::gpu::malloc<Tp>(nsize);
    }

    template <typename Tp>
    static void free(Tp* ptr)
    {
        ::tim::gpu::free(ptr);
    }

    static std::string name() { return "gpu"; }
#else
    using stream_t = int;
    using fp16_t   = float;

    template <typename Tp>
    static Tp* alloc(std::size_t nsize)
    {
        return new Tp[nsize];
    }

    template <typename Tp>
    static void free(Tp* ptr)
    {
        delete[] ptr;
    }

    static std::string name() { return "gpu_on_cpu"; }
#endif
};

template <typename Tp>
TIMEMORY_GLOBAL_FUNCTION void
set_handle(Tp* _v);

//--------------------------------------------------------------------------------------//
/// \struct tim::device::handle
/// \tparam Tp device object type with `start()` and `stop()` member functions
///
/// \brief This handle creates an object on the stack which is valid for ONE set of calls
/// to `start(...)` and `stop()` on an instance of `Tp`.
template <typename Tp>
struct handle
{
    /// stores instance of \param _targ and calls starts with given arguments
    template <typename Up      = Tp, typename... Args,
              enable_if_t<trait::is_available<Up>::value &&
                              trait::is_available<handle<Up>>::value,
                          int> = 0>
    TIMEMORY_DEVICE_FUNCTION handle(Tp* _targ, Args... _args);

    template <typename Up       = Tp, typename... Args,
              enable_if_t<!trait::is_available<Up>::value ||
                              !trait::is_available<handle<Up>>::value,
                          long> = 0>
    TIMEMORY_DEVICE_FUNCTION handle(Tp* _targ, Args... _args);

    /// calls the stop member function on instance stored during construction
    TIMEMORY_DEVICE_FUNCTION ~handle();

    /// calls the function call operator with given arguments on the instance
    /// stored during construction
    template <typename... Args>
    TIMEMORY_DEVICE_FUNCTION void operator()(Args... _args) const;

    /// provides an explicit way to call stop before destruction. If this function
    /// is explicitly called, the destructor will not call stop because it will
    /// reset the pointer to null.
    TIMEMORY_DEVICE_FUNCTION void stop();

    /// constructs a handle with the current instance and the arguments
    template <typename... Args>
    static TIMEMORY_DEVICE_FUNCTION handle<Tp> get(Args... _args);

private:
    template <typename Up>
    friend TIMEMORY_GLOBAL_FUNCTION void set_handle(Up*);
    static TIMEMORY_DEVICE_FUNCTION Tp*& get_instance();

    Tp* m_targ = nullptr;
};

template <typename Tp>
TIMEMORY_GLOBAL_FUNCTION void
set_handle(Tp* _v)
{
    handle<Tp>::get_instance() = _v;
}

template <typename Tp>
template <
    typename Up, typename... Args,
    enable_if_t<trait::is_available<Up>::value && trait::is_available<handle<Up>>::value,
                int>>
TIMEMORY_DEVICE_FUNCTION
handle<Tp>::handle(Tp* _targ, Args... _args)
: m_targ{ _targ }
{
    if(m_targ)
        m_targ->start(_args...);
}

template <typename Tp>
template <
    typename Up, typename... Args,
    enable_if_t<
        !trait::is_available<Up>::value || !trait::is_available<handle<Up>>::value, long>>
TIMEMORY_DEVICE_FUNCTION
handle<Tp>::handle(Tp*, Args...)
{}

template <typename Tp>
TIMEMORY_DEVICE_FUNCTION handle<Tp>::~handle()
{
    if(m_targ)
        m_targ->stop();
}

template <typename Tp>
template <typename... Args>
TIMEMORY_DEVICE_FUNCTION void
handle<Tp>::operator()(Args... _args) const
{
    // printf("[device][operator()] instance is %p\n", (void*) m_targ);
    if(m_targ)
        (*m_targ)(std::forward<Args>(_args)...);
}

template <typename Tp>
TIMEMORY_DEVICE_FUNCTION void
handle<Tp>::stop()
{
    if(m_targ)
        m_targ = (m_targ->stop(), nullptr);
}

template <typename Tp>
template <typename... Args>
TIMEMORY_DEVICE_FUNCTION handle<Tp>
                         handle<Tp>::get(Args... _args)
{
    return handle<Tp>{ get_instance(), _args... };
}

template <typename Tp>
TIMEMORY_DEVICE_FUNCTION Tp*&
                         handle<Tp>::get_instance()
{
    static TIMEMORY_DEVICE_FUNCTION Tp* _v = nullptr;
    return _v;
}

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_GPUCC) && !defined(TIMEMORY_OPENMP_TARGET)
using default_device = gpu;
#else
using default_device = cpu;
#endif

//--------------------------------------------------------------------------------------//

template <typename Up>
struct is_gpu
{
    static constexpr bool value = std::is_same<Up, device::gpu>::value;
};

//--------------------------------------------------------------------------------------//

template <typename Up>
struct is_cpu
{
    static constexpr bool value = std::is_same<Up, device::cpu>::value;
};

//--------------------------------------------------------------------------------------//

template <typename T, typename U = int, bool B = true>
using enable_if_gpu_t = enable_if_t<is_gpu<T>::value == B, U>;

//--------------------------------------------------------------------------------------//

template <typename T, typename U = int, bool B = true>
using enable_if_cpu_t = enable_if_t<is_cpu<T>::value == B, U>;

//--------------------------------------------------------------------------------------//

namespace impl
{
template <typename Tp, typename Intp, bool Valv = true>
using enable_if_gpu_int_t =
    enable_if_t<is_gpu<Tp>::value == Valv && std::is_integral<Intp>::value>;

template <typename Tp, typename Intp, bool Valv = true>
using enable_if_cpu_int_t =
    enable_if_t<is_cpu<Tp>::value == Valv && std::is_integral<Intp>::value>;

//--------------------------------------------------------------------------------------//
//
template <typename Intp>
struct range
{
    using this_type = range<Intp>;

    TIMEMORY_HOST_DEVICE_FUNCTION range(Intp _begin, Intp _end, Intp _stride)
    : m_begin(_begin)
    , m_end(_end)
    , m_stride(_stride)
    {}

    TIMEMORY_HOST_DEVICE_FUNCTION Intp& begin() { return m_begin; }
    TIMEMORY_HOST_DEVICE_FUNCTION Intp  begin() const { return m_begin; }
    TIMEMORY_HOST_DEVICE_FUNCTION Intp& end() { return m_end; }
    TIMEMORY_HOST_DEVICE_FUNCTION Intp  end() const { return m_end; }
    TIMEMORY_HOST_DEVICE_FUNCTION Intp& stride() { return m_stride; }
    TIMEMORY_HOST_DEVICE_FUNCTION Intp  stride() const { return m_stride; }

    TIMEMORY_HOST_FUNCTION const char* c_str() const
    {
        using llu                             = long long unsigned;
        constexpr size_t         desc_max_len = 512;
        static thread_local char desc[desc_max_len];
        desc[desc_max_len - 1] = '\0';
        snprintf(desc, desc_max_len, "(%llu,%llu,%llu)", (llu) m_begin, (llu) m_end,
                 (llu) m_stride);
        return desc;
    }

private:
    Intp m_begin;
    Intp m_end;
    Intp m_stride;
};
}  // namespace impl

//======================================================================================//
//
//  These provide the kernel launch parameters for the GPU
//
//======================================================================================//

template <typename DeviceT>
struct params
{
    using stream_t = typename DeviceT::stream_t;
    params()       = default;

    static_assert(
        std::is_same<stream_t, decay_t<decltype(DeviceT::default_stream)>>::value,
        "Error! default stream type is not same as stream type");

    explicit params(uint32_t _grid, uint32_t _block, uint32_t _shmem = 0,
                    stream_t _stream = DeviceT::default_stream)
    : block(_block)
    , grid(_grid)
    , shmem(_shmem)
    , stream(_stream)
    , dynamic((_grid > 0) ? false : true)
    {}

    int compute(int size)
    {
        if(grid == 0 || dynamic)
            grid = ((size + block - 1) / block);
        return grid;
    }

    static int compute(int size, int block_size)
    {
        return ((size + block_size - 1) / block_size);
    }

    friend std::ostream& operator<<(std::ostream& os, const params& obj)
    {
        std::stringstream ss;
        ss << "<<<" << obj.grid << ", " << obj.block << ", " << obj.shmem << ", "
           << ((obj.stream == 0) ? "default_stream" : "stream") << ">>>";
        os << ss.str();
        return os;
    }

    // these should not be linted
    uint32_t block   = 32;                       // NOLINT
    uint32_t grid    = 0;                        // NOLINT: 0 == compute
    uint32_t shmem   = 0;                        // NOLINT
    stream_t stream  = DeviceT::default_stream;  // NOLINT
    bool     dynamic = true;  // NOLINT: allow the grid size to be dynamically computed
};

//======================================================================================//
//
//
//              CPU LAUNCH
//
//
//======================================================================================//

// execute a function with provided device parameters
//
template <typename DeviceT, typename FuncT, typename... ArgsT,
          enable_if_cpu_t<DeviceT> = 0>
void
launch(params<DeviceT>&, FuncT&& _func, ArgsT&&... _args)
{
    std::forward<FuncT>(_func)(std::forward<ArgsT>(_args)...);
}

//--------------------------------------------------------------------------------------//
// overload that passes size
//
template <typename Intp, typename DeviceT, typename FuncT, typename... ArgsT,
          impl::enable_if_cpu_int_t<DeviceT, Intp> = 0>
void
launch(const Intp&, params<DeviceT>&, FuncT&& _func, ArgsT&&... _args)
{
    std::forward<FuncT>(_func)(std::forward<ArgsT>(_args)...);
}

//--------------------------------------------------------------------------------------//
// overload that passes size
//
template <typename Intp, typename DeviceT, typename FuncT, typename... ArgsT,
          typename StreamT                         = typename DeviceT::stream_t,
          impl::enable_if_cpu_int_t<DeviceT, Intp> = 0>
void
launch(const Intp&, StreamT, params<DeviceT>&, FuncT&& _func, ArgsT&&... _args)
{
    std::forward<FuncT>(_func)(std::forward<ArgsT>(_args)...);
}

//======================================================================================//
//
//
//              GPU LAUNCH
//
//
//======================================================================================//

// execute a function with provided device parameters
//
template <typename DeviceT, typename FuncT, typename... ArgsT,
          enable_if_gpu_t<DeviceT> = 0>
void
launch(params<DeviceT>& _p, FuncT&& _func, ArgsT&&... _args)
{
#if defined(TIMEMORY_GPUCC) && !defined(TIMEMORY_OPENMP_TARGET)
    if(_p.grid == 0)
        _p.grid = 1;
    std::forward<FuncT>(_func)<<<_p.grid, _p.block, _p.shmem, _p.stream>>>(
        std::forward<ArgsT>(_args)...);
    TIMEMORY_GPU_RUNTIME_CHECK_ERROR(::tim::gpu::get_last_error());
    consume_parameters(_args...);
#else
    // static_assert(false, "Checking");
    consume_parameters(_p, _func, _args...);
    throw std::runtime_error(
        "Launch specified on GPU but not compiled with GPU support!");
#endif
}

//--------------------------------------------------------------------------------------//
// overload that passes size
//
template <typename Intp, typename DeviceT, typename FuncT, typename... ArgsT,
          impl::enable_if_gpu_int_t<DeviceT, Intp> = 0>
void
launch(const Intp& _nsize, params<DeviceT>& _p, FuncT&& _func, ArgsT&&... _args)
{
#if defined(TIMEMORY_GPUCC) && !defined(TIMEMORY_OPENMP_TARGET)
    if(_p.grid == 0 && _nsize > 0)
        _p.grid = _p.compute(_nsize);
    else if(_p.grid == 0)
        _p.grid = 1;
    std::forward<FuncT>(_func)<<<_p.grid, _p.block, _p.shmem, _p.stream>>>(
        std::forward<ArgsT>(_args)...);
    TIMEMORY_GPU_RUNTIME_CHECK_ERROR(::tim::gpu::get_last_error());
    consume_parameters(_args...);
#else
    // static_assert(false, "Checking");
    consume_parameters(_p, _func, _args..., _nsize);
    throw std::runtime_error(
        "Launch specified on GPU but not compiled with GPU support!");
#endif
}

//--------------------------------------------------------------------------------------//
// overload that passes size and stream
//
template <typename Intp, typename DeviceT, typename FuncT, typename... ArgsT,
          typename StreamT                               = typename DeviceT::stream_t,
          impl::enable_if_gpu_int_t<DeviceT, Intp, true> = 0>
void
launch(const Intp& _nsize, StreamT _stream, params<DeviceT>& _p, FuncT&& _func,
       ArgsT&&... _args)
{
#if defined(TIMEMORY_GPUCC) && !defined(TIMEMORY_OPENMP_TARGET)
    if(_p.grid == 0 && _nsize > 0)
        _p.grid = _p.compute(_nsize);
    else if(_p.grid == 0)
        _p.grid = 1;
    std::forward<FuncT>(_func)<<<_p.grid, _p.block, _p.shmem, _stream>>>(
        std::forward<ArgsT>(_args)...);
    TIMEMORY_GPU_RUNTIME_CHECK_ERROR(::tim::gpu::get_last_error());
    consume_parameters(_args...);
#else
    // static_assert(false, "Checking");
    consume_parameters(_p, _func, _args..., _nsize, _stream);
    throw std::runtime_error(
        "Launch specified on GPU but not compiled with GPU support!");
#endif
}

//======================================================================================//
//
//  These provide loop parameters for grid-strided loops on GPU or traditional loops on
//  CPU
//
//======================================================================================//

template <typename DeviceT, size_t DIM = 0, typename Intp = int32_t>
struct grid_strided_range : impl::range<Intp>
{
    using base_type = impl::range<Intp>;
    static_assert(DIM < 0 || DIM > 2,
                  "Error DIM parameter must be 0 (x), 1 (y), or 2 (z)");
};

//--------------------------------------------------------------------------------------//
// overload for 0/x
//
template <typename DeviceT, typename Intp>
struct grid_strided_range<DeviceT, 0, Intp> : impl::range<Intp>
{
    using base_type = impl::range<Intp>;

#if(defined(TIMEMORY_USE_CUDA) || defined(TIMEMORY_USE_HIP)) &&                          \
    defined(TIMEMORY_GPUCC) && !defined(TIMEMORY_OPENMP_TARGET)
    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::gpu>::value> = 0>
    TIMEMORY_DEVICE_FUNCTION explicit grid_strided_range(Intp max_iter)
    : base_type(blockIdx.x * blockDim.x + threadIdx.x, max_iter, blockDim.x * gridDim.x)
    {}

    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

//--------------------------------------------------------------------------------------//
// overload for 1/y
//
template <typename DeviceT, typename Intp>
struct grid_strided_range<DeviceT, 1, Intp> : impl::range<Intp>
{
    using base_type = impl::range<Intp>;

#if(defined(TIMEMORY_USE_CUDA) || defined(TIMEMORY_USE_HIP)) &&                          \
    defined(TIMEMORY_GPUCC) && !defined(TIMEMORY_OPENMP_TARGET)
    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::gpu>::value> = 0>
    TIMEMORY_DEVICE_FUNCTION explicit grid_strided_range(Intp max_iter)
    : base_type(blockIdx.y * blockDim.y + threadIdx.y, max_iter, blockDim.y * gridDim.y)
    {}

    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

//--------------------------------------------------------------------------------------//
// overload for 2/z
//
template <typename DeviceT, typename Intp>
struct grid_strided_range<DeviceT, 2, Intp> : impl::range<Intp>
{
    using base_type = impl::range<Intp>;

#if(defined(TIMEMORY_USE_CUDA) || defined(TIMEMORY_USE_HIP)) &&                          \
    defined(TIMEMORY_GPUCC) && !defined(TIMEMORY_OPENMP_TARGET)
    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::gpu>::value> = 0>
    TIMEMORY_DEVICE_FUNCTION explicit grid_strided_range(Intp max_iter)
    : base_type(blockIdx.z * blockDim.z + threadIdx.z, max_iter, blockDim.z * gridDim.z)
    {}

    template <typename DevT                                       = DeviceT,
              enable_if_t<std::is_same<DevT, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

}  // namespace device
}  // namespace tim
