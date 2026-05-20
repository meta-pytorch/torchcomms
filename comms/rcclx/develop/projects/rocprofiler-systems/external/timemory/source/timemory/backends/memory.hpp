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

#include "timemory/backends/device.hpp"
#include "timemory/defines.h"
#include "timemory/macros/os.hpp"
#include "timemory/units.hpp"

#if !defined(TIMEMORY_WINDOWS)
#    define RESTRICT(TYPE) TYPE __restrict__
#else
#    define RESTRICT(TYPE) TYPE
#endif

#if defined(__INTEL_COMPILER)
#    define ASSUME_ALIGNED_MEMORY(ARRAY, WIDTH) __assume_aligned(ARRAY, WIDTH)
#elif defined(__xlC__)
#    define ASSUME_ALIGNED_MEMORY(ARRAY, WIDTH) __alignx(WIDTH, ARRAY)
#else
#    define ASSUME_ALIGNED_MEMORY(ARRAY, WIDTH)
#endif

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>

// provide allocation functions
#if defined(TIMEMORY_LINUX) || defined(TIMEMORY_WINDOWS)
#    include <malloc.h>
#elif defined(TIMEMORY_MACOS)
#    include <malloc/malloc.h>
#endif

#if defined(TIMEMORY_LINUX)
#    include <sys/sysinfo.h>
#endif

#if defined(TIMEMORY_MACOS)
#    include <sys/sysctl.h>
#endif

namespace tim
{
namespace memory
{
//--------------------------------------------------------------------------------------//
//  aligned allocation, alignment should be specified in bits
//
namespace hidden
{
template <typename Tp, typename DeviceT>
Tp*
allocate_aligned(std::size_t size, std::size_t alignment,
                 std::enable_if_t<std::is_same<DeviceT, device::cpu>::value, int> = 0)
{
#if defined(_ISOC11_SOURCE)
    return static_cast<Tp*>(aligned_alloc(alignment, size * sizeof(Tp)));
#elif defined(TIMEMORY_MACOS) ||                                                         \
    (defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L))
    void* ptr = nullptr;
    auto  ret = posix_memalign(&ptr, alignment, size * sizeof(Tp));
    (void) ret;
    return static_cast<Tp*>(ptr);
#elif defined(TIMEMORY_WINDOWS)
    return static_cast<Tp*>(_aligned_malloc(size * sizeof(Tp), alignment));
#else
    return new Tp[size];
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT>
Tp*
allocate_aligned(std::size_t size, std::size_t,
                 std::enable_if_t<std::is_same<DeviceT, device::gpu>::value, int> = 0)
{
    return gpu::malloc<Tp>(size);
}

//--------------------------------------------------------------------------------------//
//  free aligned array, should be used in conjunction with data allocated with above
//
template <typename Tp, typename DeviceT>
void
free_aligned(Tp* ptr,
             std::enable_if_t<std::is_same<DeviceT, device::cpu>::value, int> = 0)
{
#if defined(_ISOC11_SOURCE) || defined(TIMEMORY_MACOS) || (_POSIX_C_SOURCE >= 200112L)
    free(static_cast<void*>(ptr));
#elif defined(TIMEMORY_WINDOWS)
    _aligned_free(ptr);
#else
    delete[] ptr;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT>
void
free_aligned(Tp* ptr,
             std::enable_if_t<std::is_same<DeviceT, device::gpu>::value, int> = 0)
{
    gpu::free(ptr);
}

}  // namespace hidden

//--------------------------------------------------------------------------------------//
//  aligned allocation, alignment should be specified in bits
//
template <typename Tp, typename DeviceT = device::cpu>
Tp*
allocate_aligned(std::size_t size, std::size_t alignment)
{
    return hidden::allocate_aligned<Tp, DeviceT>(size, alignment);
}

//--------------------------------------------------------------------------------------//
//  free aligned array, should be used in conjunction with data allocated with above
//
template <typename Tp, typename DeviceT = device::cpu>
void
free_aligned(Tp* ptr)
{
    hidden::free_aligned<Tp, DeviceT>(ptr);
}

template <typename DeviceT = device::cpu>
int64_t
total_memory();

template <typename DeviceT = device::cpu>
int64_t
free_memory();

//--------------------------------------------------------------------------------------//
///  total memory in bytes
//
template <>
inline int64_t
total_memory<device::cpu>()
{
#if defined(TIMEMORY_LINUX)
    struct sysinfo _info
    {};
    if(sysinfo(&_info) == 0)
    {
        // sizes of the memory and swap fields are given as multiples of mem_unit bytes.
        return _info.totalram * _info.mem_unit;
    }
    else
    {
        std::ifstream _ifs{ "/proc/meminfo" };
        if(_ifs)
        {
            std::string _discard;
            for(int i = 0; i < 1; ++i)
                _ifs >> _discard;
            int64_t _value = 0;
            _ifs >> _value;
            return _value * units::kilobyte;
        }
    }
    return 0;
#elif defined(TIMEMORY_MACOS)
    int64_t count     = 0;
    size_t  count_len = sizeof(count);
    sysctlbyname("hw.memsize", &count, &count_len, nullptr, 0);
    return count * units::byte;
#elif defined(TIMEMORY_WINDOWS)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);

    return statex.ullTotalPhys * units::byte;  // reports in bytes
#else
    static_assert(false, "OS not supported");
#endif
}

//--------------------------------------------------------------------------------------//
///  available memory in bytes
//
template <>
inline int64_t
free_memory<device::cpu>()
{
#if defined(TIMEMORY_LINUX)
    struct sysinfo _info
    {};
    if(sysinfo(&_info) == 0)
    {
        // sizes of the memory and swap fields are given as multiples of mem_unit bytes.
        return _info.freeram * _info.mem_unit;
    }
    else
    {
        std::ifstream _ifs{ "/proc/meminfo" };
        if(_ifs)
        {
            std::string _discard;
            for(int i = 0; i < 4; ++i)
                _ifs >> _discard;
            int64_t _value = 0;
            _ifs >> _value;
            return _value * units::kilobyte;
        }
    }
    return 0;
#elif defined(TIMEMORY_MACOS)
    vm_statistics64_t stat  = nullptr;
    unsigned int      count = HOST_VM_INFO64_COUNT;
    if(host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) stat,
                         &count) != KERN_SUCCESS)
    {
        fprintf(stderr, "[%s] failed to get statistics for free memory\n",
                TIMEMORY_PROJECT_NAME);
        return 0;
    }
    return (stat) ? (stat->free_count * units::get_page_size()) : 0;
#elif defined(TIMEMORY_WINDOWS)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);

    return statex.ullAvailPhys * units::byte;  // reports in bytes
#else
    static_assert(false, "OS not supported");
#endif
}
}  // namespace memory
}  // namespace tim
