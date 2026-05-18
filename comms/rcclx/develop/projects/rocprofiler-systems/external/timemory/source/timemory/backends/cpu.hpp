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

#include "timemory/macros/arch.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>

#if defined(TIMEMORY_MACOS)
#    include <sys/sysctl.h>
#endif

namespace tim
{
namespace cpu
{
struct cpu_info
{
    long        frequency = 0;
    std::string vendor    = {};
    std::string model     = {};
    std::string features  = {};
};

TIMEMORY_INLINE void
memory_fence()
{
#if defined(TIMEMORY_ARCH_X86) && TIMEMORY_ARCH_X86 > 0
#    if !defined(TIMEMORY_WINDOWS)
    asm volatile("mfence;" : : : "memory");
#    endif
#elif defined(_OPENMP)
#    pragma omp flush
#endif
}
//
TIMEMORY_INLINE void
store_fence()
{
#if defined(TIMEMORY_ARCH_X86) && TIMEMORY_ARCH_X86 > 0
#    if !defined(TIMEMORY_WINDOWS)
    asm volatile("sfence;" : : : "memory");
#    endif
#else
    memory_fence();
#endif
}
//
TIMEMORY_INLINE void
load_fence()
{
#if defined(TIMEMORY_ARCH_X86) && TIMEMORY_ARCH_X86 > 0
#    if !defined(TIMEMORY_WINDOWS)
    asm volatile("lfence;" : : : "memory");
#    endif
#else
    memory_fence();
#endif
}
//
TIMEMORY_INLINE void
fence()
{
    memory_fence();
}
//

inline cpu_info
get_info()
{
#if defined(TIMEMORY_MACOS)

    cpu_info _info{};

    std::array<char, 512> _buf{};
    size_t                _len   = _buf.size();
    auto                  _reset = [&_buf, &_len]() {
        _buf = {};
        _len = _buf.size();
    };

    if(sysctlbyname("machdep.cpu.brand_string", _buf.data(), &_len, nullptr, 0) == 0)
    {
        _info.model = _buf.data();
    }
    else
        TIMEMORY_TESTING_EXCEPTION("'machdep.cpu.brand_string' failed");

    _reset();
    if(sysctlbyname("machdep.cpu.vendor", _buf.data(), &_len, nullptr, 0) == 0)
    {
        _info.vendor = _buf.data();
    }
    else
        TIMEMORY_TESTING_EXCEPTION("'sysctl machdep.cpu.vendor' failed");

    _reset();
    auto _lsz = sizeof(_info.frequency);
    if(sysctlbyname("hw.cpufrequency", &_info.frequency, &_lsz, nullptr, 0) != 0)
        TIMEMORY_TESTING_EXCEPTION("'sysctl hw.cpufrequency' failed");

    _reset();
    if(sysctlbyname("machdep.cpu.features", _buf.data(), &_len, nullptr, 0) == 0)
    {
        _info.features = _buf.data();
        _reset();
        if(sysctlbyname("machdep.cpu.leaf7_features", _buf.data(), &_len, nullptr, 0) ==
           0)
        {
            _info.features += " " + std::string{ _buf.data() };
        }
        else
            TIMEMORY_TESTING_EXCEPTION("'sysctl machdep.cpu.leaf7_features' failed");
    }
    else
    {
        TIMEMORY_TESTING_EXCEPTION("'sysctl machdep.cpu.features' failed");
    }

    return _info;

#elif defined(TIMEMORY_WINDOWS)

    return cpu_info{};

#elif defined(TIMEMORY_LINUX)

    std::ifstream ifs("/proc/cpuinfo");
    if(!ifs)
        return cpu_info{};

    cpu_info         _info{};
    std::string      line{};
    const std::regex re{ ".*: (.*)$" };

    while(std::getline(ifs, line))
    {
        if(ifs.eof())
            break;

        std::smatch match;
        if(std::regex_match(line, match, re))
        {
            if(match.size() == 2)
            {
                std::ssub_match value = match[1];

                if(line.find("model name") == 0)
                    _info.model = value.str();
                else if(line.find("cpu MHz") == 0)
                    _info.frequency = atoi(value.str().c_str());
                else if(line.find("flags") == 0)
                    _info.features = value.str();
                else if(line.find("vendor_id") == 0)
                    _info.vendor = value.str();
            }
        }
    }

    return _info;

#else

    return cpu_info{};

#endif
}
namespace cache_size
{
//
//  get the size of the L1 (data), L2, or L3 cache
//
namespace impl
{
inline size_t
cache_size(int level)
{
#if defined(TIMEMORY_MACOS)

    // configure sysctl query
    //      L1  ->  hw.l1dcachesize
    //      L2  ->  hw.l2cachesize
    //      L3  ->  hw.l3cachesize
    //
    const auto query = std::string{ "hw.l" } + std::to_string(level) +
                       ((level == 1) ? "d" : "") + "cachesize";
    size_t line_size        = 0;
    size_t sizeof_line_size = sizeof(line_size);
    sysctlbyname(query.c_str(), &line_size, &sizeof_line_size, nullptr, 0);
    return line_size;

#elif defined(TIMEMORY_WINDOWS)

    size_t                                line_size   = 0;
    DWORD                                 buffer_size = 0;
    DWORD                                 i           = 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer      = 0;

    GetLogicalProcessorInformation(0, &buffer_size);
    buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*) malloc(buffer_size);
    if(!buffer)
        return static_cast<size_t>(4096);
    GetLogicalProcessorInformation(&buffer[0], &buffer_size);

    for(i = 0; i != buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i)
    {
        if(buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == level)
        {
            line_size = buffer[i].Cache.Size;
            break;
        }
    }

    free(buffer);
    return line_size;

#elif defined(TIMEMORY_LINUX)

    // L1 has a data and instruction cache, index0 should be data
    static const std::array<std::string, 3> fpaths(
        { { "/sys/devices/system/cpu/cpu0/cache/index0/",
            "/sys/devices/system/cpu/cpu0/cache/index2/",
            "/sys/devices/system/cpu/cpu0/cache/index3/" } });

    // L1 at 0, L2 at 1, L3 at 2
    const std::string& fpath = fpaths.at(level - 1);

    // files to read
    static const std::array<std::string, 3> files(
        { { "number_of_sets", "ways_of_associativity", "coherency_line_size" } });

    uint64_t product = 1;
    for(const auto& itr : files)
    {
        std::ifstream ifs(fpath + itr);
        if(ifs)
        {
            uint64_t val;
            ifs >> val;
            product *= val;
        }
        else
        {
            return 0;
        }
        ifs.close();
    }
    return (product > 1) ? product : 0;

#else

#    warning Unrecognized platform
    return 0;

#endif
}
}  // namespace impl

//======================================================================================//

template <size_t Level>
inline size_t
get()
{
    // only enable queries 1, 2, 3
    static_assert(Level > 0 && Level < 4,
                  "Request for cache level that is not supported");

    // avoid multiple queries
    static size_t _value = impl::cache_size(Level);
    return _value;
}

//--------------------------------------------------------------------------------------//

inline size_t
get(int _level)
{
    // only enable queries 1, 2, 3
    if(_level < 1 || _level > 3)
    {
        fprintf(stderr,
                "im::ert::cache_size::get(%i) :: Requesting invalid cache level\n",
                _level);
#if defined(TIMEMORY_INTERNAL_TESTING)
        exit(EXIT_FAILURE);
#endif
    }
    // avoid multiple queries
    static const std::array<size_t, 3> _values(
        { { impl::cache_size(1), impl::cache_size(2), impl::cache_size(3) } });
    return _values.at(_level - 1);
}

//--------------------------------------------------------------------------------------//

inline size_t
get_max()
{
    // this is useful for system like KNL that do not have L3 cache
    for(auto level : { 3, 2, 1 })
    {
        size_t sz = impl::cache_size(level);
        // return first non-zero value
        if(sz > 0)
            return sz;
    }
    return 0;
}

//--------------------------------------------------------------------------------------//

}  // namespace cache_size
}  // namespace cpu
}  // namespace tim
