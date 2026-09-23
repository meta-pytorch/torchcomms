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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/**
 * \file timemory/components/rusage/backends.hpp
 * \brief Implementation of the rusage functions/utilities
 */

#pragma once

#include "timemory/backends/process.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/variadic/macros.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <string>
#include <type_traits>

#if defined(TIMEMORY_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(TIMEMORY_MACOS)
#        include <libproc.h>
#        include <mach/mach.h>
#    endif
#endif

//======================================================================================//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
struct io_cache
{
    static inline auto get_filename()
    {
        return TIMEMORY_JOIN('/', "/proc", process::get_target_id(), "io");
    }

    template <size_t N, size_t... Idx>
    static inline auto& read(std::ifstream& ifs, std::array<int64_t, N>& _data,
                             std::index_sequence<Idx...>)
    {
        static_assert(N <= 6, "Error! Only six entries in the /proc/<PID>/io");
        static_assert(N > 0, "Error! array size is zero");
        static_assert(sizeof...(Idx) <= N,
                      "Error! Number of indexes to read exceeds the array size");
        if(ifs)
        {
            std::string label{};
            TIMEMORY_FOLD_EXPRESSION(ifs >> label >> std::get<Idx>(_data));
        }
        else
        {
            _data.fill(0);
        }
        return _data;
    }

    template <size_t NumReads = 6, size_t N>
    static inline auto& read(std::array<int64_t, N>& _data)
    {
        std::ifstream ifs(get_filename().c_str());
        _data = read(ifs, _data, std::make_index_sequence<NumReads>{});
        return _data;
    }

    template <size_t NumReads = 6>
    static inline auto read()
    {
        std::array<int64_t, NumReads> _data{};
        _data = read<NumReads>(_data);
        return _data;
    }

    friend std::ostream& operator<<(std::ostream& os, const io_cache& obj)
    {
        std::stringstream ss;
        ss << "rchar: " << obj.get_char_read() << '\n';
        ss << "wchar: " << obj.get_char_written() << '\n';
        ss << "syscr: " << obj.get_syscall_read() << '\n';
        ss << "syscw: " << obj.get_syscall_written() << '\n';
        ss << "read_bytes: " << obj.get_bytes_read() << '\n';
        ss << "write_bytes: " << obj.get_bytes_written() << '\n';
        os << ss.str();
        return os;
    }

public:
    io_cache() { update(); }
    ~io_cache() = default;

    io_cache(const io_cache&) = delete;
    io_cache& operator=(const io_cache&) = delete;

    io_cache(io_cache&&) noexcept = default;
    io_cache& operator=(io_cache&&) noexcept = default;

    inline void update()
    {
#if defined(TIMEMORY_LINUX)
        m_data = read();
#endif
    }

#if !defined(TIMEMORY_LINUX)

    inline int64_t get_char_read() const { return 0; }        // NOLINT
    inline int64_t get_char_written() const { return 0; }     // NOLINT
    inline int64_t get_syscall_read() const { return 0; }     // NOLINT
    inline int64_t get_syscall_written() const { return 0; }  // NOLINT
    inline int64_t get_bytes_read() const { return 0; }       // NOLINT
    inline int64_t get_bytes_written() const { return 0; }    // NOLINT

#else

    inline int64_t get_char_read() const { return std::get<0>(m_data); }
    inline int64_t get_char_written() const { return std::get<1>(m_data); }
    inline int64_t get_syscall_read() const { return std::get<2>(m_data); }
    inline int64_t get_syscall_written() const { return std::get<3>(m_data); }
    inline int64_t get_bytes_read() const { return std::get<4>(m_data); }
    inline int64_t get_bytes_written() const { return std::get<5>(m_data); }

private:
    std::array<int64_t, 6> m_data{};
#endif
};

#if defined(TIMEMORY_WINDOWS)
#    if !defined(TIMEMORY_WIN_IO_MIN_DELAY_MSEC)
#        define TIMEMORY_WIN_IO_MIN_DELAY_MSEC 0
#    endif
struct win_io_counters
{
    static auto& instance()
    {
        static auto _instance = win_io_counters{};
        return _instance;
    }

    inline int64_t get_bytes_read()
    {
        update();
        return static_cast<int64_t>(m_io_counters.ReadTransferCount);
    }

    inline int64_t get_bytes_written()
    {
        update();
        return static_cast<int64_t>(m_io_counters.WriteTransferCount);
    }

private:
#    if TIMEMORY_WIN_IO_MIN_DELAY_MSEC
    using clock_t              = std::chrono::steady_clock;
    using time_point_t         = clock_t::time_point;
    time_point_t m_last_update = time_point_t::min();
#    endif
    IO_COUNTERS m_io_counters;

    win_io_counters() { update(); }

    bool should_update() const
    {
#    if TIMEMORY_WIN_IO_MIN_DELAY_MSEC
        auto now = clock_t::now();
        if(std::chrono::duration_cast<std::chrono::milliseconds>(now - m_last_update) >
           std::chrono::milliseconds{ 1000 })
            return true;
        return false;
#    else
        return true;
#    endif
    }

    void update()
    {
        if(should_update())
        {
            static auto handle = GetCurrentProcess();
            if(!GetProcessIoCounters(handle, &m_io_counters))
            {
                m_io_counters.ReadTransferCount = m_io_counters.ReadOperationCount = 0;
                m_io_counters.WriteTransferCount = m_io_counters.WriteOperationCount = 0;
                m_io_counters.OtherTransferCount = m_io_counters.OtherOperationCount = 0;
            }
        }
    }
};
#    undef TIMEMORY_WIN_IO_MIN_DELAY_MSEC
#endif
//
//--------------------------------------------------------------------------------------//
//
int64_t
get_char_read();
int64_t
get_char_written();
int64_t
get_syscall_read();
int64_t
get_syscall_written();
int64_t
get_bytes_read();
int64_t
get_bytes_written();
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_char_read()
{
#if defined(TIMEMORY_LINUX)
    // read one value and return it
    return io_cache::read<1>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_char_written()
{
#if defined(TIMEMORY_LINUX)
    // read two values and return the last one
    return io_cache::read<2>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_syscall_read()
{
#if defined(TIMEMORY_LINUX)
    // read one value and return it
    return io_cache::read<3>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_syscall_written()
{
#if defined(TIMEMORY_LINUX)
    // read two values and return the last one
    return io_cache::read<4>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_bytes_read()
{
#if defined(TIMEMORY_MACOS)
    rusage_info_current rusage;
    if(proc_pid_rusage(process::get_target_id(), RUSAGE_INFO_CURRENT, (void**) &rusage) ==
       0)
        return rusage.ri_diskio_bytesread;
    return 0;
#elif defined(TIMEMORY_LINUX)
    // read three values and return the last one
    return io_cache::read<5>().back();
#elif defined(TIMEMORY_WINDOWS)
    return win_io_counters::instance().get_bytes_read();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_bytes_written()
{
#if defined(TIMEMORY_MACOS)
    rusage_info_current rusage;
    if(proc_pid_rusage(process::get_target_id(), RUSAGE_INFO_CURRENT, (void**) &rusage) ==
       0)
        return rusage.ri_diskio_byteswritten;
    return 0;
#elif defined(TIMEMORY_LINUX)
    // read four values and return the last one
    return io_cache::read<6>().back();
#elif defined(TIMEMORY_WINDOWS)
    return win_io_counters::instance().get_bytes_written();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
