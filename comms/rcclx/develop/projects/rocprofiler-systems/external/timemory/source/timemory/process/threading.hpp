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

#ifndef TIMEMORY_PROCESS_THREADING_HPP_
#    define TIMEMORY_PROCESS_THREADING_HPP_
#endif

#include "timemory/defines.h"
#include "timemory/macros/os.hpp"
#include "timemory/process/defines.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(TIMEMORY_LINUX)
#    include <pthread.h>
#    include <sys/syscall.h>
#endif

#if defined(TIMEMORY_LINUX)
#    include <fstream>
#    include <unistd.h>
#endif

#if defined(TIMEMORY_MACOS)
#    include <pthread.h>
#    include <sys/sysctl.h>
#endif

#if defined(TIMEMORY_WINDOWS)
#    include <processthreadsapi.h>
#endif

#if !defined(TIMEMORY_MAX_THREADING_CALLBACKS)
#    define TIMEMORY_MAX_THREADING_CALLBACKS 8
#endif

namespace tim
{
namespace threading
{
using id_t            = std::thread::id;
using native_handle_t = std::thread::native_handle_type;
//
//--------------------------------------------------------------------------------------//
//
namespace internal
{
struct thread_id_manager;
//
std::pair<int64_t, scope::destructor>
get_id(thread_id_manager*, bool = false);
//
struct thread_id_manager
{
    using callback_t       = std::atomic<bool (*)(bool, int64_t, int64_t)>;
    using callback_array_t = std::array<callback_t, TIMEMORY_MAX_THREADING_CALLBACKS>;

    explicit thread_id_manager(int64_t _max_threads = TIMEMORY_MAX_THREADS);

    int64_t              max_threads;
    std::atomic<int64_t> global_counter;
    std::atomic<int64_t> offset_counter;
    std::vector<int64_t> available = {};
    std::vector<int64_t> offset    = {};
    std::set<int64_t>    reserved  = { 0 };
    callback_array_t     callbacks = { nullptr };
};
//
inline auto&
get_manager()
{
    static auto* _v = new thread_id_manager{};
    return _v;
}
//
struct offset_id
{
    bool is_mutable = true;
    bool is_offset  = false;
};
//
inline offset_id&
offset_this_id()
{
    static thread_local auto _v = offset_id{};
    return _v;
}
//
struct recycle_ids
{
    explicit operator bool() const { return value; }

#if defined(TIMEMORY_FORCE_UNIQUE_THREAD_IDS) && TIMEMORY_FORCE_UNIQUE_THREAD_IDS > 0
    // ignore assignments
    recycle_ids& operator=(bool) { return *this; }
#else
    // allow assignments
    recycle_ids& operator=(bool _v)
    {
        value = _v;
        return *this;
    }
#endif

private:
    bool value = false;
};
//
}  // namespace internal
//
//--------------------------------------------------------------------------------------//
//
inline internal::recycle_ids&
recycle_ids()
{
    static auto _v = internal::recycle_ids{};
    return _v;
}
//
inline void
offset_this_id(bool _v)
{
    if(internal::offset_this_id().is_mutable)
        internal::offset_this_id().is_offset = _v;
}
//
inline bool
offset_this_id()
{
    return internal::offset_this_id().is_offset;
}
//
inline int64_t
get_id()
{
    static thread_local auto _this_id = internal::get_id(internal::get_manager());
    return _this_id.first;
}
//
int
add_callback(bool (*_func)(bool, int64_t, int64_t));
//
bool
remove_callback(bool (*_func)(bool, int64_t, int64_t));
//
void
clear_callbacks();
//
std::set<int64_t>
add_reserved_id(int64_t _v = get_id());
//
std::set<int64_t>
erase_reserved_id(int64_t _v = get_id());
//
inline id_t
get_tid()
{
    return std::this_thread::get_id();
}
//
inline id_t
get_main_tid()
{
    static id_t _instance = get_tid();
    return _instance;
}
//
inline bool
is_main_thread()
{
    return (get_tid() == get_main_tid());
}
//
inline long
get_sys_tid()
{
#if defined(TIMEMORY_LINUX)
    return syscall(SYS_gettid);
#elif defined(TIMEMORY_MACOS)
    uint64_t _v = 0;
    pthread_threadid_np(pthread_self(), &_v);
    return _v;
#elif defined(TIMEMORY_WINDOWS)
    return GetCurrentThreadId();
#else
    return static_cast<long>(get_id());
#endif
}
//
void
set_thread_name(const char* _name);
//
std::string
get_thread_name();
//
//--------------------------------------------------------------------------------------//
//
struct affinity
{
    using functor_t          = std::function<int64_t(int64_t)>;
    using cpu_affinity_map_t = std::map<int64_t, int64_t>;

    enum MODE
    {
        COMPACT  = 0,
        SCATTER  = 1,
        SPREAD   = 2,
        EXPLICIT = 3
    };

    static int64_t hw_concurrency() { return std::thread::hardware_concurrency(); }
    static int64_t hw_physicalcpu();

    static auto& get_affinity_map()
    {
        static cpu_affinity_map_t _instance;
        return _instance;
    }

    static MODE& get_mode()
    {
        static MODE _instance = COMPACT;
        return _instance;
    }

    static functor_t& get_algorithm();
    static int64_t    set();
    static int64_t    set(native_handle_t athread);
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace threading
}  // namespace tim

#if defined(TIMEMORY_PROCESS_HEADER_MODE) && TIMEMORY_PROCESS_HEADER_MODE > 0
#    include "timemory/process/threading.cpp"
#endif
