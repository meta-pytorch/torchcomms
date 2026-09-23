//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#ifndef TIMEMORY_PROCESS_THREADING_CPP_
#define TIMEMORY_PROCESS_THREADING_CPP_

#include "timemory/process/defines.hpp"

#if !defined(TIMEMORY_PROCESS_THREADING_HPP_)
#    include "timemory/process/threading.hpp"
#endif

#include "timemory/defines.h"
#include "timemory/environment/types.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

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

namespace tim
{
namespace threading
{
namespace internal
{
TIMEMORY_PROCESS_INLINE
thread_id_manager::thread_id_manager(int64_t _max_threads)
: max_threads{ _max_threads }
, global_counter{ 0 }
, offset_counter{ _max_threads }
{
    available.reserve(max_threads);
    offset.reserve(max_threads);
}

TIMEMORY_PROCESS_INLINE
std::pair<int64_t, scope::destructor>
get_id(thread_id_manager* _manager, bool _debug)
{
    static bool debug_threading_get_id =
        get_env<bool>(TIMEMORY_SETTINGS_PREFIX "DEBUG_THREADING_GET_ID", false);

    if(!_debug)
        _debug = debug_threading_get_id;

    int64_t _id = -1;
    if(!_manager)
        return std::make_pair(_id, scope::destructor{ []() {} });

    auto& _global_counter = _manager->global_counter;
    auto& _offset_counter = _manager->offset_counter;
    auto& _offset         = internal::offset_this_id();

    if(_offset.is_offset && _global_counter == 0)
        _offset = { true, false };

    static thread_local int _protect = 0;
    if(_global_counter > 0 && (_protect & 1) == 0)
    {
        ++_protect;
        auto _gc_v = _global_counter.load();
        auto _of_v = _offset_counter.load();
        for(auto& itr : _manager->callbacks)
        {
            auto _cb = itr.load();
            if(!_cb)
                continue;

            auto _req = (*_cb)(_offset.is_offset, _gc_v, _of_v);
            if(_req != _offset.is_offset)
            {
                if(_offset.is_mutable)
                    _offset.is_offset = _req;
                else
                {
                    std::stringstream _msg{};
                    _msg << std::boolalpha;
                    _msg << "[timemory] callback for threading::get_id() tried to change "
                            "the offset value which was already designated as immutable. "
                            "global_counter: "
                         << _gc_v << ", offset_counter: " << _of_v
                         << ", current offset value: " << _offset.is_offset;
                    throw std::invalid_argument(_msg.str());
                }
            }
        }
        ++_protect;
    }

    if(_offset.is_offset)
    {
        if(recycle_ids())
        {
            auto_lock_t _lk{ type_mutex<internal::recycle_ids>(1) };
            auto&       _avail = _manager->offset;
            if(!_avail.empty())
            {
                // always grab from front
                _id = _avail.front();
                for(size_t i = 1; i < _avail.size(); ++i)
                    _avail[i - 1] = _avail[i];
                _avail.pop_back();
            }
        }
    }
    else if(recycle_ids() && _global_counter >= _offset_counter)
    {
        auto_lock_t _lk{ type_mutex<internal::recycle_ids>() };
        auto&       _avail = _manager->available;
        if(!_avail.empty())
        {
            // always grab from front
            _id = _avail.front();
            for(size_t i = 1; i < _avail.size(); ++i)
                _avail[i - 1] = _avail[i];
            _avail.pop_back();
        }
    }

    if(_id < 0)
        _id = (_offset.is_offset) ? --_offset_counter : _global_counter++;

    if(_debug)
    {
        timemory_print_demangled_backtrace<8>(std::cerr, std::string{},
                                              std::string{ "threading::get_id() [id=" } +
                                                  std::to_string(_id) +
                                                  std::string{ "]" },
                                              std::string{ " " }, false);
    }

    // disable the ability to offset the thread id
    _offset.is_mutable = false;

    // below is used to avoid using a mutex for synchronization
    // when the thread is being destroyed.
    static auto _offset_lk = locking::spin_mutex{};
    static auto _avail_lk  = locking::spin_mutex{};

    auto&& _dtor = [_id, _offset]() {
        // skip for main thread
        if(_id == 0)
            return;
        auto* _manager_v = get_manager();
        if(!_manager_v)
            return;
        if(_offset.is_offset)
        {
            locking::spin_lock _lk{ _offset_lk };
            _manager_v->offset.emplace_back(_id);
        }
        else
        {
            if(_manager_v->reserved.count(_id) == 0)
            {
                locking::spin_lock _lk{ _avail_lk };
                _manager_v->available.emplace_back(_id);
            }
        }
    };

    return std::make_pair(_id, scope::destructor{ std::move(_dtor) });
}
}  // namespace internal

// clang-format off
TIMEMORY_PROCESS_INLINE
int
add_callback(bool (*_func)(bool, int64_t, int64_t))
// clang-format on
{
    auto*& _manager = internal::get_manager();
    if(!_manager)
        return -1;

    size_t _idx = 0;
    for(; _idx < _manager->callbacks.size(); ++_idx)
    {
        auto& itr   = _manager->callbacks.at(_idx);
        auto  _targ = itr.load();
        if(!_targ && _func)
        {
            if(itr.compare_exchange_strong(_targ, _func) && itr.load() == _func)
                return static_cast<int>(_idx);
        }
        else if(_targ && _func && _targ == _func)
        {
            return static_cast<int>(_idx);
        }
    }
    return -1;
}

// clang-format off
TIMEMORY_PROCESS_INLINE
bool
remove_callback(bool (*_func)(bool, int64_t, int64_t))
// clang-format on
{
    auto*& _manager = internal::get_manager();
    if(!_manager)
        return true;

    size_t _idx = 0;
    for(; _idx < _manager->callbacks.size(); ++_idx)
    {
        auto& itr   = _manager->callbacks.at(_idx);
        auto  _targ = itr.load();
        if(_targ == _func)
        {
            itr.store(nullptr);
            return true;
        }
    }
    return false;
}

TIMEMORY_PROCESS_INLINE
void
clear_callbacks()
{
    auto*& _manager = internal::get_manager();
    if(!_manager)
        return;

    for(auto& itr : _manager->callbacks)
        itr.store(nullptr);
}

// clang-format off
TIMEMORY_PROCESS_INLINE
std::set<int64_t>
add_reserved_id(int64_t _v)
// clang-format on
{
    auto_lock_t _lk{ type_mutex<internal::recycle_ids>() };
    auto*       _manager = internal::get_manager();
    if(!_manager)
        return std::set<int64_t>{};
    else if(_v > 0)
        _manager->reserved.emplace(_v);
    return _manager->reserved;
}

// clang-format off
TIMEMORY_PROCESS_INLINE
std::set<int64_t>
erase_reserved_id(int64_t _v)
// clang-format on
{
    auto_lock_t _lk{ type_mutex<internal::recycle_ids>() };
    auto*       _manager = internal::get_manager();
    if(!_manager)
        return std::set<int64_t>{};
    if(_v > 0)
        _manager->reserved.erase(_v);
    return _manager->reserved;
}

// clang-format off
TIMEMORY_PROCESS_INLINE
void
set_thread_name(const char* _name)
// clang-format on
{
#if defined(TIMEMORY_UNIX)
    auto _length_error = [_name]() {
        fprintf(stderr,
                "[threading::set_thread_name] the length of '%s' + null-terminator (%i) "
                "exceeds the max allowed limit (usually 16)\n",
                _name, (int) (strlen(_name) + 1));
    };

    constexpr size_t _size = 16;
    size_t           _n    = std::min<size_t>(_size - 1, strlen(_name));
    char             _buff[_size];
    memset(_buff, '\0', _size * sizeof(char));
    memcpy(_buff, _name, _n * sizeof(char));
#endif

#if defined(TIMEMORY_LINUX)
    auto _err = pthread_setname_np(pthread_self(), _buff);
    if(_err == ERANGE)
        _length_error();
#elif defined(TIMEMORY_MACOS)
    auto _err = pthread_setname_np(_buff);
    if(_err == ERANGE)
        _length_error();
#elif defined(TIMEMORY_WINDOWS)
    auto     _n     = strlen(_name);
    wchar_t* _wname = new wchar_t[_n + 1];
    for(size_t i = 0; i < _n; ++i)
        _wname[i] = _name[i];
    _wname[_n] = '\0';
    SetThreadDescription(GetCurrentThread(), _wname);
#endif
}

// clang-format off
TIMEMORY_PROCESS_INLINE
std::string
get_thread_name()
// clang-format on
{
#if defined(TIMEMORY_UNIX)
    constexpr size_t _buff_len = 32;
    char             _buff[_buff_len];
    memset(_buff, '\0', _buff_len * sizeof(char));
    auto _err = pthread_getname_np(pthread_self(), _buff, _buff_len);
    if(_err == ERANGE)
    {
        fprintf(stderr,
                "[threading::get_thread_name] buffer for pthread_getname_np was not "
                "large enough: %zu\n",
                _buff_len);
    }
    return std::string{ _buff };
#elif defined(TIMEMORY_WINDOWS)
    wchar_t*    data  = nullptr;
    std::string _name = {};
    HRESULT     hr    = GetThreadDescription(GetCurrentThread(), &data);
    if(SUCCEEDED(hr))
    {
        constexpr size_t _buff_len = 64;
        char             _buff[_buff_len];
        _name.resize(_buff_len);
        for(size_t i = 0; i < _buff_len; ++i)
        {
            _name[i] = data[i];
            if(data[i] == '\0')
                break;
        }
        LocalFree(data);
        _name.shrink_to_fit();
    }
    return _name;
#endif
}

TIMEMORY_PROCESS_INLINE
int64_t
affinity::hw_physicalcpu()
{
    static int64_t _value = []() -> int64_t {
#if defined(TIMEMORY_MACOS)
        int    count;
        size_t count_len = sizeof(count);
        sysctlbyname("hw.physicalcpu", &count, &count_len, nullptr, 0);
        return static_cast<int64_t>(count);
#elif defined(TIMEMORY_LINUX)
        std::ifstream ifs("/proc/cpuinfo");
        if(ifs)
        {
            std::set<int64_t> core_ids;
            std::string       line;
            while(true)
            {
                getline(ifs, line);
                if(!ifs.good())
                    break;
                if(line.find("core id") != std::string::npos)
                {
                    auto cid = from_string<int64_t>(delimit(line, " :,;").back());
                    if(cid >= 0)
                        core_ids.insert(cid);
                }
            }
            return core_ids.size();
        }
        return hw_concurrency();
#else
        return hw_concurrency();
#endif
    }();
    return _value;
}

TIMEMORY_PROCESS_INLINE
affinity::functor_t&
affinity::get_algorithm()
{
    static functor_t _instance = [](int64_t tid) {
        //
        //  assigns the cpu affinity in a compact sequence
        //
        static functor_t _compact_instance = [](int64_t _tid) -> int64_t {
            static std::atomic<int64_t> _counter(0);
            static thread_local int64_t _this_count = _counter++;
            auto                        proc_itr    = get_affinity_map().find(_tid);
            if(proc_itr == get_affinity_map().end())
                get_affinity_map()[_tid] = _this_count;
            return get_affinity_map()[_tid];
        };
        //
        //  assigns the cpu affinity in a scattered sequence
        //
        static functor_t _scatter_instance = [](int64_t _tid) -> int64_t {
            static std::atomic<int64_t> _counter(0);
            static thread_local int64_t _this_count = _counter++;
            auto _val     = (_this_count * hw_physicalcpu()) % hw_concurrency();
            auto proc_itr = get_affinity_map().find(_tid);
            if(proc_itr == get_affinity_map().end())
                get_affinity_map()[_tid] = _val;
            return get_affinity_map()[_tid];
        };
        //
        //  assigns the cpu affinity explicitly
        //
        static functor_t _explicit_instance = [](int64_t _tid) -> int64_t {
            auto proc_itr = get_affinity_map().find(_tid);
            if(proc_itr != get_affinity_map().end())
                return proc_itr->second;
            return -1;
        };
        //
        //  checks the configured mode and applies the appropriate algorithm
        //
        switch(get_mode())
        {
            case COMPACT: return _compact_instance(tid);
            case SCATTER:
            case SPREAD: return _scatter_instance(tid);
            case EXPLICIT: return _explicit_instance(tid);
        };
        //
        // default to compact algorithm
        //
        return _compact_instance(tid);
    };
    //
    return _instance;
}

TIMEMORY_PROCESS_INLINE
int64_t
affinity::set()
{
#if defined(TIMEMORY_LINUX)
    auto proc_id = get_algorithm()(get_id());
    if(proc_id >= 0)
    {
        cpu_set_t cpuset;
        pthread_t thread = pthread_self();
        CPU_ZERO(&cpuset);
        CPU_SET(proc_id, &cpuset);
        pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    }
    return proc_id;
#else
    return -1;
#endif
}

TIMEMORY_PROCESS_INLINE
int64_t
affinity::set(native_handle_t athread)
{
#if defined(TIMEMORY_LINUX)
    auto      proc_id = get_algorithm()(get_id());
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(proc_id, &cpuset);
    pthread_setaffinity_np(athread, sizeof(cpu_set_t), &cpuset);
    return proc_id;
#else
    consume_parameters(athread);
    return -1;
#endif
}
}  // namespace threading
}  // namespace tim

#endif
