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

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/data/ring_buffer.hpp"
#include "timemory/defines.h"
#include "timemory/log/logger.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/signals/signal_mask.hpp"
#include "timemory/utility/locking.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <semaphore.h>
#include <sstream>
#include <thread>
#include <vector>

#define TIMEMORY_SEMAPHORE_CHECK(VAL)                                                    \
    {                                                                                    \
        if(VAL != 0)                                                                     \
        {                                                                                \
            perror(#VAL);                                                                \
            throw std::runtime_error(#VAL);                                              \
        }                                                                                \
    }

#define TIMEMORY_SEMAPHORE_CHECK_MSG(VAL, ...)                                           \
    {                                                                                    \
        if(VAL != 0)                                                                     \
        {                                                                                \
            perror(__VA_ARGS__);                                                         \
            throw std::runtime_error(__VA_ARGS__);                                       \
        }                                                                                \
    }

#define TIMEMORY_SEMAPHORE_HANDLE_EINTR(FUNC, EC, ...)                                   \
    {                                                                                    \
        EC = FUNC(__VA_ARGS__);                                                          \
        if(EC != 0)                                                                      \
            EC = errno;                                                                  \
        while(EC == EINTR)                                                               \
        {                                                                                \
            EC = FUNC(__VA_ARGS__);                                                      \
            if(EC != 0)                                                                  \
                EC = errno;                                                              \
        }                                                                                \
    }

#define TIMEMORY_SEMAPHORE_TRYWAIT(SEM, EC)                                              \
    TIMEMORY_SEMAPHORE_HANDLE_EINTR(sem_trywait, EC, &SEM)

namespace tim
{
namespace sampling
{
using semaphore_t = sem_t;

/// \struct tim::sampling::allocator
/// \tparam Tp The tim::sampling::sampler template type
///
/// \brief This structure handles allocating new memory outside of the signal handler
/// for \ref tim::sampling::sampler when it is dynamic. The sampler signals when
/// it's buffer is full and the thread owned by this class wakes up, swaps out
/// the full buffer for an empty one and then moves the data into it's memory
template <typename Tp>
struct allocator
{
    using sampler_type      = Tp;
    using this_type         = allocator<Tp>;
    using data_type         = typename Tp::bundle_type;
    using buffer_type       = data_storage::ring_buffer<data_type>;
    using lock_type         = std::unique_lock<std::mutex>;
    using offload_func_type = void (*)(int64_t, buffer_type&&);

    struct sampler_data
    {
        explicit sampler_data(int64_t _tid_v)
        : tid{ _tid_v }
        {}

        int64_t                  tid = -1;
        std::mutex               buffer_mutex;
        std::vector<data_type>   data    = {};
        std::vector<buffer_type> buffers = {};
    };

    allocator();
    ~allocator();

    allocator(const allocator&)     = delete;
    allocator(allocator&&) noexcept = default;
    allocator& operator=(const allocator&) = delete;
    allocator& operator=(allocator&&) noexcept = default;

    /// Set a callback function for handling non-empty memory buffers.
    /// This is useful if you want to manually manage the memory, e.g.
    /// write the data to a file.
    /// When set, this function gets invoked when:
    /// - a memory buffer is full
    /// - the sampler holding the allocator instance is stopped
    /// - the \ref get_data() function is called
    /// Invoking \ref get_data() within the offload function will cause a deadlock
    auto set_offload(offload_func_type _v) { return m_offload.exchange(_v); }

    /// This function will return any data stored internally (i.e. not offloaded)
    auto get_data(const Tp*) const;

    bool is_alive() const { return m_alive; }

    /// waits for the allocator's thread to finish
    void join();

    /// blocks the specified signal number from being delievered
    /// to the allocator's thread
    void block_signal(int _v);

    /// provide data to the allocator. Will either invoke offload callback or store
    /// internally
    void emplace(const Tp*, buffer_type&&);

    /// set the verbosity -- currently unused
    void set_verbose(int _v) { m_verbose = _v; }

    /// reserve this number of buckets for sampler data
    void reserve(size_t _v) { m_data.reserve(_v); }

    /// return the number of allocated sampler data instances
    auto size() const { return m_data.size(); }

    void flush();
    void reset();
    void allocate(Tp*);
    void deallocate(Tp*);

private:
    using atomic_offload_func_t = std::atomic<offload_func_type>;
    using data_map_t    = std::unordered_map<const Tp*, std::shared_ptr<sampler_data>>;
    using flush_func_t  = std::function<void()>;
    using exit_func_t   = std::function<void()>;
    using notify_func_t = std::function<void(bool*)>;
    using move_func_t   = std::function<void(sampler_type*, buffer_type&&)>;

    /// the execution function for the allocator's thread
    static void execute(allocator*);

    static void default_notify(bool*);

    /// this function makes sure that allocator isn't interrupted by signals
    void block_pending_signals();

    void update_data();
    void update_data(const Tp*);

    template <typename FuncT>
    void set_notify(FuncT&& _v);

    template <typename FuncT>
    void set_move(FuncT&& _v);

    template <typename FuncT>
    void set_exit(FuncT&& _v);

    template <typename FuncT>
    void set_flush(FuncT&& _v);

    std::shared_ptr<sampler_data> get(const Tp*) const;

    std::atomic<bool>            m_alive{ false };
    int                          m_verbose = 0;
    semaphore_t                  m_sem;
    atomic_offload_func_t        m_offload = { nullptr };
    mutable std::mutex           m_signal_mutex;
    mutable std::recursive_mutex m_data_mutex;
    std::promise<void>           m_ready         = {};
    std::set<int>      m_block_signals_pending   = { SIGALRM, SIGVTALRM, SIGPROF };
    std::set<int>      m_block_signals_completed = {};
    std::exception_ptr m_thread_exception        = {};
    flush_func_t       m_flush                   = []() {};
    exit_func_t        m_exit                    = []() {};
    notify_func_t      m_notify                  = &default_notify;
    move_func_t        m_move                    = [](Tp*, buffer_type&&) {};
    data_map_t         m_data                    = {};
    std::thread        m_thread;  // construct thread after data to prevent UB
};

template <typename Tp>
allocator<Tp>::allocator()
: m_thread{ &execute, this }
{
    // wait for a max of 1 second for thread to start
    m_ready.get_future().wait_for(std::chrono::seconds{ 1 });
    m_alive.store(true);
}

template <typename Tp>
allocator<Tp>::~allocator()
{
    m_exit();
    join();
    if(sem_destroy(&m_sem) != 0)
        TIMEMORY_PRINTF(stderr, "failed to destroy semaphore in sampling allocator");
}

template <typename Tp>
void
allocator<Tp>::join()
{
    if(m_thread.joinable())
        m_thread.join();
    if(m_thread_exception)
        std::rethrow_exception(m_thread_exception);
}

template <typename Tp>
void
allocator<Tp>::block_signal(int _v)
{
    lock_type _lk{ m_signal_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();

    if(m_block_signals_completed.count(_v) == 0)
    {
        m_block_signals_pending.emplace(_v);
    }
}

template <typename Tp>
void
allocator<Tp>::block_pending_signals()
{
    lock_type _lk{ m_signal_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();

    auto& _pending = m_block_signals_pending;
    if(!_pending.empty())
    {
        auto& _completed = m_block_signals_completed;
        for(const auto& itr : _pending)
        {
            signals::block_signals({ itr }, signals::sigmask_scope::thread);
            _completed.emplace(itr);
        }
        _pending.clear();
    }
}

template <typename Tp>
template <typename FuncT>
void
allocator<Tp>::set_notify(FuncT&& _v)
{
    m_notify = std::forward<FuncT>(_v);
}

template <typename Tp>
template <typename FuncT>
void
allocator<Tp>::set_move(FuncT&& _v)
{
    m_move = std::forward<FuncT>(_v);
}

template <typename Tp>
template <typename FuncT>
void
allocator<Tp>::set_exit(FuncT&& _v)
{
    m_exit = std::forward<FuncT>(_v);
}

template <typename Tp>
template <typename FuncT>
void
allocator<Tp>::set_flush(FuncT&& _v)
{
    m_flush = std::forward<FuncT>(_v);
}

template <typename Tp>
std::shared_ptr<typename allocator<Tp>::sampler_data>
allocator<Tp>::get(const Tp* _v) const
{
    auto_lock_t _data_lk{ m_data_mutex };
    auto        ditr = m_data.find(_v);
    if(ditr == m_data.end())
        throw std::runtime_error("Invalid instance");
    if(!ditr->second)
        throw std::runtime_error("nullptr to allocator sampler_data");
    return ditr->second;
}

template <typename Tp>
void
allocator<Tp>::emplace(const Tp* _inst, buffer_type&& _v)
{
    auto _data = get(_inst);
    if(m_offload)
    {
        (*m_offload)(_data->tid, std::move(_v));
    }
    else
    {
        lock_type _lk{ _data->buffer_mutex };
        _data->buffers.emplace_back(std::move(_v));
    }
}

template <typename Tp>
void
allocator<Tp>::flush()
{
    m_flush();
}

template <typename Tp>
void
allocator<Tp>::reset()
{
    set_move([](Tp*, buffer_type&&) {});
    set_notify(&default_notify);

    auto_lock_t _data_lk{ m_data_mutex };
    for(auto& itr : m_data)
    {
        const_cast<Tp*>(itr.first)->set_notify(m_notify);
        const_cast<Tp*>(itr.first)->set_move(m_move);
    }
}

template <typename Tp>
void
allocator<Tp>::allocate(Tp* _v)
{
    if(!_v)
        return;

    _v->set_notify(m_notify);
    _v->set_move(m_move);

    auto_lock_t _data_lk{ m_data_mutex };
    auto        _tid = _v->m_tid;
    m_data.emplace(_v, std::make_shared<sampler_data>(_tid));
}

template <typename Tp>
void
allocator<Tp>::deallocate(Tp* _v)
{
    if(!_v)
        return;

    _v->set_notify(&default_notify);
    _v->set_move([](Tp*, buffer_type&&) {});

    auto_lock_t _data_lk{ m_data_mutex };
    auto        itr = m_data.find(_v);
    if(itr != m_data.end())
        m_data.erase(itr);
}

template <typename Tp>
void
allocator<Tp>::update_data()
{
    auto_lock_t _data_lk{ m_data_mutex };
    for(const auto& itr : m_data)
        update_data(itr.first);
}

template <typename Tp>
void
allocator<Tp>::update_data(const Tp* _inst)
{
    auto _data = get(_inst);

    // swap out buffer array for a temporary to avoid
    // holding lock while processing
    auto _buffers = std::vector<buffer_type>{};
    {
        lock_type _lk{ _data->buffer_mutex, std::defer_lock };
        if(!_lk.owns_lock())
            _lk.lock();
        std::swap(_data->buffers, _buffers);
    }

    /// if a callback function for handling was the memory buffers was provided,
    /// transfer the data via that callback and return
    if(m_offload)
    {
        for(auto& itr : _buffers)
        {
            if(!itr.is_empty())
                (*m_offload)(_data->tid, std::move(itr));
            else
                itr.destroy();
        }
        return;
    }

    /// if no offload callback function was provided, empty out the
    /// the ring buffer and store for retrieval later (which
    /// is done via the \ref get_data() function)
    size_t _n = 0;
    for(auto& itr : _buffers)
        _n += itr.count();
    _data->data.reserve(_data->data.size() + _n);
    for(auto& itr : _buffers)
    {
        while(!itr.is_empty())
        {
            auto _v = data_type{};
            itr.read(&_v);
            _data->data.emplace_back(std::move(_v));
        }
        itr.destroy();
    }
}

template <typename Tp>
auto
allocator<Tp>::get_data(const Tp* _inst) const
{
    using return_type = std::vector<data_type>;

    auto_lock_t _data_lk{ m_data_mutex };
    const_cast<this_type*>(this)->update_data(_inst);
    auto itr = m_data.find(_inst);
    return (itr == m_data.end() || !itr->second) ? return_type{} : itr->second->data;
}

template <typename Tp>
void
allocator<Tp>::default_notify(bool* _completed)
{
    if(_completed)
        *_completed = true;
}

template <typename Tp>
void
allocator<Tp>::execute(allocator* _alloc)
{
    threading::offset_this_id(true);
    static std::atomic<int64_t> _instance_id{ 0 };
    auto                        _inst_id = _instance_id++;
    threading::set_thread_name(
        std::string{ "samp.alloc" + std::to_string(_inst_id) }.c_str());

    using buffer_vec_t = std::vector<std::pair<const Tp*, buffer_type>>;

    bool          _completed         = false;
    bool          _clean_exit        = true;
    const int64_t _local_buffer_size = 5;
    auto          _local_buffer      = buffer_vec_t{};
    auto          _pending           = std::atomic<int64_t>{ 0 };
    semaphore_t*  _sem               = &_alloc->m_sem;
    auto          _buffer_mutex      = locking::spin_mutex{};

    TIMEMORY_SEMAPHORE_CHECK(sem_init(_sem, 0, 0));
    _local_buffer.reserve(_local_buffer_size);

    auto _swap_buffer = [&](buffer_vec_t& _new_buffer) {
        size_t _n = 0;
        {
            locking::spin_lock _lk{ _buffer_mutex };
            if(_local_buffer.empty())
                return;
            std::swap(_local_buffer, _new_buffer);
            _n = _new_buffer.size();
        }
        _pending -= _n;
        for(auto& itr : _new_buffer)
        {
            if(!itr.second.is_empty())
            {
                _alloc->emplace(itr.first, std::move(itr.second));
            }
        }
    };
    auto _notify = [&](semaphore_t* _sem_v) {
        if(_alloc && _sem_v)
        {
            TIMEMORY_SEMAPHORE_CHECK(sem_post(_sem_v));
        }
    };
    auto _exit = [&]() {
        _completed = true;
        TIMEMORY_SEMAPHORE_CHECK(sem_post(_sem));
    };
    auto _flush = [&]() {
        int _val = 0;
        do
        {
            TIMEMORY_SEMAPHORE_CHECK(sem_getvalue(_sem, &_val));
            if(_val > 0)
            {
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
            }
        } while(_val > 0);
    };

    try
    {
        _alloc->set_notify([&](bool* _v) {
            _notify(_sem);
            if(_v)
                *_v = true;
        });

        _alloc->set_move([&](Tp* _inst, buffer_type&& _buffer) {
            {
                buffer_type        _buffer_v = std::move(_buffer);
                locking::spin_lock _lk{ _buffer_mutex };
                _local_buffer.emplace_back(_inst, std::move(_buffer_v));
                ++_pending;
            }
            _notify(_sem);
        });

        _alloc->set_flush(_flush);

        _alloc->set_exit(_exit);

        _alloc->m_ready.set_value();

        while(!_completed)
        {
            auto _new_buffer = buffer_vec_t{};
            _new_buffer.reserve(_local_buffer_size);

            std::this_thread::yield();
            TIMEMORY_SEMAPHORE_CHECK(sem_wait(_sem));

            _swap_buffer(_new_buffer);

            // block the pending signals
            _alloc->block_pending_signals();
        }
    } catch(...)
    {
        _clean_exit = false;
        // Set the exception pointer in case of an exception
        _alloc->m_thread_exception = std::current_exception();
    }

    if(_clean_exit)
    {
        while(_pending.load() > 0)
        {
            auto _preswap_pending = _pending.load();
            auto _new_buffer      = buffer_vec_t{};
            _new_buffer.reserve(_preswap_pending);
            _swap_buffer(_new_buffer);
            if(_pending.load() == _preswap_pending)
                --_pending;
        }

        auto _new_buffer = buffer_vec_t{};
        _new_buffer.reserve(1);

        _swap_buffer(_new_buffer);

        _alloc->update_data();
    }
    _sem = nullptr;

    // disable communication with allocator
    _alloc->reset();
    _alloc->set_flush([]() {});
    _alloc->m_alive.store(false);
}

template <>
struct allocator<void>
{
    template <typename Tp>
    explicit allocator(Tp*)
    {}

    allocator()                     = default;
    ~allocator()                    = default;
    allocator(const allocator&)     = delete;
    allocator(allocator&&) noexcept = default;
    allocator& operator=(const allocator&) = delete;
    allocator& operator=(allocator&&) noexcept = default;

    static constexpr bool   is_alive() { return false; }
    static constexpr void   block_signal(int) {}
    static constexpr void   join() {}
    static constexpr void   set_verbose(int) {}
    static constexpr void   reserve(size_t) {}
    static constexpr size_t size() { return 0; }
    static constexpr void   flush() {}
    static constexpr void   reset() {}

    template <typename... Args>
    static constexpr void emplace(Args&&...)
    {}

    template <typename Tp>
    static void allocate(Tp*)
    {}

    template <typename Tp>
    static void deallocate(Tp*)
    {}

    template <typename FuncT>
    static decltype(auto) set_offload(FuncT&& _func)
    {
        return std::forward<FuncT>(_func);
    }
};

}  // namespace sampling
}  // namespace tim
