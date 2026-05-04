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

/** \file timemory/ert/barrier.hpp
 * \headerfile timemory/ert/barrier.hpp "timemory/ert/barrier.hpp"
 * Provides multi-threading barriers
 *
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace tim
{
namespace ert
{
using std::size_t;

//--------------------------------------------------------------------------------------//
//  creates a multithreading barrier
//
class thread_barrier
{
public:
    using size_type = int64_t;
    using mutex_t   = std::mutex;
    using condvar_t = std::condition_variable;
    using atomic_t  = std::atomic<size_type>;
    using lock_t    = std::unique_lock<mutex_t>;
    using promise_t = std::promise<void>;
    using future_t  = std::shared_future<void>;

public:
    enum class mode : short
    {
        notify = 0,
        spin,
        cv
    };

    explicit thread_barrier(size_t _nthreads)
    : m_num_threads{ static_cast<size_type>(_nthreads) }
    , m_future{ m_promise.get_future().share() }
    {}

    thread_barrier(const thread_barrier&) = delete;
    thread_barrier(thread_barrier&&)      = delete;

    thread_barrier& operator=(const thread_barrier&) = delete;
    thread_barrier& operator=(thread_barrier&&) = delete;

    size_type size() const { return m_num_threads; }

    // check if this is the thread the created barrier
    bool is_master() const { return std::this_thread::get_id() == m_master; }

    // the generic wait method
    auto wait(mode _mode = mode::notify)
    {
        if(is_master())
            return invoke_error();

        switch(_mode)
        {
            case mode::notify: return notify_wait();
            case mode::spin: return spin_wait();
            case mode::cv: return cv_wait();
        }
    }

    auto get_count(mode _mode = mode::notify)
    {
        switch(_mode)
        {
            case mode::notify: return m_notify.load();
            case mode::spin: return m_counter;
            case mode::cv: return m_counter;
        }
        return size();
    }

private:
    // call from worker thread -- spin wait (fast)
    void spin_wait()
    {
        {
            lock_t _lk{ m_mutex };
            ++m_counter;
            ++m_waiting;
        }

        while(m_counter < m_num_threads)
        {
            while(spin_lock.test_and_set(std::memory_order_acquire))  // acquire lock
            {
            }  // spin
            spin_lock.clear(std::memory_order_release);
        }

        {
            lock_t _lk{ m_mutex };
            --m_waiting;
            if(m_waiting == 0)
                m_counter = 0;  // reset barrier
        }
    }

    // call from worker thread -- condition variable wait (slower)
    void cv_wait()
    {
        lock_t _lk{ m_mutex };
        ++m_counter;
        ++m_waiting;
        m_cv.wait(_lk, [&] { return m_counter >= m_num_threads; });
        m_cv.notify_one();
        --m_waiting;
        if(m_waiting == 0)
            m_counter = 0;  // reset barrier
    }

    // workers increment an atomic until and wait on future until
    // master sets the promise once the
    void notify_wait()
    {
        // make copy of future
        auto      _fut = m_future;
        size_type _id  = ++m_notify;

        if(_id == m_num_threads)
        {
            // swap out the member promise future for next notify_wait() call
            promise_t _promise{};
            future_t  _future = _promise.get_future().share();
            std::swap(_promise, m_promise);
            std::swap(_future, m_future);
            // reset the notify value
            m_notify.store(0);
            // release all the waiting threads
            _promise.set_value();
        }

        _fut.wait();
    }

    void invoke_error()
    {
#if defined(TIMEMORY_INTERNAL_TESTING)
        TIMEMORY_EXCEPTION("master thread calling worker wait function\n");
#endif
    }

    // the constructing thread will be set to master
    std::thread::id  m_master      = std::this_thread::get_id();
    size_type        m_num_threads = 0;  // number of threads that will wait on barrier
    size_type        m_waiting     = 0;  // number of threads waiting on lock
    size_type        m_counter     = 0;  // number of threads that have entered wait func
    std::atomic_flag spin_lock     = ATOMIC_FLAG_INIT;  // for spin lock
    mutex_t          m_mutex;
    condvar_t        m_cv;
    std::atomic<size_type> m_notify{ 0 };
    promise_t              m_promise;
    future_t               m_future;
};

}  // namespace ert
}  // namespace tim
