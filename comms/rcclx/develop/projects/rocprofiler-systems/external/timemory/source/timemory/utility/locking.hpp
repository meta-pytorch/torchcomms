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

#include "timemory/api.hpp"
#include "timemory/macros/attributes.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <system_error>
#include <thread>
#include <type_traits>

namespace tim
{
inline namespace locking
{
template <typename MutexT>
using auto_lock = std::unique_lock<MutexT>;

/// \typedef std::recursive_mutex mutex_t
/// \brief Recursive mutex is used for convenience since the
/// performance penalty vs. a regular mutex is not really an issue since there are not
/// many locks in general.
using mutex_t = std::recursive_mutex;

/// \typedef std::unique_lock<std::recursive_mutex> auto_lock_t
/// \brief Unique lock type around \ref mutex_t
using auto_lock_t = auto_lock<mutex_t>;

/// \fn mutex_t& type_mutex(uint64_t)
/// \tparam Tp data type for lock
/// \tparam ApiT API for lock
/// \tparam N max size
/// \tparam MutexT mutex data type
///
/// \brief A simple way to get a mutex for a class or common behavior, e.g.
/// `type_mutex<decltype(std::cout)>()` provides a mutex for synchronizing output streams.
/// Recommend using in conjunction with auto-lock:
/// `tim::auto_lock_t _lk{ type_mutex<Foo>() }`.
template <typename Tp, typename ApiT = TIMEMORY_API, size_t N = 4,
          typename MutexT = mutex_t>
inline MutexT&
type_mutex(uint64_t _n = 0) TIMEMORY_VISIBILITY("default");

template <typename Tp, typename ApiT, size_t N, typename MutexT>
MutexT&
type_mutex(uint64_t _n)
{
    static std::array<MutexT, N> _mutexes{};
    return _mutexes.at(_n % N);
}

template <typename Tp, typename ApiT = TIMEMORY_API, size_t N = 4, typename MutexT>
inline MutexT&
type_mutex(type_list<MutexT>, uint64_t _n = 0)
{
    return type_mutex<Tp, ApiT, N, MutexT>(_n);
}

/// helper function to prevent deadlocks, default parameters will poll for 10 milliseconds
/// 100 times (i.e. 1 second)
/// \param _lk std::unique_lock instance
/// \param _duration time to wait before trying to acquire lock again
/// \return Whether or not the lock was successfully acquired
template <int32_t N        = 100, typename MutexT,
          typename RepT    = typename std::chrono::milliseconds::rep,
          typename PeriodT = typename std::chrono::milliseconds::period>
inline bool
try_lock_for_n(std::unique_lock<MutexT>&                   _lk,
               const std::chrono::duration<RepT, PeriodT>& _duration =
                   std::chrono::milliseconds{ 10 },
               std::integral_constant<int32_t, N> = {})
{
    if(_lk.owns_lock())
        return true;

    try
    {
        if(N == 0)
        {
            while(!_lk.try_lock())
                std::this_thread::sleep_for(_duration);
            return _lk.owns_lock();
        }
        else
        {
            for(int32_t i = 0; i < N; ++i)
            {
                if(_lk.try_lock())
                    return true;
                std::this_thread::sleep_for(_duration);
            }
        }
    } catch(const std::system_error& _e)
    {
        fprintf(stderr, "[timemory][%s@%s:%i] %s :: %s\n", __FUNCTION__, __FILE__,
                __LINE__, _e.what(), _e.code().message().c_str());
    }
    return _lk.owns_lock();
}

///
/// simple mutex which spins on an atomic while trying to lock.
/// Provided for internal use for when there is low contention
/// but we want to avoid using pthread mutexes since those
/// are wrapped by library
///
struct spin_mutex
{
    spin_mutex()  = default;
    ~spin_mutex() = default;

    spin_mutex(const spin_mutex&)     = delete;
    spin_mutex(spin_mutex&&) noexcept = delete;

    spin_mutex& operator=(const spin_mutex&) = delete;
    spin_mutex& operator=(spin_mutex&&) noexcept = delete;

    void lock();
    void unlock();
    bool try_lock();

private:
    bool             m_holds = false;
    std::atomic_flag m_value = ATOMIC_FLAG_INIT;
};

inline void
spin_mutex::lock()
{
    while(!try_lock())
    {
        std::this_thread::yield();
    }
}

inline void
spin_mutex::unlock()
{
    if(m_holds)
        m_value.clear(std::memory_order_release);
}

inline bool
spin_mutex::try_lock()
{
    auto _result = !m_value.test_and_set(std::memory_order_acquire);
    if(_result)
        m_holds = true;
    return _result;
}

///
/// RAII wrapper for spin_mutex
///
struct spin_lock
{
    spin_lock(spin_mutex&);
    spin_lock(spin_mutex&, std::defer_lock_t);
    ~spin_lock();

    spin_lock(const spin_lock&)     = delete;
    spin_lock(spin_lock&&) noexcept = delete;

    spin_lock& operator=(const spin_lock&) = delete;
    spin_lock& operator=(spin_lock&&) noexcept = delete;

    bool owns_lock() const;

    void lock();
    void unlock();
    bool try_lock();

private:
    bool        m_owns = false;
    spin_mutex& m_mutex;
};

inline spin_lock::spin_lock(spin_mutex& _v)
: m_mutex{ _v }
{
    lock();
}

inline spin_lock::spin_lock(spin_mutex& _v, std::defer_lock_t)
: m_mutex{ _v }
{}

inline spin_lock::~spin_lock() { unlock(); }

inline bool
spin_lock::owns_lock() const
{
    return m_owns;
}

inline void
spin_lock::lock()
{
    if(!owns_lock())
    {
        m_mutex.lock();
        m_owns = true;
    }
}

inline void
spin_lock::unlock()
{
    if(owns_lock())
    {
        m_mutex.unlock();
        m_owns = false;
    }
}

inline bool
spin_lock::try_lock()
{
    if(!owns_lock())
    {
        m_owns = m_mutex.try_lock();
    }
    return m_owns;
}
}  // namespace locking
}  // namespace tim
