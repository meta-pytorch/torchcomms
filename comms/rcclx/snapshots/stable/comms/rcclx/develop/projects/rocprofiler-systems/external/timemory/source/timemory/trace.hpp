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

/** \file "timemory/trace.hpp"
 * Header file for library tracing operations
 *
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trace
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::trace::trace
/// \brief Prevents recursion with a thread for tracing functions. See also \see
/// tim::trace::lock
///
/// \code{.cpp}
/// tim::trace::lock<tim::trace::trace> lk{};
/// if(!lk)
///     return;
/// \endcode
struct trace
{
    using type                  = std::true_type;
    static constexpr bool value = true;
};
//
/// \struct tim::trace::region
/// \brief Prevents recursion within a thread for region functions. See also \see
/// tim::trace::lock
///
/// \code{.cpp}
/// tim::trace::lock<tim::trace::region> lk{};
/// if(!lk)
///     return;
/// \endcode
struct region
{
    using type                  = std::true_type;
    static constexpr bool value = true;
};
//
/// \struct tim::trace::region
/// \brief Prevents recursion within a thread for library functions. See also \see
/// tim::trace::lock
///
/// \code{.cpp}
/// tim::trace::lock<tim::trace::library> lk{};
/// if(!lk)
///     return;
/// \endcode
struct library
{
    using type                  = std::true_type;
    static constexpr bool value = true;
};
//
/// \struct tim::trace::compiler
/// \brief Prevents recursion within a thread for compiler instrumentation functions. See
/// also \see tim::trace::lock
///
/// \code{.cpp}
/// tim::trace::lock<tim::trace::compiler> lk{};
/// if(!lk)
///     return;
/// \endcode
struct compiler
{
    using type                  = std::true_type;
    static constexpr bool value = true;
};
//
/// \struct tim::trace::region
/// \brief Prevents recursion within a process for threading wrappers. See also \see
/// tim::trace::lock
///
/// \code{.cpp}
/// tim::trace::lock<tim::trace::threading> lk{};
/// if(!lk)
///     return;
/// \endcode
struct threading
{
    using type                  = std::false_type;
    static constexpr bool value = false;
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::trace::lock
/// \brief A lightweight synchronization object for preventing recursion. The first
/// template parameter should have a constexpr boolean indicating whether the lock
/// is thread-local (Tp::value == true) or whether the lock is global (Tp::value == false)
///
/// \code{.cpp}
/// tim::trace::lock<tim::trace::library> lk{};
/// if(!lk)
///     return;
/// \endcode
template <typename Tp, bool ThrLoc = Tp::value>
struct lock;

template <typename Tp>
struct lock<Tp, true>
{
    lock()
    : m_value(!get_global())
    {
        if(m_value)
            get_global() = true;
    }

    ~lock()
    {
        if(m_value)
            get_global() = false;
    }

    lock(lock&&) noexcept = default;
    lock& operator=(lock&&) noexcept = default;

    lock(const lock&) = delete;
    lock& operator=(const lock&) = delete;

    explicit operator bool() const { return m_value; }

    bool& get_local() { return m_value; }

    bool release()
    {
        if(!m_value)
            return false;

        get_global() = false;
        m_value      = false;
        return true;
    }

    bool acquire()
    {
        int64_t itr = 0;
        while(!m_value)
        {
            if((m_value = !get_global()))
                get_global() = true;
            if(itr++ >= std::numeric_limits<int16_t>::max())
                break;
        }
        return m_value;
    }

public:
    static bool& get_global()
    {
        static thread_local bool _instance = false;
        return _instance;
    }

private:
    bool m_value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct lock<Tp, false>
{
    lock()
    : m_value(exchange(true))
    {}

    ~lock()
    {
        if(m_value)
            exchange(false);
    }

    lock(lock&&) noexcept = default;
    lock& operator=(lock&&) noexcept = default;

    lock(const lock&) = delete;
    lock& operator=(const lock&) = delete;

    explicit operator bool() const { return m_value; }

    bool& get_local() { return m_value; }

    bool release()
    {
        if(!m_value)
            return false;

        m_value = exchange(false);
        return true;
    }

    bool acquire()
    {
        int64_t itr = 0;
        if(!m_value)
        {
            while(!(m_value = exchange(true)))
            {
                if(itr++ >= std::numeric_limits<int16_t>::max())
                    break;
            }
        }
        return m_value;
    }

public:
    static auto load() { return get_global().load(std::memory_order_relaxed); }
    static auto exchange(bool _value)
    {
        auto _load = load();
        if(_load == _value)
            return false;
        return get_global().compare_exchange_strong(_load, _value,
                                                    std::memory_order_relaxed);
    }

    static std::atomic<bool>& get_global()
    {
        static std::atomic<bool> _instance{ false };
        return _instance;
    }

private:
    bool m_value;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trace
}  // namespace tim

//--------------------------------------------------------------------------------------//
