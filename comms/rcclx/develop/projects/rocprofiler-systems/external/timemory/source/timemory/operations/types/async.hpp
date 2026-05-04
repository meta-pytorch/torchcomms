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

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

#include <tuple>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct async
{
    using type      = Tp;
    using task_type = std::packaged_task<void()>;
    using this_type = async<Tp>;
    using lock_type = std::unique_lock<std::mutex>;

    explicit async(type& obj)
    : m_data{ &obj }
    , m_thread{ execute_thread, this }
    {}

    ~async()
    {
        {
            m_pool_state.store(false);
            lock_type _task_lock{ m_task_lock, std::defer_lock };
            if(!_task_lock.owns_lock())
                _task_lock.lock();
            m_task_cond.notify_one();
        }
        m_thread.join();
    }

    async(const async&) = delete;
    async& operator=(const async&) = delete;

    async(async&&) = default;
    async& operator=(async&&) = default;

    template <typename FuncT, typename... Args>
    auto operator()(FuncT&& func, Args... args)
    {
        lock_type _task_lock{ m_task_lock };
        enqueue(std::forward<FuncT>(func), 0, std::move(args)...);
        ++m_task_size;
        m_task_cond.notify_one();
    }

    void wait()
    {
        std::deque<std::future<void>> _wait{};
        {
            lock_type _task_lock{ m_task_lock, std::defer_lock };
            std::swap(m_task_wait, _wait);
            if(!_task_lock.owns_lock())
                _task_lock.lock();
            m_task_cond.notify_all();
        }
        for(auto& itr : _wait)
            itr.wait();
    }

private:
    static void execute_thread(this_type* _this)
    {
        while(_this->m_pool_state.load() || (_this->m_task_size.load() == 0))
        {
            // synchronization
            lock_type _task_lock(_this->m_task_lock, std::defer_lock);

            auto leave_pool = [&]() {
                return !_this->m_pool_state.load() && (_this->m_task_size.load() == 0);
            };

            while(_this->m_task_size.load() == 0 && _this->m_pool_state.load())
            {
                // lock before sleeping on condition
                if(!_task_lock.owns_lock())
                    _task_lock.lock();

                // break out of loop
                if(_this->m_task_size.load() > 0)
                    break;

                // return from function
                if(!_this->m_pool_state.load())
                    return;

                // Wait until there is a task in the queue
                // Unlocks mutex while waiting, then locks it back when signaled
                // use lambda to control waking
                _this->m_task_cond.wait(_task_lock, [&]() {
                    return (_this->m_task_size.load() > 0 || !_this->m_pool_state.load());
                });
            }

            // leave pool if conditions dictate it
            if(leave_pool())
                return;

            // acquire the lock
            if(!_task_lock.owns_lock())
                _task_lock.lock();

            std::packaged_task<void()> _task{};
            if(!_this->m_task_pool.empty() && _this->m_task_size.load() > 0)
            {
                --(_this->m_task_size);
                _task = std::move(_this->m_task_pool.front());
                _this->m_task_pool.pop_front();
            }

            // release the lock
            if(_task_lock.owns_lock())
                _task_lock.unlock();

            // execute the task(s)
            if(_task.valid())
            {
                _task();
            }
        }
    }

    template <typename FuncT, typename... Args>
    auto enqueue(FuncT&& func, int, Args... args)
        -> decltype(func(std::declval<type&>(), args...), void())
    {
        auto&& _fut =
            m_task_pool
                .emplace(m_task_pool.end(),
                         [=]() { std::move(func)(*m_data, std::move(args)...); })
                ->get_future();

        m_task_wait.emplace_back(std::move(_fut));
    }

    template <typename FuncT, typename... Args>
    auto enqueue(FuncT&& func, long, Args... args)
        -> decltype(func(std::declval<type*>(), args...), void())
    {
        auto&& _fut = m_task_pool
                          .emplace(m_task_pool.end(),
                                   [=]() { std::move(func)(m_data, std::move(args)...); })
                          ->get_future();

        m_task_wait.emplace_back(std::move(_fut));
    }

    template <typename FuncT, typename... Args>
    auto enqueue(FuncT&& func, long long, Args... args) -> decltype(func(args...), void())
    {
        auto&& _fut = m_task_pool
                          .emplace(m_task_pool.end(),
                                   [=]() { std::move(func)(std::move(args)...); })
                          ->get_future();
        m_task_wait.emplace_back(std::move(_fut));
    }

private:
    type*                                  m_data = nullptr;
    std::atomic<bool>                      m_pool_state{ true };
    std::atomic<size_t>                    m_task_size{ 0 };
    std::mutex                             m_task_lock{};
    std::condition_variable                m_task_cond{};
    std::deque<std::packaged_task<void()>> m_task_pool = {};
    std::deque<std::future<void>>          m_task_wait = {};
    std::thread                            m_thread;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
