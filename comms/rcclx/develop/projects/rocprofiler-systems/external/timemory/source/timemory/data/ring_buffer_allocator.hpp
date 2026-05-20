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

#include "timemory/data/ring_buffer.hpp"
#include "timemory/environment/declaration.hpp"
#include "timemory/units.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace tim
{
namespace data
{
/// \class tim::data::ring_buffer_allocator
/// \tparam Tp The data type for the allocator
/// \tparam MMapV Whether to use mmap (if available)
/// \tparam BuffCntV The default buffer count (will be rounded up to multiple of page
/// size)
///
/// \brief allocator that uses array of (ring) buffers to coalesce memory. Requires
/// This allocator propagates on container swap and container move assignment.
/// Use TIMEMORY_RING_BUFFER_ALLOCATOR_BUFFER_COUNT env variable to specify the default
/// number of allocations or use the `set_buffer_count` / `set_buffer_count_cb`.
/// When a reserve is requested and the request is greater than the free
/// spaces in the buffer, the free spaces are stored in a "dangling" array of spaces
/// which are used when single allocations are requested.
template <typename Tp, bool MMapV, size_t BuffCntV>
class ring_buffer_allocator : public std::allocator<Tp>
{
public:
    // The following will be the same for virtually all allocators.
    using value_type                             = Tp;
    using pointer                                = Tp*;
    using reference                              = Tp&;
    using const_pointer                          = const Tp*;
    using const_reference                        = const Tp&;
    using size_type                              = size_t;
    using difference_type                        = ptrdiff_t;
    using base_type                              = std::allocator<Tp>;
    using buffer_type                            = data_storage::ring_buffer<Tp>;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;

public:
    // constructors and destructors
    ring_buffer_allocator()                                 = default;
    ~ring_buffer_allocator()                                = default;
    ring_buffer_allocator(const ring_buffer_allocator&)     = default;
    ring_buffer_allocator(ring_buffer_allocator&&) noexcept = default;

public:
    // operators
    ring_buffer_allocator& operator=(const ring_buffer_allocator&) = default;
    ring_buffer_allocator& operator=(ring_buffer_allocator&&) noexcept = default;
    bool operator==(const ring_buffer_allocator& rhs) const { return (this == &rhs); }
    bool operator!=(const ring_buffer_allocator& rhs) const { return (this != &rhs); }

public:
    Tp*       address(Tp& _r) const { return &_r; }
    const Tp* address(const Tp& _r) const { return &_r; }

    size_t max_size() const
    {
        // avoid signed/unsigned warnings independent of size_t definition
        return (static_cast<size_t>(0) - static_cast<size_t>(1)) / sizeof(Tp);
    }

    // The following must be the same for all allocators.
    template <typename U>
    struct rebind
    {
        using other = ring_buffer_allocator<U, MMapV, BuffCntV>;
    };

    void construct(Tp* const _p, const Tp& _v) const { ::new((void*) _p) Tp{ _v }; }

    void construct(Tp* const _p, Tp&& _v) const { ::new((void*) _p) Tp{ std::move(_v) }; }

    template <typename... ArgsT>
    void construct(Tp* const _p, ArgsT&&... _args) const
    {
        ::new((void*) _p) Tp{ std::forward<ArgsT>(_args)... };
    }

    void destroy(Tp* const _p) const { _p->~Tp(); }

    Tp* allocate(const size_t n) const
    {
        if(n == 0)
            return nullptr;

        // integer overflow check that throws std::length_error in case of overflow
        if(n > max_size())
        {
            throw std::length_error(
                "ring_buffer_allocator<Tp>::allocate() - Integer overflow.");
        }

        // for a single allocation, try to reuse dangling allocations
        if(n == 1 && !m_buffer_data->dangles.empty())
        {
            Tp* _p = m_buffer_data->dangles.back();
            m_buffer_data->dangles.pop_back();
            return _p;
        }

        // ensure m_buffer_data->current is assigned and a buffer has been created
        init_current(n);

        // when n is greater than the remainder of the ring buffer allocation,
        // add the remainder of the allocation to the dangling allocations and
        // then reinitialize current with n to ensure that the memory is contiguous
        if(n > m_buffer_data->current->free())
        {
            m_buffer_data->dangles.reserve(m_buffer_data->dangles.size() +
                                           m_buffer_data->current->free());
            for(size_t i = 0; i < m_buffer_data->current->free(); ++i)
            {
                auto _req = m_buffer_data->current->request();
                if(_req)
                    break;
                m_buffer_data->dangles.emplace_back(_req);
            }
            // ensure a new buffer is created
            m_buffer_data->current = nullptr;
            // new buffer will have at least n pages of contiguous memory
            init_current(n);
        }

        // get first entry
        auto* _p = m_buffer_data->current->request();

        // ensure that extra remainder is marked as used by ring buffer
        for(size_t i = 1; i < n; ++i)
            m_buffer_data->current->request();

        return _p;
    }

    void deallocate(Tp* const ptr, const size_t n) const
    {
        // reserve if n > 1
        if(n > 1)
            m_buffer_data->dangles.reserve(m_buffer_data->dangles.size() + n);
        // place all "deallocated" pointers in dangling array
        for(size_t i = 0; i < n; ++i)
        {
            Tp* _p = ptr + i;
            m_buffer_data->dangles.emplace_back(_p);
        }
    }

    Tp* allocate(const size_t n, const void* const /* hint */) const
    {
        return allocate(n);
    }

    void reserve(const size_t n) { init_current(n); }

    /// define a callback function for initializing the buffer size. Will throw
    /// if a request for the buffer size has already occured.
    template <typename FuncT>
    static void set_buffer_count_cb(FuncT&& _f)
    {
        if(BuffCntV > 0 || get_config_data().initialized)
            throw std::runtime_error("Error! Buffer size has already been fixed");
        get_config_data().buffer_count_cb = std::forward<FuncT>(_f);
    }

    /// set the minimum number of objects for the ring buffer. Will throw if a request for
    /// the buffer size has already occured.
    static void set_buffer_count(size_t _buff_sz)
    {
        set_buffer_count_cb([_buff_sz]() { return _buff_sz; });
    }

    /// transfers the buffers to another allocator
    void steal_resources(ring_buffer_allocator& rhs)
    {
        m_buffer_data->buffers.reserve(m_buffer_data->buffers.size() +
                                       rhs.m_buffer_this.buffers.size());
        m_buffer_data->dangles.reserve(m_buffer_data->dangles.size() +
                                       rhs.m_buffer_this.dangles.size());
        for(auto& itr : rhs.m_buffer_this.buffers)
            m_buffer_data->buffers.emplace_back(std::move(itr));
        for(auto& itr : rhs.m_buffer_this.dangles)
            m_buffer_data->dangles.emplace_back(itr);
        if(m_buffer_data->current == nullptr)
            m_buffer_data->current = rhs.m_buffer_this.current;
        rhs.m_buffer_data = m_buffer_data;
    }

#if !defined(TIMEMORY_INTERNAL_TESTING)
private:
#endif

    struct config_data
    {
        bool                    initialized     = false;
        size_t                  buffer_count    = 0;
        std::function<size_t()> buffer_count_cb = []() {
            return get_env<size_t>("TIMEMORY_RING_BUFFER_ALLOCATOR_BUFFER_COUNT",
                                   units::get_page_size() / sizeof(Tp));
        };

        size_t get()
        {
            initialized = true;
            return (BuffCntV > 0) ? BuffCntV : buffer_count_cb();
        }
    };

    struct buffer_data
    {
        buffer_type*                              current = nullptr;
        std::vector<std::unique_ptr<buffer_type>> buffers = {};
        std::vector<Tp*>                          dangles = {};
    };

    static config_data& get_config_data()
    {
        static config_data _v{};
        return _v;
    }

    static size_t get_buffer_count()
    {
        // comma operator to set init to true
        static auto _v = get_config_data().get();
        return _v;
    }

    buffer_data* get_buffer_data() const { return m_buffer_data; }

private:
    void init_current(size_t _n) const
    {
        if(m_buffer_data->current == nullptr || m_buffer_data->current->is_full())
        {
            _n         = std::max<size_t>(_n, get_buffer_count());
            auto _uniq = std::make_unique<buffer_type>(_n, MMapV);
            m_buffer_data->buffers.emplace_back(std::move(_uniq));
            m_buffer_data->current = m_buffer_data->buffers.back().get();
        }
    }

    mutable buffer_data  m_buffer_this{};
    mutable buffer_data* m_buffer_data = &m_buffer_this;
};
}  // namespace data
}  // namespace tim
