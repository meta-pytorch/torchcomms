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

/** \file timemory/ert/aligned_allocator.hpp
 * \headerfile timemory/ert/aligned_allocator.hpp "timemory/ert/aligned_allocator.hpp"
 * Provides an aligned allocator
 *
 */

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/backends/memory.hpp"
#include "timemory/macros/os.hpp"

namespace tim
{
namespace ert
{
//--------------------------------------------------------------------------------------//
//  allocator that can be used with STL containers
//
//  default template parameter AlignV uses (8 * sizeof(Tp)) because alignment should
//  be specified in bits and sizeof(Tp) return bytes so for float (sizeof == 4), this
//  would be 32-bit alignment and for double (sizeof == 8), this would be 64-bit alignment
//
template <typename Tp, std::size_t AlignV = 8 * sizeof(Tp)>
class aligned_allocator
{
public:
    // The following will be the same for virtually all allocators.
    using value_type      = Tp;
    using pointer         = Tp*;
    using reference       = Tp&;
    using const_pointer   = const Tp*;
    using const_reference = const Tp&;
    using size_type       = std::size_t;
    using difference_type = ptrdiff_t;

public:
    // constructors and destructors
    aligned_allocator()                         = default;
    aligned_allocator(const aligned_allocator&) = default;
    aligned_allocator(aligned_allocator&&) noexcept {}
    template <typename U>
    aligned_allocator(const aligned_allocator<U, AlignV>&)
    {}
    ~aligned_allocator() = default;

public:
    // operators
    aligned_allocator& operator=(const aligned_allocator&) = delete;
    aligned_allocator& operator==(aligned_allocator&&)     = delete;
    bool operator!=(const aligned_allocator& other) const { return !(*this == other); }
    bool operator==(const aligned_allocator&) const { return true; }

public:
    TIMEMORY_NODISCARD Tp*   address(Tp& r) const { return &r; }
    TIMEMORY_NODISCARD const Tp* address(const Tp& s) const { return &s; }

    TIMEMORY_NODISCARD std::size_t max_size() const
    {
        // avoid signed/unsigned warnings independent of size_t definition
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(Tp);
    }

    // The following must be the same for all allocators.
    template <typename U>
    struct rebind
    {
        typedef aligned_allocator<U, AlignV> other;
    };

    void construct(Tp* const p, const Tp& t) const
    {
        void* const pv = static_cast<void*>(p);
        new(pv) Tp(t);
    }

    template <typename... ArgsT>
    void construct(Tp* const p, ArgsT&&... args) const
    {
        ::new((void*) p) Tp(std::forward<ArgsT>(args)...);
    }

    void destroy(Tp* const p) const { p->~Tp(); }

    TIMEMORY_NODISCARD Tp* allocate(const std::size_t n) const
    {
        if(n == 0)
            return nullptr;

        // integer overflow check that throws std::length_error in case of overflow
        if(n > max_size())
        {
            throw std::length_error(
                "aligned_allocator<Tp>::allocate() - Integer overflow.");
        }

        // Mallocator wraps malloc().
        void* const ptr = memory::allocate_aligned<Tp, device::cpu>(n, AlignV);

        // throw std::bad_alloc in the case of memory allocation failure.
        if(ptr == nullptr)
        {
            std::cerr << "Allocation of type " << typeid(Tp).name() << " of size " << n
                      << " and alignment " << AlignV << " failed. ptr = " << ptr
                      << std::endl;
            throw std::bad_alloc();
        }

        return static_cast<Tp*>(ptr);
    }

    void deallocate(Tp* const ptr, const std::size_t) const
    {
        memory::free_aligned<Tp, device::cpu>(ptr);
    }

    // same for all allocators that ignore hints.
    template <typename U>
    TIMEMORY_NODISCARD Tp* allocate(const std::size_t n, const U* /* const hint */) const
    {
        return allocate(n);
    }

    // returns the alignment in bytes
    std::size_t get_alignment() { return AlignV / 8; }
};

}  // namespace ert
}  // namespace tim
