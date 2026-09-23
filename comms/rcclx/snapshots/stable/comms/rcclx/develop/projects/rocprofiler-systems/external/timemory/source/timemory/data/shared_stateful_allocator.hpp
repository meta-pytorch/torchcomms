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

#include "timemory/utility/locking.hpp"

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
/// \class tim::data::shared_stateful_allocator
/// \tparam AllocT The allocator type
///
/// \brief Ensures that allocators which have state are not deleted until
/// all handles to the allocator have been released.
template <typename AllocT>
struct shared_stateful_allocator
{
    using this_type = shared_stateful_allocator<AllocT>;
    using pointer   = std::shared_ptr<AllocT>;

    // call this function to get an allocator
    static auto request()
    {
        auto_lock_t _lk{ type_mutex<this_type>() };
        return data()->request_impl();
    }

    // call this function when the allocator should be
    // evaluated for freeing up it's memory
    static auto release(pointer& _v)
    {
        auto_lock_t _lk{ type_mutex<this_type>() };
        return data()->release_impl(_v);
    }

private:
    using alloctor_array_t = std::vector<pointer>;

    pointer request_impl()
    {
        m_data.emplace_back(std::make_shared<AllocT>());
        return m_data.back();
    }

    int64_t release_impl(pointer& _v)
    {
        if(m_data.empty())
            return 0;
        auto itr = m_data.begin();
        for(; itr != m_data.end(); ++itr)
        {
            if(itr->get() == _v.get())
                break;
        }
        if(itr == m_data.end())
            return 0;
        auto _count = itr->use_count();
        // only instances are held by m_data and _v
        if(_count == 2)
            itr->reset();
        return (_count < 2) ? 0 : _count - 2;
    }

    alloctor_array_t m_data = {};

    static this_type* data()
    {
        static auto* _v = new this_type{};
        return _v;
    }
};

}  // namespace data
}  // namespace tim
