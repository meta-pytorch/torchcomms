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

#include "timemory/components/base/types.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/storage/graph.hpp"
#include "timemory/storage/types.hpp"

namespace tim
{
namespace component
{
/// \struct base_iterator
/// \brief Handles the storage iterator of a component
///
template <typename Tp>
struct base_iterator
{
    using iterator_type = typename graph<node::graph<Tp>>::iterator;

    TIMEMORY_DEFAULT_OBJECT(base_iterator)

    TIMEMORY_INLINE auto get_iterator() const { return graph_itr; }
    TIMEMORY_INLINE void set_iterator(iterator_type itr) { graph_itr = itr; }

protected:
    iterator_type graph_itr = iterator_type{ nullptr };
};
//
template <>
struct base_iterator<null_type>
{
    using iterator_type = void*;

    TIMEMORY_DEFAULT_OBJECT(base_iterator)

    TIMEMORY_INLINE static auto get_iterator() { return nullptr; }
    TIMEMORY_INLINE void        set_iterator(iterator_type) {}
};
}  // namespace component
}  // namespace tim
