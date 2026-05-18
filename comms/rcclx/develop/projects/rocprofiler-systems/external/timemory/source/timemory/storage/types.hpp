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

/**
 * \file timemory/storage/types.hpp
 * \brief Declare the storage types
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/data/types.hpp"  // data::ring_buffer_allocator
#include "timemory/hash/declaration.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/storage/macros.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace tim
{
//
class manager;
//
struct settings;
//
struct standalone_storage
{};
//
//--------------------------------------------------------------------------------------//
//
//                              storage
//
//--------------------------------------------------------------------------------------//
//
namespace node
{
//
template <typename Tp>
struct data;
//
template <typename Tp>
struct graph;
//
template <typename Tp>
struct result;
//
template <typename Tp, typename StatT>
struct entry;
//
template <typename Tp>
struct tree;
//
}  // namespace node
//
//--------------------------------------------------------------------------------------//
//
namespace base
{
//
class TIMEMORY_VISIBILITY("default") data_cleanup
{
public:
    data_cleanup()          = default;
    virtual ~data_cleanup() = default;

    virtual void deregister() = 0;
    virtual void destroy()    = 0;
};
//
class TIMEMORY_VISIBILITY("default") storage;
//
}  // namespace base
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
//
template <typename Type, bool ImplementsStorage>
class TIMEMORY_VISIBILITY("default") storage
{};
//
template <typename StorageType>
struct storage_deleter;
//
}  // namespace impl
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp = typename trait::collects_data<Tp>::type>
class TIMEMORY_VISIBILITY("default") storage;
//
template <typename Tp>
using storage_singleton =
    singleton<Tp, std::unique_ptr<Tp, impl::storage_deleter<Tp>>, TIMEMORY_API>;
//
template <typename Tp>
TIMEMORY_NOINLINE storage_singleton<Tp>*
                  get_storage_singleton() TIMEMORY_VISIBILITY("default");
//
template <typename NodeT>
class graph_data;
//
template <typename T>
class tgraph_node;
//
template <typename T, typename AllocatorT = data::ring_buffer_allocator<tgraph_node<T>>>
class graph;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
