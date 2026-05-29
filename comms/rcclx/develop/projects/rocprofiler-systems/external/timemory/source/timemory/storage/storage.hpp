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

#include "timemory/mpl/type_traits.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/singleton.hpp"

#include <memory>
#include <type_traits>

namespace tim
{
/// \class tim::storage<Tp, Vp>
/// \tparam Tp Component type
/// \tparam Vp Component intermediate value type
///
/// \brief Responsible for maintaining the call-stack storage in timemory. This class
/// and the serialization library are responsible for most of the timemory compilation
/// time.
template <typename Tp, typename Vp>
class storage final : public impl::storage<Tp, trait::uses_value_storage<Tp, Vp>::value>
{
public:
    static constexpr bool uses_value_storage_v = trait::uses_value_storage<Tp, Vp>::value;
    using this_type                            = storage<Tp, Vp>;
    using base_type                            = impl::storage<Tp, uses_value_storage_v>;
    using deleter_t                            = impl::storage_deleter<base_type>;
    using smart_pointer                        = std::unique_ptr<base_type, deleter_t>;
    using singleton_t                          = singleton<base_type, smart_pointer>;
    using pointer                              = typename singleton_t::pointer;
    using auto_lock_t                          = typename singleton_t::auto_lock_t;
    using iterator                             = typename base_type::iterator;
    using const_iterator                       = typename base_type::const_iterator;

    friend struct impl::storage_deleter<this_type>;
    friend class manager;

    using base_type::base_type;
    /// get the pointer to the storage on the current thread. Will initialize instance if
    /// one does not exist.
    using base_type::instance;
    /// get the pointer to the storage on the primary thread. Will initialize instance if
    /// one does not exist.
    using base_type::master_instance;
    /// get the pointer to the storage on the current thread w/o initializing if one does
    /// not exist
    using base_type::noninit_instance;
    /// get the pointer to the storage on the primary thread w/o initializing if one does
    /// not exist
    using base_type::noninit_master_instance;
    /// returns whether storage is finalizing on the primary thread
    using base_type::master_is_finalizing;
    /// returns whether storage is finalizing on the current thread
    using base_type::worker_is_finalizing;
    /// returns whether storage is finalizing on any thread
    using base_type::is_finalizing;
    /// reset the storage data
    using base_type::reset;
    /// returns whether any data has been stored
    using base_type::empty;
    /// get the current estimated number of nodes
    using base_type::size;
    /// inspect the graph and get the true number of nodes
    using base_type::true_size;
    /// get the depth of the last node which pushed to hierarchical storage. Nodes which
    /// used \ref tim::scope::flat or have \ref tim::trait::flat_storage type-trait
    /// set to true will not affect this value
    using base_type::depth;
    /// drop the current node depth and set the current node to it's parent
    using base_type::pop;
    /// insert a new node
    using base_type::insert;
    /// add a component to the stack which can be flushed if the merging or output is
    /// requested/required
    using base_type::stack_pop;
    /// remove component from the stack that will be flushed if the merging or output is
    /// requested/required
    using base_type::stack_push;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
class storage<Tp, type_list<>>
: public storage<Tp, std::conditional_t<trait::is_available<Tp>::value,
                                        typename Tp::value_type, void>>
{
public:
    using Vp =
        std::conditional_t<trait::is_available<Tp>::value, typename Tp::value_type, void>;
    static constexpr bool uses_value_storage_v = trait::uses_value_storage<Tp, Vp>::value;
    using this_type                            = storage<Tp, Vp>;
    using base_type                            = impl::storage<Tp, uses_value_storage_v>;
    using deleter_t                            = impl::storage_deleter<base_type>;
    using smart_pointer                        = std::unique_ptr<base_type, deleter_t>;
    using singleton_t                          = singleton<base_type, smart_pointer>;
    using pointer                              = typename singleton_t::pointer;
    using auto_lock_t                          = typename singleton_t::auto_lock_t;
    using iterator                             = typename base_type::iterator;
    using const_iterator                       = typename base_type::const_iterator;

    friend struct impl::storage_deleter<this_type>;
    friend class manager;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
