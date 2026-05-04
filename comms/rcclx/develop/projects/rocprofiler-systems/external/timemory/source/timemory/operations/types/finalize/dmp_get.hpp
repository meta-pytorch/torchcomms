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
 * \file timemory/operations/types/finalize_get.hpp
 * \brief Definition for various functions for finalize_get in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/finalize/get.hpp"

#include <map>
#include <vector>

namespace tim
{
namespace operation
{
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct dmp_get<Type, true>
{
    static constexpr bool value  = true;
    using storage_type           = impl::storage<Type, value>;
    using result_type            = typename storage_type::result_array_t;
    using distrib_type           = typename storage_type::dmp_result_t;
    using result_node            = typename storage_type::result_node;
    using graph_type             = typename storage_type::graph_t;
    using graph_node             = typename storage_type::graph_node;
    using hierarchy_type         = typename storage_type::uintvector_t;
    using get_type               = get<Type, value>;
    using basic_tree_type        = typename get_type::basic_tree_vector_type;
    using basic_tree_vector_type = std::vector<basic_tree_type>;
    using basic_tree_map_type    = std::map<std::string, basic_tree_vector_type>;

    explicit TIMEMORY_COLD dmp_get(storage_type& _storage)
    : m_storage(&_storage)
    {}

    TIMEMORY_COLD distrib_type& operator()(distrib_type&);
    TIMEMORY_COLD basic_tree_vector_type& operator()(basic_tree_vector_type&);
    TIMEMORY_COLD basic_tree_map_type& operator()(basic_tree_map_type&);

    template <typename Archive>
    TIMEMORY_COLD enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
                  operator()(Archive&);

private:
    storage_type* m_storage = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct dmp_get<Type, false>
{
    static constexpr bool value = false;
    using storage_type          = impl::storage<Type, value>;

    dmp_get(storage_type&) {}

    template <typename Tp>
    Tp& operator()(Tp&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename dmp_get<Type, true>::distrib_type&
dmp_get<Type, true>::operator()(distrib_type& results)
{
    if(!m_storage)
        return results;

    auto& data = *m_storage;
    auto  _sz  = results.size();
#if defined(TIMEMORY_USE_UPCXX) && defined(TIMEMORY_USE_MPI)
    results.clear();
    auto _mpi = (mpi::is_initialized()) ? data.mpi_get() : distrib_type{};
    auto _upc = (upc::is_initialized()) ? data.upc_get() : distrib_type{};
    for(auto&& itr : _mpi)
        results.emplace_back(std::move(itr));
    for(auto&& itr : _upc)
        results.emplace_back(std::move(itr));
#elif defined(TIMEMORY_USE_UPCXX)
    if(upc::is_initialized())
    {
        for(auto&& itr : data.upc_get())
            results.emplace_back(std::move(itr));
    }
#elif defined(TIMEMORY_USE_MPI)
    if(mpi::is_initialized())
    {
        for(auto&& itr : data.mpi_get())
            results.emplace_back(std::move(itr));
    }
#endif
    // if none of the above added any data, add the serial results
    if(_sz == results.size())
        results.emplace_back(std::move(data.get()));
    return results;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename dmp_get<Type, true>::basic_tree_vector_type&
dmp_get<Type, true>::operator()(basic_tree_vector_type& bt)
{
    if(!m_storage)
        return bt;

    auto& data  = *m_storage;
    bool  empty = true;

#if defined(TIMEMORY_USE_UPCXX)
    if(upc::is_initialized())
    {
        empty = false;
        data.upc_get(bt);
    }
#endif

#if defined(TIMEMORY_USE_MPI)
    if(mpi::is_initialized())
    {
        empty = false;
        data.mpi_get(bt);
    }
#endif

    // if none of the above added any data, add the serial results
    if(empty)
        data.get(bt);

    return bt;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename dmp_get<Type, true>::basic_tree_map_type&
dmp_get<Type, true>::operator()(basic_tree_map_type& bt)
{
    if(!m_storage)
        return bt;

    auto& data  = *m_storage;
    bool  empty = true;

#if defined(TIMEMORY_USE_UPCXX)
    if(upc::is_initialized())
    {
        empty                        = false;
        basic_tree_vector_type& _val = bt["upcxx"];
        data.upc_get(_val);
    }
#endif

#if defined(TIMEMORY_USE_MPI)
    if(mpi::is_initialized())
    {
        empty                        = false;
        basic_tree_vector_type& _val = bt["mpi"];
        data.mpi_get(_val);
    }
#endif

    // if none of the above added any data, add the serial results
    if(empty)
    {
        basic_tree_vector_type& _val = bt["process"];
        data.get(_val);
    }

    return bt;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
dmp_get<Type, true>::operator()(Archive& ar)
{
    if(!m_storage)
        return ar;

    auto& data  = *m_storage;
    bool  empty = true;

#if defined(TIMEMORY_USE_UPCXX)
    if(upc::is_initialized())
    {
        empty = false;
        data.upc_get(ar);
    }
#endif

#if defined(TIMEMORY_USE_MPI)
    if(mpi::is_initialized())
    {
        empty = false;
        data.mpi_get(ar);
    }
#endif

    // if none of the above added any data, add the serial results
    if(empty)
        data.get(ar);

    return ar;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
