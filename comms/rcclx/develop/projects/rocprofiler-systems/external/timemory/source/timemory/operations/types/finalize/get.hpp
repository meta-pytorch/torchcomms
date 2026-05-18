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

#include "timemory/backends/dmp.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/serialization.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/basic_tree.hpp"
#include "timemory/storage/node.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/types.hpp"

#include <string>
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
struct get<Type, true>
{
    static constexpr bool value  = true;
    using storage_type           = impl::storage<Type, value>;
    using result_type            = typename storage_type::result_array_t;
    using distrib_type           = typename storage_type::dmp_result_t;
    using result_node            = typename storage_type::result_node;
    using graph_type             = typename storage_type::graph_t;
    using graph_node             = typename storage_type::graph_node;
    using hierarchy_type         = typename storage_type::uintvector_t;
    using basic_tree_type        = basic_tree<node::tree<Type>>;
    using basic_tree_vector_type = std::vector<basic_tree_type>;
    using metadata               = typename serialization<Type>::metadata;

    TIMEMORY_DEFAULT_OBJECT(get)

    explicit TIMEMORY_COLD get(storage_type& _storage)
    : m_storage(&_storage)
    {}

    explicit TIMEMORY_COLD get(storage_type* _storage)
    : m_storage(_storage)
    {}

    TIMEMORY_COLD result_type& operator()(result_type&);
    TIMEMORY_COLD basic_tree_vector_type& operator()(basic_tree_vector_type&);
    TIMEMORY_COLD std::vector<basic_tree_vector_type>& operator()(
        std::vector<basic_tree_vector_type>& _data)
    {
        basic_tree_vector_type _obj{};
        (*this)(_obj);
        _data.emplace_back(_obj);
        return _data;
    }

    template <typename Archive>
    TIMEMORY_COLD enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
                  operator()(Archive&);

    template <typename Archive>
    TIMEMORY_COLD enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
                  operator()(Archive&, metadata);

public:
    static TIMEMORY_COLD auto get_identifier(const Type& _obj = Type{})
    {
        return serialization<Type>::get_identifier(_obj);
    }
    static TIMEMORY_COLD auto get_label(const Type& _obj = Type{})
    {
        return serialization<Type>::get_label(_obj);
    }
    static TIMEMORY_COLD auto get_description(const Type& _obj = Type{})
    {
        return serialization<Type>::get_description(_obj);
    }
    static TIMEMORY_COLD auto get_unit(const Type& _obj = Type{})
    {
        return serialization<Type>::get_unit(_obj);
    }
    static TIMEMORY_COLD auto get_display_unit(const Type& _obj = Type{})
    {
        return serialization<Type>::get_display_unit(_obj);
    }

private:
    storage_type* m_storage = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct get<Type, false>
{
    static constexpr bool value = false;
    using storage_type          = impl::storage<Type, value>;

    get(storage_type&) {}

    template <typename Tp>
    Tp& operator()(Tp&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename get<Type, true>::result_type&
get<Type, true>::operator()(result_type& ret)
{
    if(!m_storage)
        return ret;

    auto& data               = *m_storage;
    bool  _thread_scope_only = trait::thread_scope_only<Type>::value;
    bool  _use_tid_prefix    = (!settings::collapse_threads() || _thread_scope_only);
    bool  _use_pid_prefix    = (!settings::collapse_processes());
    auto  _num_thr_count     = manager::get_thread_count();
    auto  _num_pid_count     = dmp::size();

    data.m_node_init = dmp::is_initialized();
    data.m_node_rank = dmp::rank();
    data.m_node_size = dmp::size();

    //------------------------------------------------------------------------------//
    //
    //  Compute the thread prefix
    //
    //------------------------------------------------------------------------------//
    auto _get_thread_prefix = [&](const graph_node& itr) {
        if(!_use_tid_prefix || itr.tid() == std::numeric_limits<uint16_t>::max())
            return std::string(">>> ");

        // prefix spacing
        static uint16_t width = 1;
        if(_num_thr_count > 9)
            width = std::max(width, (uint16_t)(log10(_num_thr_count) + 1));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << itr.tid() << ">>> ";
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //
    //  Compute the node prefix
    //
    //------------------------------------------------------------------------------//
    auto _get_node_prefix = [&](const graph_node& itr) {
        if(!data.m_node_init || !_use_pid_prefix)
            return _get_thread_prefix(itr);

        auto _nc    = settings::node_count();  // node-count
        auto _idx   = data.m_node_rank;
        auto _range = std::make_pair(-1, -1);

        if(_nc > 0 && _nc < data.m_node_size)
        {
            // calculate some size parameters and generate map of the pids to node ids
            int32_t nmod  = _num_pid_count % _nc;
            int32_t bins  = _num_pid_count / _nc + ((nmod == 0) ? 0 : 1);
            int32_t bsize = _num_pid_count / bins;
            int32_t ncnt  = 0;  // current count
            int32_t midx  = 0;  // current bin map index
            std::map<int32_t, std::set<int32_t>> binmap;
            for(int32_t i = 0; i < _num_pid_count; ++i)
            {
                binmap[midx].insert(i);
                // check to see if we reached the bin size
                if(++ncnt == bsize)
                {
                    // set counter to zero and advance the node
                    ncnt = 0;
                    ++midx;
                }
            }

            // loop over the bins
            for(const auto& bitr : binmap)
            {
                // if rank is found in a bin, assing range to first and last entry
                if(bitr.second.find(_idx) != bitr.second.end())
                {
                    auto vitr    = bitr.second.begin();
                    _range.first = *vitr;
                    vitr         = bitr.second.end();
                    --vitr;
                    _range.second = *vitr;
                }
            }

            if(settings::debug())
            {
                std::stringstream ss;
                for(const auto& bitr : binmap)
                {
                    ss << ", [" << bitr.first << "] ";
                    std::stringstream bss;
                    for(const auto& nitr : bitr.second)
                        bss << ", " << nitr;
                    ss << bss.str().substr(2);
                }
                std::string _msg = "Intervals: ";
                if(ss.str().length() > 2)
                    _msg += ss.str().substr(2);
                TIMEMORY_PRINT_HERE("[%s][pid=%i][tid=%i]> %s. range = { %i, %i }",
                                    demangle<get<Type, true>>().c_str(),
                                    (int) process::get_id(), (int) threading::get_id(),
                                    _msg.c_str(), (int) _range.first,
                                    (int) _range.second);
            }
        }

        // prefix spacing
        static uint16_t width = 1;
        if(_num_pid_count > 9)
            width = std::max(width, (uint16_t)(log10(_num_pid_count) + 1));
        std::stringstream ss;
        ss.fill('0');
        if(_range.first >= 0 && _range.second >= 0)
        {
            ss << "|" << std::setw(width) << _range.first << ":" << std::setw(width)
               << _range.second << _get_thread_prefix(itr);
        }
        else
        {
            ss << "|" << std::setw(width) << _idx << _get_thread_prefix(itr);
        }
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //
    //  Compute the indentation
    //
    //------------------------------------------------------------------------------//
    // fix up the prefix based on the actual depth
    auto _compute_modified_prefix = [&](const graph_node& itr) {
        std::string _prefix      = data.get_prefix(itr);
        std::string _indent      = {};
        std::string _node_prefix = _get_node_prefix(itr);

        int64_t _depth = itr.depth() - 1;
        if(_depth > 0)
        {
            for(int64_t ii = 0; ii < _depth - 1; ++ii)
                _indent += "  ";
            _indent += "|_";
        }

        return _node_prefix + _indent + _prefix;
    };

    // convert graph to a vector
    auto convert_graph = [&]() {
        result_type _list{};
        {
            // the head node should always be ignored
            int64_t _min   = std::numeric_limits<int64_t>::max();
            auto&   _graph = data.graph();
            for(auto itr = _graph.begin(); itr != _graph.end(); ++itr)
            {
                if(!itr)
                {
                    TIMEMORY_PRINT_HERE("[%s] Warning! Invalid iterator!",
                                        demangle<Type>().c_str());
                    continue;
                }
                _min = std::min<int64_t>(_min, itr->depth());
            }

            for(auto itr = _graph.begin(); itr != _graph.end(); ++itr)
            {
                if(!itr)
                {
                    TIMEMORY_PRINT_HERE("[%s] Warning! Invalid iterator!",
                                        demangle<Type>().c_str());
                    continue;
                }
                // skip if invalid
                if(operation::get_is_invalid<>{}(itr->data()))
                    continue;
                if(itr->depth() > _min)
                {
                    auto _depth     = itr->depth() - (_min + 1);
                    auto _prefix    = _compute_modified_prefix(*itr);
                    auto _rolling   = itr->id();
                    auto _stats     = itr->stats();
                    auto _parent    = graph_type::parent(itr);
                    auto _hierarchy = hierarchy_type{};
                    auto _tid       = itr->tid();
                    auto _pid       = itr->pid();
                    while(_parent && _parent->depth() > _min)
                    {
                        // ensure parent is always updated regardless of branch taken
                        scope::destructor _dtor{ [&_parent]() {
                            _parent = graph_type::parent(_parent);
                        } };
                        if(operation::get_is_invalid<>{}(_parent->data()))
                            continue;
                        _hierarchy.push_back(_parent->id());
                        _rolling += _parent->id();
                    }
                    if(_hierarchy.size() > 1)
                        std::reverse(_hierarchy.begin(), _hierarchy.end());
                    _hierarchy.push_back(itr->id());
                    auto _entry = result_node(itr->id(), itr->obj(), _prefix, _depth,
                                              _rolling, _hierarchy, _stats, _tid, _pid);
                    _list.push_back(std::move(_entry));
                }
            }
        }

        // if collapse is disabled or thread-scope only, there is nothing to merge
        if(!settings::collapse_threads() || _thread_scope_only)
            return _list;

        result_type _combined{};
        operation::finalize::merge<Type, true>(_combined, _list);
        return _combined;
    };

    ret = convert_graph();
    return ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename get<Type, true>::basic_tree_vector_type&
get<Type, true>::operator()(basic_tree_vector_type& bt)
{
    using sibling_iterator = typename graph_type::sibling_iterator;

    if(!m_storage)
        return bt;

    auto& data = *m_storage;
    auto& t    = data.graph();
    for(sibling_iterator itr = t.begin(); itr != t.end(); ++itr)
        bt.push_back(basic_tree_type{}(t, itr));

    bt = merge<Type, true>{}(bt);
    return bt;
    // return (trait::thread_scope_only<Type>::value || !settings::collapse_threads())
    //           ? bt
    //           : merge<Type, true>{}(bt);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
get<Type, true>::operator()(Archive& ar, metadata)
{
    serialization<Type>{}(ar, metadata{});
    return ar;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
enable_if_t<concepts::is_output_archive<Archive>::value, Archive&>
get<Type, true>::operator()(Archive& ar)
{
    if(!m_storage)
        return ar;

    m_storage->m_node_init = dmp::is_initialized();
    m_storage->m_node_rank = dmp::rank();
    m_storage->m_node_size = dmp::size();
    m_storage->merge();

    auto bt = basic_tree_vector_type{};
    serialization<Type>{}(ar, bt);
    return ar;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
