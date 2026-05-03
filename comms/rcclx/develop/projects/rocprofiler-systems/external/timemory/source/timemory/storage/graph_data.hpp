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

//--------------------------------------------------------------------------------------//

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/graph.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/demangle.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

//--------------------------------------------------------------------------------------//
//
namespace tim
{
//--------------------------------------------------------------------------------------//
//
//  graph instance + current node + head node
//
//--------------------------------------------------------------------------------------//
/// \class tim::graph_data
/// \brief \ref tim::graph instance + current node + head note + sea-level. Sea-level is
/// defined as the node depth after a fork from another graph instance and is only
/// relevant for worker-threads)

template <typename NodeT>
class graph_data
{
public:
    using this_type          = graph_data<NodeT>;
    using graph_t            = tim::graph<NodeT>;
    using iterator           = typename graph_t::iterator;
    using const_iterator     = typename graph_t::const_iterator;
    using inverse_insert_t   = std::vector<std::pair<int64_t, iterator>>;
    using pre_order_iterator = typename graph_t::pre_order_iterator;
    using sibling_iterator   = typename graph_t::sibling_iterator;

public:
    // graph_data() = default;

    explicit graph_data(const NodeT& rhs, int64_t _depth, graph_data* _master = nullptr)
    : m_has_head(true)
    , m_depth(_depth)
    , m_sea_level(_depth)
    , m_master(_master)
    {
        m_head    = m_graph.set_head(rhs);
        m_current = m_head;
        m_dummies.insert({ m_depth, m_current });
    }

    ~graph_data() { m_graph.clear(); }

    // allow move and copy construct
    graph_data(this_type&&) = delete;
    graph_data& operator=(this_type&&) = delete;

    // delete copy-assignment
    graph_data(const this_type&) = delete;
    graph_data& operator=(const this_type&) = delete;

    bool has_head() const { return m_has_head; }
    auto dummy_count() const { return m_dummies.size(); }
    bool at_sea_level() const { return (m_depth == m_sea_level); }

    int64_t& depth() { return m_depth; }
    int64_t  depth() const { return m_depth; }
    int64_t& sea_level() { return m_sea_level; }
    int64_t  sea_level() const { return m_sea_level; }

    graph_t&       graph() { return m_graph; }
    const graph_t& graph() const { return m_graph; }

    iterator& head() { return m_head; }
    iterator& current() { return m_current; }

    iterator       begin() { return m_graph.begin(); }
    iterator       end() { return m_graph.end(); }
    const_iterator begin() const { return m_graph.begin(); }
    const_iterator end() const { return m_graph.end(); }

    inline void sync_sea_level()
    {
        if(m_master)
        {
            TIMEMORY_DEBUG_PRINT_HERE(
                "[%s][%i]> synchronizing sea-level for depth = %i, master "
                "depth = %i",
                demangle<NodeT>().c_str(), (int) threading::get_id(), (int) depth(),
                (int) m_master->depth());
            add_dummy();
        }
    }

    inline void clear()
    {
        m_graph.clear();
        m_has_head  = false;
        m_depth     = 0;
        m_sea_level = 0;
        m_current   = nullptr;
        m_dummies.clear();
    }

    inline void set_master(graph_data* _master)
    {
        if(_master != this)
            m_master = _master;
    }

    inline void add_dummy()
    {
        if(!m_master)
            return;

        if(depth() == m_master->depth())
            return;

        TIMEMORY_DEBUG_PRINT_HERE(
            "[%s][%i]> Adding dummy for depth = %i, master depth = %i",
            demangle<NodeT>().c_str(), (int) threading::get_id(), (int) depth(),
            (int) m_master->depth());

        auto _current = m_master->current();
        auto _hash    = _current->hash();
        auto _depth   = _current->depth();

        NodeT node{
            _hash, NodeT::get_dummy(), _depth, threading::get_id(), process::get_id(),
            true
        };
        m_depth     = _depth;
        m_sea_level = _depth;
        m_current   = m_graph.insert_after(m_head, std::move(node));

        m_dummies.insert({ m_depth, m_current });
    }

    inline void reset()
    {
        // for(auto& itr : m_dummies)
        //    m_graph.erase(itr.second);
        m_graph.erase_children(m_head);
        m_depth   = 0;
        m_current = m_head;
    }

    inline iterator pop_graph()
    {
        if(m_depth > 0 && !m_graph.is_head(m_current))
        {
            --m_depth;
            m_current = graph_t::parent(m_current);
        }
        else if(m_depth == 0)
        {
            m_current = m_head;
        }
        return m_current;
    }

    inline iterator find(iterator itr) const
    {
        if(!itr)
            return end();

        auto _equal = [](auto _lhs, auto _rhs) {
            return (_lhs->hash() == _rhs->hash()) && (_lhs->depth() == _rhs->depth());
        };

        for(pre_order_iterator fitr = begin(); fitr != end(); ++fitr)
        {
            if(fitr && _equal(itr, fitr))
                return fitr;
            sibling_iterator entry = fitr;
            for(auto eitr = entry.begin(); eitr != entry.end(); ++eitr)
            {
                if(eitr && _equal(itr, eitr))
                    return eitr;
            }
        }
        return end();
    }

    static inline hash_value_t get_rolling_hash(iterator itr)
    {
        hash_value_t _accum = 0;
        if(!itr)
            return _accum;
        _accum += itr->id();
        auto _parent = graph_t::parent(itr);
        // using Tp     = std::decay_t<decltype(std::declval<iterator>()->data())>;
        while(_parent)
        {
            // if(!operation::get_is_invalid<Tp, false>{}(_parent->data()))
            _accum += _parent->hash();
            _parent = graph_t::parent(_parent);
            if(!_parent)
                break;
        }
        return _accum;
    }

    inline iterator append_child(const NodeT& node)
    {
        ++m_depth;
        return (m_current = m_graph.append_child(m_current, node));
    }

    inline iterator append_child(NodeT&& node)
    {
        ++m_depth;
        return (m_current = m_graph.append_child(m_current, std::move(node)));
    }

    inline iterator append_head(NodeT& node)
    {
        return m_graph.append_child(m_head, node);
    }

    inline iterator emplace_child(iterator _itr, const NodeT& node)
    {
        return m_graph.append_child(_itr, node);
    }

    inline iterator emplace_child(iterator _itr, NodeT&& node)
    {
        return m_graph.append_child(_itr, std::move(node));
    }

    inverse_insert_t get_inverse_insert() const
    {
        inverse_insert_t ret{};
        ret.reserve(m_dummies.size());
        for(const auto& itr : m_dummies)
            ret.emplace_back(itr.first, itr.second);
        std::reverse(ret.begin(), ret.end());
        return ret;
    }

private:
    bool                             m_has_head  = false;
    int64_t                          m_depth     = 0;
    int64_t                          m_sea_level = 0;
    graph_t                          m_graph;
    iterator                         m_current = nullptr;
    iterator                         m_head    = nullptr;
    graph_data*                      m_master  = nullptr;
    std::multimap<int64_t, iterator> m_dummies = {};
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
