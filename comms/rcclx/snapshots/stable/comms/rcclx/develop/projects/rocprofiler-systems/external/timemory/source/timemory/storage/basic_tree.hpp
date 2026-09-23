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

#include "timemory/hash/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/types.hpp"

#include <iterator>
#include <set>
#include <vector>

namespace tim
{
/// \struct tim::basic_tree
/// \tparam Tp Component type
///
/// \brief Basic hierarchical tree implementation. Expects population from
/// \ref tim::graph
template <typename Tp>
struct basic_tree
{
    using this_type     = basic_tree<Tp>;
    using value_type    = Tp;
    using child_type    = this_type;
    using child_pointer = std::shared_ptr<child_type>;
    using children_type = std::vector<child_pointer>;
    using children_base = std::vector<child_type>;

    TIMEMORY_DEFAULT_OBJECT(basic_tree)

    /// construction from `tim::graph<Tp>`
    template <typename GraphT, typename ItrT>
    TIMEMORY_COLD this_type& operator()(const GraphT& g, ItrT root);

    TIMEMORY_COLD this_type& operator+=(const this_type& rhs);
    TIMEMORY_COLD this_type& operator-=(const this_type& rhs);

    template <typename Archive>
    TIMEMORY_COLD void save(Archive& ar, const unsigned int) const;

    template <typename Archive>
    TIMEMORY_COLD void load(Archive& ar, const unsigned int);

    /// return the current tree node
    auto& get_value() { return m_value; }
    /// return the array of child nodes
    auto& get_children() { return m_children; }

    const auto& get_value() const { return m_value; }
    const auto& get_children() const { return m_children; }

    friend bool operator==(const this_type& lhs, const this_type& rhs)
    {
        // auto _lhash = get_hash_id(get_hash_aliases(), lhs.m_value.hash());
        // auto _rhash = get_hash_id(get_hash_aliases(), rhs.m_value.hash());
        return (lhs.m_value.hash() == rhs.m_value.hash());
    }

    friend bool operator!=(const this_type& lhs, const this_type& rhs)
    {
        return !(lhs == rhs);
    }

private:
    value_type    m_value    = {};
    children_type m_children = {};
};
//
template <typename Tp>
template <typename GraphT, typename ItrT>
basic_tree<Tp>&
basic_tree<Tp>::operator()(const GraphT& g, ItrT root)
{
    using iterator_t = typename GraphT::sibling_iterator;
    using entry_type = std::decay_t<decltype(std::declval<ItrT>()->data())>;

    m_value              = *root;
    iterator_t _begin    = g.begin(root);
    iterator_t _end      = g.end(root);
    auto       nchildren = std::distance(_begin, _end);
    if(nchildren > 0)
    {
        m_children.reserve(nchildren);
        for(auto itr = _begin; itr != _end; ++itr)
        {
            if(!itr->is_dummy() &&
               !operation::get_is_invalid<entry_type, false>{}(itr->data()))
            {
                m_value.exclusive().data() -= itr->data();
                m_value.exclusive().stats() -= itr->stats();
                m_children.emplace_back(std::make_shared<child_type>());
                m_children.back()->operator()(g, itr);
            }
            else
            {
                iterator_t _dbegin = g.begin(itr);
                iterator_t _dend   = g.end(itr);
                for(auto ditr = _dbegin; ditr != _dend; ++ditr)
                {
                    if(!ditr->is_dummy())
                    {
                        m_children.emplace_back(std::make_shared<child_type>());
                        m_children.back()->operator()(g, ditr);
                    }
                }
            }
        }
    }
    return *this;
}

template <typename Tp>
basic_tree<Tp>&
basic_tree<Tp>::operator+=(const this_type& rhs)
{
    if(*this == rhs)
    {
        m_value += rhs.m_value;
    }
    if(false)
    {
        for(auto& ritr : rhs.m_children)
            m_children.insert(m_children.end(), ritr);
    }
    else
    {
        std::set<size_t> found{};
        auto nitr = std::min<size_t>(m_children.size(), rhs.m_children.size());
        // add identical entries
        for(size_t i = 0; i < nitr; ++i)
        {
            if((*m_children.at(i)) == (*rhs.m_children.at(i)))
            {
                found.insert(i);
                (*m_children.at(i)) += (*rhs.m_children.at(i));
            }
        }
        // add to first matching entry
        for(size_t i = 0; i < rhs.m_children.size(); ++i)
        {
            if(found.find(i) != found.end())
                continue;
            for(size_t j = 0; j < m_children.size(); ++j)
            {
                if((*m_children.at(j)) == (*rhs.m_children.at(i)))
                {
                    found.insert(i);
                    (*m_children.at(j)) += (*rhs.m_children.at(i));
                }
            }
        }
        // append to end if not found anywhere
        for(size_t i = 0; i < rhs.m_children.size(); ++i)
        {
            if(found.find(i) != found.end())
                continue;
            m_children.insert(m_children.end(), rhs.m_children.at(i));
        }
    }
    return *this;
}

template <typename Tp>
basic_tree<Tp>&
basic_tree<Tp>::operator-=(const this_type& rhs)
{
    if(*this == rhs)
    {
        m_value -= rhs.m_value;
    }

    std::set<size_t> found{};
    auto             nitr = std::min<size_t>(m_children.size(), rhs.m_children.size());
    // add identical entries
    for(size_t i = 0; i < nitr; ++i)
    {
        if((*m_children.at(i)) == (*rhs.m_children.at(i)))
        {
            found.insert(i);
            m_children.at(i) -= rhs.m_children.at(i);
        }
    }
    // add to first matching entry
    for(size_t i = 0; i < rhs.m_children.size(); ++i)
    {
        if(found.find(i) != found.end())
            continue;
        for(size_t j = 0; j < m_children.size(); ++j)
        {
            if((*m_children.at(j)) == (*rhs.m_children.at(i)))
            {
                found.insert(i);
                m_children.at(j) -= rhs.m_children.at(i);
            }
        }
    }
    return *this;
}

template <typename Tp>
template <typename Archive>
void
basic_tree<Tp>::save(Archive& ar, const unsigned int) const
{
    // this is for backward compatiblity
    children_base _children{};
    for(const auto& itr : m_children)
        _children.emplace_back(*itr);
    ar(cereal::make_nvp("node", m_value), cereal::make_nvp("children", _children));
}

template <typename Tp>
template <typename Archive>
void
basic_tree<Tp>::load(Archive& ar, const unsigned int)
{
    // this is for backward compatiblity
    children_base _children{};
    ar(cereal::make_nvp("node", m_value), cereal::make_nvp("children", _children));
    for(auto&& itr : _children)
        m_children.emplace_back(std::make_shared<child_type>(std::move(itr)));
}
//
}  // namespace tim
