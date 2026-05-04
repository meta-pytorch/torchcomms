//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#pragma once

#include "timemory/data/ring_buffer_allocator.hpp"
#include "timemory/data/shared_stateful_allocator.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/units.hpp"

#include <cassert>
#include <cstddef>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

//--------------------------------------------------------------------------------------//

namespace tim
{
//======================================================================================//

/// A node in the graph, combining links to other nodes as well as the actual
/// data.
template <typename T>
class tgraph_node
{
    // size: 5*4=20 bytes (on 32 bit arch), can be reduced by 8.
public:
    tgraph_node()  = default;
    ~tgraph_node() = default;
    explicit tgraph_node(const T&);  // NOLINT
    explicit tgraph_node(T&&) noexcept;

#if defined(TIMEMORY_WINDOWS) || defined(TIMEMORY_NVCC_COMPILER)
    tgraph_node(const tgraph_node&) = default;
    tgraph_node& operator=(const tgraph_node&) = default;
#else
    tgraph_node(const tgraph_node&) = delete;
    tgraph_node& operator=(const tgraph_node&) = delete;
#endif

    tgraph_node(tgraph_node&&) = default;             // NOLINT
    tgraph_node& operator=(tgraph_node&&) = default;  // NOLINT

    tgraph_node<T>* parent       = nullptr;
    tgraph_node<T>* first_child  = nullptr;
    tgraph_node<T>* last_child   = nullptr;
    tgraph_node<T>* prev_sibling = nullptr;
    tgraph_node<T>* next_sibling = nullptr;
    T               data         = T{};

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("data", data));
    }
};

//======================================================================================//

template <typename T>
tgraph_node<T>::tgraph_node(const T& val)  // NOLINT
: data(val)                                // NOLINT
{}

//--------------------------------------------------------------------------------------//

template <typename T>
tgraph_node<T>::tgraph_node(T&& val) noexcept
: data(std::move(val))
{}

//======================================================================================//
/// \class tim::graph
/// \brief Arbitrary Graph / Tree (i.e. binary-tree but not binary). It is unlikely that
/// this class will interacted with directly.
///
template <typename T, typename AllocatorT>
class graph
{
protected:
    using graph_node = tgraph_node<T>;
    using this_type  = graph<T, AllocatorT>;

public:
    /// Value of the data stored at a node.
    using value_type = T;

    class iterator_base;
    class pre_order_iterator;
    class sibling_iterator;

    graph();          // empty constructor
    graph(const T&);  // constructor setting given element as head
    graph(const iterator_base&);
    ~graph();
    graph(const this_type&)     = delete;
    graph(this_type&&) noexcept = default;
    graph& operator=(const graph&) = delete;
    graph& operator=(graph&&) noexcept = default;

    /// Base class for iterators, only pointers stored, no traversal logic.
    class iterator_base
    {
    public:
        typedef T                               value_type;
        typedef T*                              pointer;
        typedef T&                              reference;
        typedef size_t                          size_type;
        typedef ptrdiff_t                       difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

        iterator_base() = default;
        iterator_base(graph_node* _v)
        : node{ _v }
        {}

        iterator_base(const iterator_base&)     = default;
        iterator_base(iterator_base&&) noexcept = default;

    public:
        // public operators
        iterator_base& operator=(const iterator_base&) = default;
        iterator_base& operator=(iterator_base&&) noexcept = default;

        operator bool() const { return node != nullptr; }

        T& operator*() const;
        T* operator->() const;

    public:
        // public member functions
        graph_node* parent() const;
        /// Number of children of the node pointed to by the iterator.
        unsigned int number_of_children() const;

        sibling_iterator begin() const;
        sibling_iterator end() const;

    public:
        // public data member
        graph_node* node = nullptr;
    };

    /// Depth-first iterator, first accessing the node, then its children.
    class pre_order_iterator : public iterator_base
    {
    public:
        pre_order_iterator();
        pre_order_iterator(graph_node*);
        pre_order_iterator(const iterator_base&);
        pre_order_iterator(const sibling_iterator&);

        pre_order_iterator(const pre_order_iterator&)     = default;
        pre_order_iterator(pre_order_iterator&&) noexcept = default;

    public:
        // public operators
        pre_order_iterator& operator=(const pre_order_iterator&) = default;
        pre_order_iterator& operator=(pre_order_iterator&&) noexcept = default;

        bool                operator==(const pre_order_iterator&) const;
        bool                operator!=(const pre_order_iterator&) const;
        pre_order_iterator& operator++();
        pre_order_iterator& operator--();
        pre_order_iterator  operator++(int);
        pre_order_iterator  operator--(int);
        pre_order_iterator& operator+=(unsigned int);
        pre_order_iterator& operator-=(unsigned int);
        pre_order_iterator  operator+(unsigned int);
        pre_order_iterator  operator-(unsigned int);

    public:
        using iterator_base::parent;
    };

    /// The default iterator types throughout the graph class.
    typedef pre_order_iterator                 iterator;
    typedef const iterator                     const_iterator;
    typedef typename iterator::difference_type difference_type;

    /// Iterator which traverses only the nodes which are siblings of each
    /// other.
    class sibling_iterator : public iterator_base
    {
    public:
        sibling_iterator();
        sibling_iterator(graph_node*);
        sibling_iterator(const iterator_base&);

        sibling_iterator(const sibling_iterator&)     = default;
        sibling_iterator(sibling_iterator&&) noexcept = default;

    public:
        // public operators
        sibling_iterator& operator=(const sibling_iterator&) = default;
        sibling_iterator& operator=(sibling_iterator&&) noexcept = default;

        bool              operator==(const sibling_iterator&) const;
        bool              operator!=(const sibling_iterator&) const;
        sibling_iterator& operator++();
        sibling_iterator& operator--();
        sibling_iterator  operator++(int);
        sibling_iterator  operator--(int);
        sibling_iterator& operator+=(unsigned int);
        sibling_iterator& operator-=(unsigned int);
        sibling_iterator  operator+(unsigned int);
        sibling_iterator  operator-(unsigned int);

    public:
        // public member functions
        graph_node* range_first() const;
        graph_node* range_last() const;

    public:
        using iterator_base::parent;
    };

    /// Return iterator to the beginning of the graph.
    inline pre_order_iterator begin() const;

    /// Return iterator to the end of the graph.
    inline pre_order_iterator end() const;

    /// Return sibling iterator to the first child of given node.
    static sibling_iterator begin(const iterator_base&);

    /// Return sibling end iterator for children of given node.
    static sibling_iterator end(const iterator_base&);

    /// Return iterator to the parent of a node.
    template <typename IterT>
    static IterT parent(IterT);

    /// Return iterator to the previous sibling of a node.
    template <typename IterT>
    static IterT previous_sibling(IterT);

    /// Return iterator to the next sibling of a node.
    template <typename IterT>
    static IterT next_sibling(IterT);

    /// Erase all nodes of the graph.
    inline void clear();

    /// Erase element at position pointed to by iterator, return incremented
    /// iterator.
    template <typename IterT>
    inline IterT erase(IterT);

    /// Erase all children of the node pointed to by iterator.
    inline void erase_children(const iterator_base&);

    /// Insert empty node as last/first child of node pointed to by position.
    template <typename IterT>
    inline IterT append_child(IterT position);
    template <typename IterT>
    inline IterT prepend_child(IterT position);

    /// Insert node as last/first child of node pointed to by position.
    template <typename IterT>
    inline IterT append_child(IterT position, const T& x);
    template <typename IterT>
    inline IterT append_child(IterT position, T&& x);
    template <typename IterT>
    inline IterT prepend_child(IterT position, const T& x);
    template <typename IterT>
    inline IterT prepend_child(IterT position, T&& x);

    /// Append the node (plus its children) at other_position as last/first
    /// child of position.
    template <typename IterT>
    inline IterT append_child(IterT position, IterT other_position);
    template <typename IterT>
    inline IterT prepend_child(IterT position, IterT other_position);

    /// Append the nodes in the from-to range (plus their children) as
    /// last/first children of position.
    template <typename IterT>
    inline IterT append_children(IterT position, sibling_iterator from,
                                 const sibling_iterator& to);
    template <typename IterT>
    inline IterT prepend_children(IterT position, sibling_iterator from,
                                  sibling_iterator to);

    /// Short-hand to insert topmost node in otherwise empty graph.
    inline pre_order_iterator set_head(const T& x);
    inline pre_order_iterator set_head(T&& x);

    /// Insert node as previous sibling of node pointed to by position.
    template <typename IterT>
    inline IterT insert(IterT position, const T& x);
    template <typename IterT>
    inline IterT insert(IterT position, T&& x);

    /// Specialisation of previous member.
    inline sibling_iterator insert(sibling_iterator position, const T& x);

    /// Insert node (with children) pointed to by subgraph as previous sibling
    /// of node pointed to by position. Does not change the subgraph itself (use
    /// move_in or move_in_below for that).
    template <typename IterT>
    inline IterT insert_subgraph(IterT position, const iterator_base& subgraph);

    /// Insert node as next sibling of node pointed to by position.
    template <typename IterT>
    inline IterT insert_after(IterT position, const T& x);
    template <typename IterT>
    inline IterT insert_after(IterT position, T&& x);

    /// Insert node (with children) pointed to by subgraph as next sibling of
    /// node pointed to by position.
    template <typename IterT>
    inline IterT insert_subgraph_after(IterT position, const iterator_base& subgraph);

    /// Replace node at 'position' with other node (keeping same children);
    /// 'position' becomes invalid.
    template <typename IterT>
    inline IterT replace(IterT position, const T& x);

    template <typename IterT>
    inline IterT replace(IterT position, T&& x);

    /// Replace node at 'position' with subgraph starting at 'from' (do not
    /// erase subgraph at 'from'); see above.
    template <typename IterT>
    inline IterT replace(IterT position, const iterator_base& from);

    /// Replace string of siblings (plus their children) with copy of a new
    /// string (with children); see above
    inline sibling_iterator replace(sibling_iterator        orig_begin,
                                    const sibling_iterator& orig_end,
                                    sibling_iterator        new_begin,
                                    const sibling_iterator& new_end);

    /// Move all children of node at 'position' to be siblings, returns
    /// position.
    template <typename IterT>
    inline IterT flatten(IterT position);

    /// Move nodes in range to be children of 'position'.
    template <typename IterT>
    inline IterT reparent(IterT position, sibling_iterator begin,
                          const sibling_iterator& end);
    /// Move all child nodes of 'from' to be children of 'position'.
    template <typename IterT>
    inline IterT reparent(IterT position, IterT from);

    /// Replace node with a new node, making the old node (plus subgraph) a
    /// child of the new node.
    template <typename IterT>
    inline IterT wrap(IterT position, const T& x);

    /// Replace the range of sibling nodes (plus subgraphs), making these
    /// children of the new node.
    template <typename IterT>
    inline IterT wrap(IterT from, IterT to, const T& x);

    /// Move 'source' node (plus its children) to become the next sibling of
    /// 'target'.
    template <typename IterT>
    inline IterT move_after(IterT target, IterT source);

    /// Move 'source' node (plus its children) to become the previous sibling of
    /// 'target'.
    template <typename IterT>
    inline IterT            move_before(IterT target, IterT source);
    inline sibling_iterator move_before(sibling_iterator target, sibling_iterator source);

    /// Move 'source' node (plus its children) to become the node at 'target'
    /// (erasing the node at 'target').
    template <typename IterT>
    inline IterT move_ontop(IterT target, IterT source);

    /// Extract the subgraph starting at the indicated node, removing it from
    /// the original graph.
    inline graph move_out(iterator);

    /// Inverse of take_out: inserts the given graph as previous sibling of
    /// indicated node by a move operation, that is, the given graph becomes
    /// empty. Returns iterator to the top node.
    template <typename IterT>
    inline IterT move_in(IterT, graph&);

    /// As above, but now make the graph a child of the indicated node.
    template <typename IterT>
    inline IterT move_in_below(IterT, graph&);

    /// As above, but now make the graph the nth child of the indicated node (if
    /// possible).
    template <typename IterT>
    inline IterT move_in_as_nth_child(IterT, size_t, graph&);

    /// Merge with other graph, creating new branches and leaves only if they
    /// are not already present.
    inline void merge(const sibling_iterator&, const sibling_iterator&, sibling_iterator,
                      const sibling_iterator&, bool duplicate_leaves = false,
                      bool first = false);

    /// Reduce duplicate nodes
    template <
        typename ComparePred = std::function<bool(sibling_iterator, sibling_iterator)>,
        typename ReducePred  = std::function<void(sibling_iterator, sibling_iterator)>>
    inline void reduce(
        const sibling_iterator&, const sibling_iterator&, std::set<sibling_iterator>&,
        ComparePred&& = [](sibling_iterator lhs,
                           sibling_iterator rhs) { return (*lhs == *rhs); },
        ReducePred&&  = [](sibling_iterator lhs, sibling_iterator rhs) { *lhs += *rhs; });

    /// Compare two ranges of nodes (compares nodes as well as graph structure).
    template <typename IterT>
    inline bool equal(const IterT& one, const IterT& two, const IterT& three) const;
    template <typename IterT, typename BinaryPredicate>
    inline bool equal(const IterT& one, const IterT& two, const IterT& three,
                      BinaryPredicate) const;
    template <typename IterT>
    inline bool equal_subgraph(const IterT& one, const IterT& two) const;
    template <typename IterT, typename BinaryPredicate>
    inline bool equal_subgraph(const IterT& one, const IterT& two, BinaryPredicate) const;

    /// Extract a new graph formed by the range of siblings plus all their
    /// children.
    inline graph subgraph(sibling_iterator from, sibling_iterator to) const;
    inline void  subgraph(graph&, sibling_iterator from, sibling_iterator to) const;

    /// Exchange the node (plus subgraph) with its sibling node (do nothing if
    /// no sibling present).
    inline void swap(sibling_iterator it);

    /// Exchange two nodes (plus subgraphs). The iterators will remain valid and
    /// keep pointing to the same nodes, which now sit at different locations in
    /// the graph.
    inline void swap(iterator, iterator);

    template <typename IterT, typename FuncT, typename ExecT, typename WaitT>
    inline void sort(IterT _beg, IterT _end, FuncT&& _functor, ExecT&& _exec,
                     WaitT&& _wait);

    template <typename FuncT, typename ExecT, typename WaitT>
    inline void sort(FuncT&& _func, ExecT&& _exec, WaitT&& _wait)
    {
        sort(begin(), end(), std::forward<FuncT>(_func), std::forward<ExecT>(_exec),
             std::forward<WaitT>(_wait));
    }

    template <typename IterT, typename FuncT, typename ExecT, typename WaitT>
    inline void sort(IterT itr, FuncT&& _func, ExecT&& _exec, WaitT&& _wait)
    {
        sort(itr, IterT{ next_sibling(itr) }, std::forward<FuncT>(_func),
             std::forward<ExecT>(_exec), std::forward<WaitT>(_wait));
    }

    template <typename IterT, typename FuncT>
    inline void sort(IterT itr, FuncT&& _func)
    {
        sort(
            itr, std::forward<FuncT>(_func), [](auto _f) { _f(); }, []() {});
    }

    template <typename IterT, typename FuncT>
    inline void sort(IterT _beg, IterT _end, FuncT&& _func)
    {
        sort(
            _beg, _end, std::forward<FuncT>(_func), [](auto _f) { _f(); }, []() {});
    }

    template <typename FuncT>
    inline void sort(FuncT&& _func)
    {
        sort(
            begin(), end(), std::forward<FuncT>(_func), [](auto _f) { _f(); }, []() {});
    }

    /// Count the total number of nodes.
    inline size_t size() const;

    /// Check if graph is empty.
    inline bool empty() const;

    /// Compute the depth to the root or to a fixed other iterator.
    static int depth(const iterator_base&);
    static int depth(const iterator_base&, const iterator_base&);

    /// Determine the maximal depth of the graph. An empty graph has
    /// max_depth=-1.
    inline int max_depth() const;

    /// Determine the maximal depth of the graph with top node at the given
    /// position.
    inline int max_depth(const iterator_base&) const;

    /// Count the number of children of node at position.
    static unsigned int number_of_children(const iterator_base&);

    /// Count the number of siblings (left and right) of node at iterator. Total
    /// nodes at this level is +1.
    inline unsigned int number_of_siblings(const iterator_base&) const;

    /// Determine whether node at position is in the subgraphs with root in the
    /// range.
    inline bool is_in_subgraph(const iterator_base& position, const iterator_base& begin,
                               const iterator_base& end) const;

    /// Determine whether the iterator is an 'end' iterator and thus not
    /// actually pointing to a node.
    inline bool is_valid(const iterator_base&) const;

    /// Determine whether the iterator is one of the 'head' nodes at the top
    /// level, i.e. has no parent.
    static bool is_head(const iterator_base&);

    /// Determine the index of a node in the range of siblings to which it
    /// belongs.
    inline unsigned int index(sibling_iterator it) const;

    /// Inverse of 'index': return the n-th child of the node at position.
    static sibling_iterator child(const iterator_base& position, unsigned int);

    /// Return iterator to the sibling indicated by index
    inline sibling_iterator sibling(const iterator_base& position, unsigned int) const;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        for(auto itr = begin(); itr != end(); ++itr)
            ar(cereal::make_nvp("node", *itr));
    }

    void steal_resources(this_type& rhs) { m_stolen_alloc.emplace_back(rhs.m_alloc); }

    graph_node* head = nullptr;  // head/feet are always dummy; if an iterator
    graph_node* feet = nullptr;  // points to them it is invalid

private:
    inline void m_head_initialize();

private:
    using allocator_traits    = std::allocator_traits<AllocatorT>;
    using allocator_pointer   = std::shared_ptr<AllocatorT>;
    allocator_pointer m_alloc = data::shared_stateful_allocator<AllocatorT>::request();
    std::vector<allocator_pointer> m_stolen_alloc = {};
};

//======================================================================================//
// Graph
//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::graph()
{
    m_head_initialize();
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::graph(const T& x)
{
    m_head_initialize();
    set_head(x);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::graph(const iterator_base& other)
{
    m_head_initialize();
    set_head((*other));
    replace(begin(), other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::~graph()
{
    clear();
    if(m_alloc)
    {
        allocator_traits::destroy(*m_alloc, head);
        allocator_traits::destroy(*m_alloc, feet);
        allocator_traits::deallocate(*m_alloc, head, 1);
        allocator_traits::deallocate(*m_alloc, feet, 1);
    }
    while(!m_stolen_alloc.empty())
    {
        auto _v = m_stolen_alloc.back();
        m_stolen_alloc.pop_back();
        data::shared_stateful_allocator<AllocatorT>::release(_v);
    }
    data::shared_stateful_allocator<AllocatorT>::release(m_alloc);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::m_head_initialize()
{
    head = allocator_traits::allocate(*m_alloc, 1, nullptr);  // MSVC does not have
                                                              // default second argument
    feet = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, head, std::move(tgraph_node<T>{}));
    allocator_traits::construct(*m_alloc, feet, std::move(tgraph_node<T>{}));

    head->parent       = nullptr;
    head->first_child  = nullptr;
    head->last_child   = nullptr;
    head->prev_sibling = nullptr;  // head;
    head->next_sibling = feet;     // head;

    feet->parent       = nullptr;
    feet->first_child  = nullptr;
    feet->last_child   = nullptr;
    feet->prev_sibling = head;
    feet->next_sibling = nullptr;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::clear()
{
    if(head)
    {
        while(head->next_sibling != feet)
            erase(pre_order_iterator(head->next_sibling));
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::erase_children(const iterator_base& it)
{
    if(it.node == nullptr)
        return;

    graph_node* cur = it.node->first_child;

    if(cur)
    {
        while(cur->next_sibling && cur->next_sibling != feet)
            erase(pre_order_iterator(cur->next_sibling));
    }

    it.node->first_child = nullptr;
    it.node->last_child  = nullptr;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::erase(IterT it)
{
    graph_node* cur = it.node;
    assert(cur != head);
    assert(cur != feet);
    if(cur == head || cur == feet)
        return it;
    IterT ret = it;
    ++ret;
    erase_children(it);
    if(cur->parent && cur->prev_sibling == nullptr)
    {
        cur->parent->first_child = cur->next_sibling;
    }
    else
    {
        cur->prev_sibling->next_sibling = cur->next_sibling;
    }

    if(cur->parent && cur->next_sibling == nullptr)
    {
        cur->parent->last_child = cur->prev_sibling;
    }
    else
    {
        cur->next_sibling->prev_sibling = cur->prev_sibling;
    }

    if(m_alloc)
    {
        allocator_traits::destroy(*m_alloc, cur);
        allocator_traits::deallocate(*m_alloc, cur, 1);
    }
    it.node = nullptr;
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::begin() const
{
    return pre_order_iterator(head->next_sibling);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::end() const
{
    return pre_order_iterator(feet);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::begin(const iterator_base& pos)
{
    if(pos.node->first_child == nullptr)
        return end(pos);
    return pos.node->first_child;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::end(const iterator_base&)
{
    return sibling_iterator{ nullptr };
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::parent(IterT position)
{
    return IterT{ (position.node) ? position.node->parent : nullptr };
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::previous_sibling(IterT position)
{
    if(position.node)
    {
        IterT ret{ position };
        ret.node = position.node->prev_sibling;
        return ret;
    }
    return IterT{ nullptr };
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::next_sibling(IterT position)
{
    if(position.node)
    {
        IterT ret{ position };
        ret.node = position.node->next_sibling;
        return ret;
    }
    return IterT{ nullptr };
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::append_child(IterT position)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, std::move(tgraph_node<T>{}));
    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent = position.node;
    if(position.node->last_child != nullptr)
    {
        position.node->last_child->next_sibling = tmp;
    }
    else
    {
        position.node->first_child = tmp;
    }
    tmp->prev_sibling         = position.node->last_child;
    position.node->last_child = tmp;
    tmp->next_sibling         = nullptr;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::prepend_child(IterT position)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, std::move(tgraph_node<T>{}));
    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent = position.node;
    if(position.node->first_child != nullptr)
    {
        position.node->first_child->prev_sibling = tmp;
    }
    else
    {
        position.node->last_child = tmp;
    }
    tmp->next_sibling         = position.node->first_child;
    position.node->prev_child = tmp;
    tmp->prev_sibling         = nullptr;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::append_child(IterT position, const T& x)
{
    // If your program fails here you probably used 'append_child' to add the
    // top node to an empty graph. From version 1.45 the top element should be
    // added using 'insert'. See the documentation for further information, and
    // sorry about the API change.
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, x);
    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent = position.node;
    if(position.node->last_child != nullptr)
    {
        position.node->last_child->next_sibling = tmp;
    }
    else
    {
        position.node->first_child = tmp;
    }
    tmp->prev_sibling         = position.node->last_child;
    position.node->last_child = tmp;
    tmp->next_sibling         = nullptr;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::append_child(IterT position, T&& x)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, std::forward<T>(x));

    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent = position.node;
    if(position.node->last_child != nullptr)
    {
        position.node->last_child->next_sibling = tmp;
    }
    else
    {
        position.node->first_child = tmp;
    }
    tmp->prev_sibling         = position.node->last_child;
    position.node->last_child = tmp;
    tmp->next_sibling         = nullptr;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::prepend_child(IterT position, const T& x)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, x);
    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent = position.node;
    if(position.node->first_child != nullptr)
    {
        position.node->first_child->prev_sibling = tmp;
    }
    else
    {
        position.node->last_child = tmp;
    }
    tmp->next_sibling          = position.node->first_child;
    position.node->first_child = tmp;
    tmp->prev_sibling          = nullptr;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::prepend_child(IterT position, T&& x)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, std::forward<T>(x));

    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent = position.node;
    if(position.node->first_child != nullptr)
    {
        position.node->first_child->prev_sibling = tmp;
    }
    else
    {
        position.node->last_child = tmp;
    }
    tmp->next_sibling          = position.node->first_child;
    position.node->first_child = tmp;
    tmp->prev_sibling          = nullptr;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::append_child(IterT position, IterT other)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    IterT aargh = append_child(position, value_type{});
    return move_ontop(aargh, other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::prepend_child(IterT position, IterT other)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    IterT aargh = prepend_child(position, value_type{});
    return move_ontop(aargh, other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::append_children(IterT position, sibling_iterator from,
                                      const sibling_iterator& to)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    IterT ret = from;

    while(from != to)
    {
        insert_subgraph(position.end(), from);
        ++from;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::prepend_children(IterT position, sibling_iterator from,
                                       sibling_iterator to)
{
    assert(position.node != head);
    assert(position.node != feet);
    assert(position.node);

    if(from == to)
        return from;  // should return end of graph?

    IterT ret;
    do
    {
        --to;
        ret = insert_subgraph(position.begin(), to);
    } while(to != from);

    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::set_head(const T& x)
{
    assert(head->next_sibling == feet);
    return insert(iterator(feet), x);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::set_head(T&& x)
{
    assert(head->next_sibling == feet);
    return insert(iterator(feet), std::move(x));
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::insert(IterT position, const T& x)
{
    if(position.node == nullptr)
    {
        position.node = feet;  // Backward compatibility: when calling insert on
                               // a null node, insert before the feet.
    }
    assert(position.node != head);  // Cannot insert before head.

    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, x);
    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent                 = position.node->parent;
    tmp->next_sibling           = position.node;
    tmp->prev_sibling           = position.node->prev_sibling;
    position.node->prev_sibling = tmp;

    if(tmp->prev_sibling == nullptr)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->first_child = tmp;
    }
    else
        tmp->prev_sibling->next_sibling = tmp;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::insert(IterT position, T&& x)
{
    if(position.node == nullptr)
    {
        position.node = feet;  // Backward compatibility: when calling insert on
                               // a null node, insert before the feet.
    }
    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, std::forward<T>(x));

    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent                 = position.node->parent;
    tmp->next_sibling           = position.node;
    tmp->prev_sibling           = position.node->prev_sibling;
    position.node->prev_sibling = tmp;

    if(tmp->prev_sibling == nullptr)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->first_child = tmp;
    }
    else
        tmp->prev_sibling->next_sibling = tmp;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::insert(sibling_iterator position, const T& x)
{
    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, x);
    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->next_sibling = position.node;
    if(position.node == nullptr)
    {  // iterator points to end of a subgraph
        tmp->parent             = position.parent();
        tmp->prev_sibling       = position.range_last();
        tmp->parent->last_child = tmp;
    }
    else
    {
        tmp->parent                 = position.node->parent;
        tmp->prev_sibling           = position.node->prev_sibling;
        position.node->prev_sibling = tmp;
    }

    if(tmp->prev_sibling == nullptr)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->first_child = tmp;
    }
    else
        tmp->prev_sibling->next_sibling = tmp;
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::insert_after(IterT position, const T& x)
{
    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, x);
    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent                 = position.node->parent;
    tmp->prev_sibling           = position.node;
    tmp->next_sibling           = position.node->next_sibling;
    position.node->next_sibling = tmp;

    if(tmp->next_sibling == nullptr)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->last_child = tmp;
    }
    else
    {
        tmp->next_sibling->prev_sibling = tmp;
    }
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::insert_after(IterT position, T&& x)
{
    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, std::forward<T>(x));

    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;

    tmp->parent                 = position.node->parent;
    tmp->prev_sibling           = position.node;
    tmp->next_sibling           = position.node->next_sibling;
    position.node->next_sibling = tmp;

    if(tmp->next_sibling == nullptr)
    {
        if(tmp->parent)  // when inserting nodes at the head, there is no parent
            tmp->parent->last_child = tmp;
    }
    else
    {
        tmp->next_sibling->prev_sibling = tmp;
    }
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::insert_subgraph(IterT position, const iterator_base& _subgraph)
{
    // insert dummy
    IterT it = insert(position, value_type{});
    // replace dummy with subgraph
    return replace(it, _subgraph);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::insert_subgraph_after(IterT                position,
                                            const iterator_base& _subgraph)
{
    // insert dummy
    IterT it = insert_after(position, value_type{});
    // replace dummy with subgraph
    return replace(it, _subgraph);
}

//--------------------------------------------------------------------------------------//

// template <typename T, typename AllocatorT>
// template <typename IterT>
// IterT graph<T, AllocatorT>::insert_subgraph(sibling_iterator
// position, IterT subgraph)
// 	{
// 	// insert dummy
// 	IterT it(insert(position, value_type{}));
// 	// replace dummy with subgraph
// 	return replace(it, subgraph);
// 	}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::replace(IterT position, const T& x)
{
    allocator_traits::destroy(*m_alloc, position.node);
    allocator_traits::construct(*m_alloc, position.node, x);
    return position;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::replace(IterT position, T&& x)
{
    allocator_traits::destroy(*m_alloc, position.node);
    allocator_traits::construct(*m_alloc, position.node, std::forward<T>(x));
    return position;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::replace(IterT position, const iterator_base& from)
{
    assert(position.node != head);
    graph_node* current_from = from.node;
    graph_node* start_from   = from.node;
    graph_node* current_to   = position.node;

    // replace the node at position with head of the replacement graph at from
    //	std::cout << "warning!" << position.node << std::endl;
    erase_children(position);
    //	std::cout << "no warning!" << std::endl;
    graph_node* tmp = allocator_traits::allocate(*m_alloc, 1, nullptr);
    allocator_traits::construct(*m_alloc, tmp, (*from));
    tmp->first_child = nullptr;
    tmp->last_child  = nullptr;
    if(current_to->prev_sibling == nullptr)
    {
        if(current_to->parent != nullptr)
            current_to->parent->first_child = tmp;
    }
    else
    {
        current_to->prev_sibling->next_sibling = tmp;
    }
    tmp->prev_sibling = current_to->prev_sibling;
    if(current_to->next_sibling == nullptr)
    {
        if(current_to->parent != nullptr)
            current_to->parent->last_child = tmp;
    }
    else
    {
        current_to->next_sibling->prev_sibling = tmp;
    }
    tmp->next_sibling = current_to->next_sibling;
    tmp->parent       = current_to->parent;
    allocator_traits::destroy(*m_alloc, current_to);
    allocator_traits::deallocate(*m_alloc, current_to, 1);
    current_to = tmp;

    // only at this stage can we fix 'last'
    graph_node* last = from.node->next_sibling;

    pre_order_iterator toit = tmp;
    // copy all children
    do
    {
        assert(current_from != nullptr);
        if(current_from->first_child != nullptr)
        {
            current_from = current_from->first_child;
            toit         = append_child(toit, current_from->data);
        }
        else
        {
            while(current_from->next_sibling == nullptr && current_from != start_from)
            {
                current_from = current_from->parent;
                toit         = parent(toit);
                assert(current_from != nullptr);
            }
            current_from = current_from->next_sibling;
            if(current_from != last && current_from)
            {
                toit = append_child(parent(toit), current_from->data);
            }
        }
    } while(current_from != last && current_from);

    return current_to;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::replace(sibling_iterator        orig_begin,
                              const sibling_iterator& orig_end,
                              sibling_iterator new_begin, const sibling_iterator& new_end)
{
    graph_node* orig_first = orig_begin.node;
    graph_node* new_first  = new_begin.node;
    graph_node* orig_last  = orig_first;
    while((++orig_begin) != orig_end)
        orig_last = orig_last->next_sibling;
    graph_node* new_last = new_first;
    while((++new_begin) != new_end)
        new_last = new_last->next_sibling;

    // insert all siblings in new_first..new_last before orig_first
    bool               first = true;
    pre_order_iterator ret;
    while(true)
    {
        pre_order_iterator tt = insert_subgraph(pre_order_iterator(orig_first),
                                                pre_order_iterator(new_first));
        if(first)
        {
            ret   = tt;
            first = false;
        }
        if(new_first == new_last)
            break;
        new_first = new_first->next_sibling;
    }

    // erase old range of siblings
    bool        last = false;
    graph_node* next = orig_first;
    while(true)
    {
        if(next == orig_last)
            last = true;
        next = next->next_sibling;
        erase((pre_order_iterator) orig_first);
        if(last)
            break;
        orig_first = next;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::flatten(IterT position)
{
    if(position.node->first_child == nullptr)
        return position;

    graph_node* tmp = position.node->first_child;
    while(tmp)
    {
        tmp->parent = position.node->parent;
        tmp         = tmp->next_sibling;
    }
    if(position.node->next_sibling)
    {
        position.node->last_child->next_sibling   = position.node->next_sibling;
        position.node->next_sibling->prev_sibling = position.node->last_child;
    }
    else
    {
        position.node->parent->last_child = position.node->last_child;
    }
    position.node->next_sibling               = position.node->first_child;
    position.node->next_sibling->prev_sibling = position.node;
    position.node->first_child                = nullptr;
    position.node->last_child                 = nullptr;

    return position;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::reparent(IterT position, sibling_iterator _begin,
                               const sibling_iterator& _end)
{
    graph_node* first = _begin.node;
    graph_node* last  = first;

    assert(first != position.node);

    if(_begin == _end)
        return _begin;
    // determine last node
    while((++_begin) != _end)
    {
        last = last->next_sibling;
    }
    // move subgraph
    if(first->prev_sibling == nullptr)
    {
        first->parent->first_child = last->next_sibling;
    }
    else
    {
        first->prev_sibling->next_sibling = last->next_sibling;
    }
    if(last->next_sibling == nullptr)
    {
        last->parent->last_child = first->prev_sibling;
    }
    else
    {
        last->next_sibling->prev_sibling = first->prev_sibling;
    }
    if(position.node->first_child == nullptr)
    {
        position.node->first_child = first;
        position.node->last_child  = last;
        first->prev_sibling        = nullptr;
    }
    else
    {
        position.node->last_child->next_sibling = first;
        first->prev_sibling                     = position.node->last_child;
        position.node->last_child               = last;
    }
    last->next_sibling = nullptr;

    graph_node* pos = first;
    for(;;)
    {
        pos->parent = position.node;
        if(pos == last)
            break;
        pos = pos->next_sibling;
    }

    return first;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::reparent(IterT position, IterT from)
{
    if(from.node->first_child == nullptr)
        return position;
    return reparent(position, from.node->first_child, end(from));
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::wrap(IterT position, const T& x)
{
    assert(position.node != nullptr);
    sibling_iterator fr = position;
    sibling_iterator to = position;
    ++to;
    IterT ret = insert(position, x);
    reparent(ret, fr, to);
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::wrap(IterT from, IterT to, const T& x)
{
    assert(from.node != nullptr);
    IterT ret = insert(from, x);
    reparent(ret, from, to);
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::move_after(IterT target, IterT source)
{
    graph_node* dst = target.node;
    graph_node* src = source.node;
    assert(dst);
    assert(src);

    if(dst == src)
        return source;
    if(dst->next_sibling)
    {
        if(dst->next_sibling == src)  // already in the right spot
            return source;
    }

    // take src out of the graph
    if(src->prev_sibling != nullptr)
    {
        src->prev_sibling->next_sibling = src->next_sibling;
    }
    else
    {
        src->parent->first_child = src->next_sibling;
    }
    if(src->next_sibling != nullptr)
    {
        src->next_sibling->prev_sibling = src->prev_sibling;
    }
    else
    {
        src->parent->last_child = src->prev_sibling;
    }

    // connect it to the new point
    if(dst->next_sibling != nullptr)
    {
        dst->next_sibling->prev_sibling = src;
    }
    else
    {
        dst->parent->last_child = src;
    }
    src->next_sibling = dst->next_sibling;
    dst->next_sibling = src;
    src->prev_sibling = dst;
    src->parent       = dst->parent;
    return src;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::move_before(IterT target, IterT source)
{
    graph_node* dst = target.node;
    graph_node* src = source.node;
    assert(dst);
    assert(src);

    if(dst == src)
        return source;
    if(dst->prev_sibling)
    {
        if(dst->prev_sibling == src)  // already in the right spot
            return source;
    }

    // take src out of the graph
    if(src->prev_sibling != nullptr)
    {
        src->prev_sibling->next_sibling = src->next_sibling;
    }
    else
    {
        src->parent->first_child = src->next_sibling;
    }
    if(src->next_sibling != nullptr)
    {
        src->next_sibling->prev_sibling = src->prev_sibling;
    }
    else
    {
        src->parent->last_child = src->prev_sibling;
    }

    // connect it to the new point
    if(dst->prev_sibling != nullptr)
    {
        dst->prev_sibling->next_sibling = src;
    }
    else
    {
        dst->parent->first_child = src;
    }
    src->prev_sibling = dst->prev_sibling;
    dst->prev_sibling = src;
    src->next_sibling = dst;
    src->parent       = dst->parent;
    return src;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::move_ontop(IterT target, IterT source)
{
    graph_node* dst = target.node;
    graph_node* src = source.node;
    assert(dst);
    assert(src);

    if(dst == src)
        return source;

    //	if(dst==src->prev_sibling) {
    //
    //		}

    // remember connection points
    graph_node* b_prev_sibling = dst->prev_sibling;
    graph_node* b_next_sibling = dst->next_sibling;
    graph_node* b_parent       = dst->parent;

    // remove target
    erase(target);

    // take src out of the graph
    if(src->prev_sibling != nullptr)
    {
        src->prev_sibling->next_sibling = src->next_sibling;
    }
    else
    {
        src->parent->first_child = src->next_sibling;
    }
    if(src->next_sibling != nullptr)
    {
        src->next_sibling->prev_sibling = src->prev_sibling;
    }
    else
    {
        src->parent->last_child = src->prev_sibling;
    }

    // connect it to the new point
    if(b_prev_sibling != nullptr)
    {
        b_prev_sibling->next_sibling = src;
    }
    else
    {
        b_parent->first_child = src;
    }
    if(b_next_sibling != nullptr)
    {
        b_next_sibling->prev_sibling = src;
    }
    else
    {
        b_parent->last_child = src;
    }
    src->prev_sibling = b_prev_sibling;
    src->next_sibling = b_next_sibling;
    src->parent       = b_parent;
    return src;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>
graph<T, AllocatorT>::move_out(iterator source)
{
    graph ret;

    // Move source node into the 'ret' graph.
    ret.head->next_sibling = source.node;
    ret.feet->prev_sibling = source.node;
    source.node->parent    = nullptr;

    // Close the links in the current graph.
    if(source.node->prev_sibling != nullptr)
        source.node->prev_sibling->next_sibling = source.node->next_sibling;

    if(source.node->next_sibling != nullptr)
        source.node->next_sibling->prev_sibling = source.node->prev_sibling;

    // Fix source prev/next links.
    source.node->prev_sibling = ret.head;
    source.node->next_sibling = ret.feet;

    return ret;  // A good compiler will move this, not copy.
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::move_in(IterT loc, graph& other)
{
    if(other.head->next_sibling == other.feet)
        return loc;  // other graph is empty

    graph_node* other_first_head = other.head->next_sibling;
    graph_node* other_last_head  = other.feet->prev_sibling;

    sibling_iterator prev(loc);
    --prev;

    prev.node->next_sibling        = other_first_head;
    loc.node->prev_sibling         = other_last_head;
    other_first_head->prev_sibling = prev.node;
    other_last_head->next_sibling  = loc.node;

    // Adjust parent pointers.
    graph_node* walk = other_first_head;
    while(true)
    {
        walk->parent = loc.node->parent;
        if(walk == other_last_head)
            break;
        walk = walk->next_sibling;
    }

    // Close other graph.
    other.head->next_sibling = other.feet;
    other.feet->prev_sibling = other.head;

    return other_first_head;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
IterT
graph<T, AllocatorT>::move_in_as_nth_child(IterT loc, size_t n, graph& other)
{
    if(other.head->next_sibling == other.feet)
        return loc;  // other graph is empty

    graph_node* other_first_head = other.head->next_sibling;
    graph_node* other_last_head  = other.feet->prev_sibling;

    if(n == 0)
    {
        if(loc.node->first_child == nullptr)
        {
            loc.node->first_child          = other_first_head;
            loc.node->last_child           = other_last_head;
            other_last_head->next_sibling  = nullptr;
            other_first_head->prev_sibling = nullptr;
        }
        else
        {
            loc.node->first_child->prev_sibling = other_last_head;
            other_last_head->next_sibling       = loc.node->first_child;
            loc.node->first_child               = other_first_head;
            other_first_head->prev_sibling      = nullptr;
        }
    }
    else
    {
        --n;
        graph_node* walk = loc.node->first_child;
        while(true)
        {
            if(walk == nullptr)
            {
                throw std::range_error(
                    "graph: move_in_as_nth_child position out of range");
            }
            if(n == 0)
                break;
            --n;
            walk = walk->next_sibling;
        }
        if(walk->next_sibling == nullptr)
        {
            loc.node->last_child = other_last_head;
        }
        else
        {
            walk->next_sibling->prev_sibling = other_last_head;
        }
        other_last_head->next_sibling  = walk->next_sibling;
        walk->next_sibling             = other_first_head;
        other_first_head->prev_sibling = walk;
    }

    // Adjust parent pointers.
    graph_node* walk = other_first_head;
    while(true)
    {
        walk->parent = loc.node;
        if(walk == other_last_head)
            break;
        walk = walk->next_sibling;
    }

    // Close other graph.
    other.head->next_sibling = other.feet;
    other.feet->prev_sibling = other.head;

    return other_first_head;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::merge(const sibling_iterator& to1, const sibling_iterator& to2,
                            sibling_iterator from1, const sibling_iterator& from2,
                            bool duplicate_leaves, bool first)
{
    while(from1 != from2)
    {
        sibling_iterator    fnd;
        auto                nsiblings = number_of_siblings(to1);
        decltype(nsiblings) count     = nullptr;
        for(sibling_iterator itr = to1; itr != to2; ++itr, ++count)
        {
            if(itr && from1 && *itr == *from1)
            {
                fnd = itr;
                break;
            }
            if(count > nsiblings)
            {
                fnd = to2;
                break;
            }
        }
        // auto fnd = std::find(to1, to2, *from1);
        if(fnd != to2)  // element found
        {
            if(from1.begin() == from1.end())  // full depth reached
            {
                if(duplicate_leaves)
                    append_child(parent(to1), (*from1));
            }
            else  // descend further
            {
                if(!first)
                    *fnd += *from1;
                if(from1 != from2)
                {
                    merge(fnd.begin(), fnd.end(), from1.begin(), from1.end(),
                          duplicate_leaves);
                }
            }
        }
        else
        {  // element missing
            insert_subgraph(to2, from1);
        }
        do
        {
            ++from1;
        } while(!from1 && from1 != from2);
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename ComparePred, typename ReducePred>
void
graph<T, AllocatorT>::reduce(const sibling_iterator&     lhs, const sibling_iterator&,
                             std::set<sibling_iterator>& _erase, ComparePred&& _compare,
                             ReducePred&& _reduce)
{
    if(!is_valid(lhs))
        return;

    for(pre_order_iterator litr = lhs; litr != feet; ++litr)
    {
        if(!litr)
            continue;

        uint32_t nsiblings = number_of_siblings(litr);
        if(nsiblings < 2)
            continue;

        uint32_t idx = index(litr);
        for(uint32_t i = 0; i < nsiblings; ++i)
        {
            if(i == idx)
                continue;

            sibling_iterator ritr = sibling(litr, i);

            if(!ritr)
                continue;

            // skip if same iterator
            if(litr.node == ritr.node)
                continue;

            if(_erase.find(ritr) != _erase.end())
                continue;

            if(_compare(litr, ritr))
            {
                pre_order_iterator pritr(ritr);
                // printf("\n");
                // pre_order_iterator critr = pritr.begin();
                // auto aitr = append_child(litr, critr);
                auto aitr = insert_subgraph_after(litr, pritr);
                reduce(aitr.begin(), feet, _erase, _compare, _reduce);
                // insert_subgraph_after(litr, pritr);
                _erase.insert(ritr);
                reduce(litr.begin(), feet, _erase, _compare, _reduce);
                _reduce(litr, ritr);
                // this->erase(ritr);

                // break;
            }
        }

        for(auto& itr : _erase)
            this->erase(itr);

        if(_erase.size() > 0)
        {
            _erase.clear();
            break;
        }
        // reduce(litr.begin(), feet, _erase, _compare, _reduce);
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
bool
graph<T, AllocatorT>::equal(const IterT& one_, const IterT& two,
                            const IterT& three_) const
{
    std::equal_to<T> comp;
    return equal(one_, two, three_, comp);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT>
bool
graph<T, AllocatorT>::equal_subgraph(const IterT& one_, const IterT& two_) const
{
    std::equal_to<T> comp;
    return equal_subgraph(one_, two_, comp);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT, typename BinaryPredicate>
bool
graph<T, AllocatorT>::equal(const IterT& one_, const IterT& two, const IterT& three_,
                            BinaryPredicate fun) const
{
    pre_order_iterator one(one_);
    pre_order_iterator three(three_);

    //	if(one==two && is_valid(three) && three.number_of_children()!=0)
    //		return false;
    while(one != two && is_valid(three))
    {
        if(!fun(*one, *three))
            return false;
        if(one.number_of_children() != three.number_of_children())
            return false;
        ++one;
        ++three;
    }
    return true;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT, typename BinaryPredicate>
bool
graph<T, AllocatorT>::equal_subgraph(const IterT& one_, const IterT& two_,
                                     BinaryPredicate fun) const
{
    pre_order_iterator one(one_);
    pre_order_iterator two(two_);

    if(!fun(*one, *two))
        return false;
    if(number_of_children(one) != number_of_children(two))
        return false;
    return equal(begin(one), end(one), begin(two), fun);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::empty() const
{
    pre_order_iterator it  = begin();
    pre_order_iterator eit = end();
    return (it == eit);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
int
graph<T, AllocatorT>::depth(const iterator_base& it)
{
    graph_node* pos = it.node;
    assert(pos != nullptr);
    int ret = 0;
    while(pos->parent != nullptr)
    {
        pos = pos->parent;
        ++ret;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
int
graph<T, AllocatorT>::depth(const iterator_base& it, const iterator_base& root)
{
    graph_node* pos = it.node;
    assert(pos != nullptr);
    int ret = 0;
    while(pos->parent != nullptr && pos != root.node)
    {
        pos = pos->parent;
        ++ret;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
int
graph<T, AllocatorT>::max_depth() const
{
    int maxd = -1;
    for(graph_node* it = head->next_sibling; it != feet; it = it->next_sibling)
        maxd = std::max(maxd, max_depth(it));

    return maxd;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
int
graph<T, AllocatorT>::max_depth(const iterator_base& pos) const
{
    graph_node* tmp = pos.node;

    if(tmp == nullptr || tmp == head || tmp == feet)
        return -1;

    int curdepth = 0;
    int maxdepth = 0;
    while(true)
    {  // try to walk the bottom of the graph
        while(tmp->first_child == nullptr)
        {
            if(tmp == pos.node)
                return maxdepth;
            if(tmp->next_sibling == nullptr)
            {
                // try to walk up and then right again
                do
                {
                    tmp = tmp->parent;
                    if(tmp == nullptr)
                        return maxdepth;
                    --curdepth;
                } while(tmp->next_sibling == nullptr);
            }
            if(tmp == pos.node)
                return maxdepth;
            tmp = tmp->next_sibling;
        }
        tmp = tmp->first_child;
        ++curdepth;
        maxdepth = std::max(curdepth, maxdepth);
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
unsigned int
graph<T, AllocatorT>::number_of_children(const iterator_base& it)
{
    graph_node* pos = it.node->first_child;
    if(pos == nullptr)
        return 0;

    unsigned int ret = 1;
    while((pos = pos->next_sibling))
        ++ret;
    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
unsigned int
graph<T, AllocatorT>::number_of_siblings(const iterator_base& it) const
{
    graph_node*  pos = it.node;
    unsigned int ret = 0;
    // count forward
    while(pos->next_sibling && pos->next_sibling != head && pos->next_sibling != feet)
    {
        ++ret;
        pos = pos->next_sibling;
    }
    // count backward
    pos = it.node;
    while(pos->prev_sibling && pos->prev_sibling != head && pos->prev_sibling != feet)
    {
        ++ret;
        pos = pos->prev_sibling;
    }

    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::swap(sibling_iterator it)
{
    graph_node* nxt = it.node->next_sibling;
    if(nxt)
    {
        if(it.node->prev_sibling)
        {
            it.node->prev_sibling->next_sibling = nxt;
        }
        else
        {
            it.node->parent->first_child = nxt;
        }
        nxt->prev_sibling  = it.node->prev_sibling;
        graph_node* nxtnxt = nxt->next_sibling;
        if(nxtnxt)
        {
            nxtnxt->prev_sibling = it.node;
        }
        else
        {
            it.node->parent->last_child = it.node;
        }
        nxt->next_sibling     = it.node;
        it.node->prev_sibling = nxt;
        it.node->next_sibling = nxtnxt;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
void
graph<T, AllocatorT>::swap(iterator one, iterator two)
{
    // if one and two are adjacent siblings, use the sibling swap
    if(one.node->next_sibling == two.node)
    {
        swap(one);
    }
    else if(two.node->next_sibling == one.node)
    {
        swap(two);
    }
    else
    {
        graph_node* nxt1 = one.node->next_sibling;
        graph_node* nxt2 = two.node->next_sibling;
        graph_node* pre1 = one.node->prev_sibling;
        graph_node* pre2 = two.node->prev_sibling;
        graph_node* par1 = one.node->parent;
        graph_node* par2 = two.node->parent;

        // reconnect
        one.node->parent       = par2;
        one.node->next_sibling = nxt2;
        if(nxt2)
        {
            nxt2->prev_sibling = one.node;
        }
        else
        {
            par2->last_child = one.node;
        }
        one.node->prev_sibling = pre2;
        if(pre2)
        {
            pre2->next_sibling = one.node;
        }
        else
        {
            par2->first_child = one.node;
        }

        two.node->parent       = par1;
        two.node->next_sibling = nxt1;
        if(nxt1)
        {
            nxt1->prev_sibling = two.node;
        }
        else
        {
            par1->last_child = two.node;
        }
        two.node->prev_sibling = pre1;
        if(pre1)
        {
            pre1->next_sibling = two.node;
        }
        else
        {
            par1->first_child = two.node;
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
template <typename IterT, typename FuncT, typename ExecT, typename WaitT>
void
graph<T, AllocatorT>::sort(IterT _beg, IterT _end, FuncT&& _func, ExecT&& _exec,
                           WaitT&& _wait)
{
    auto _selection_sort = [&](IterT root) {
        long n   = number_of_children(root);
        long idx = 0;
        // One by one move boundary of unsorted subarray
        for(long i = 0; i < n - 1; ++i)
        {
            // Find the minimum element in unsorted array
            idx = i;
            for(long j = i + 1; j < n; ++j)
            {
                auto lhs = child(root, j);
                auto rhs = child(root, idx);
                if(_func(*lhs, *rhs))
                    idx = j;
            }
            // Swap the found minimum element with the first element
            auto lhs = child(root, idx);
            auto rhs = child(root, i);
            swap(lhs, rhs);
        }
    };

    for(IterT itr = _beg; itr != _end; ++itr)
        _exec([itr, &_selection_sort]() { _selection_sort(itr); });

    _wait();
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::is_valid(const iterator_base& it) const
{
    if(it.node == nullptr || it.node == feet || it.node == head)
    {
        return false;
    }
    {
        return true;
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::is_head(const iterator_base& it)
{
    if(it.node->parent == nullptr)
        return true;
    return false;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
inline unsigned int
graph<T, AllocatorT>::index(sibling_iterator it) const
{
    graph_node* tmp = it.node;
    if(!tmp)
        return static_cast<unsigned int>(-1);

    if(tmp->parent != nullptr)
    {
        tmp = tmp->parent->first_child;
    }
    else
    {
        while(tmp->prev_sibling != nullptr)
            tmp = tmp->prev_sibling;
    }

    unsigned int ret = 0;
    while(tmp != it.node)
    {
        ++ret;
        tmp = tmp->next_sibling;
    }

    return ret;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::sibling(const iterator_base& it, unsigned int num) const
{
    graph_node* tmp = it.node;
    if(!tmp)
        return sibling_iterator(nullptr);

    if(tmp->parent != nullptr)
    {
        tmp = tmp->parent->first_child;
    }
    else
    {
        while(tmp->prev_sibling != nullptr)
            tmp = tmp->prev_sibling;
    }

    while((num--) != 0u)
    {
        assert(tmp != nullptr);
        tmp = tmp->next_sibling;
    }
    return tmp;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::child(const iterator_base& itr, unsigned int _num)
{
    if(!itr.node)
        return sibling_iterator{ nullptr };

    graph_node* _v = itr.node->first_child;
    for(unsigned int i = 0; i < _num; ++i)
    {
        if(!_v || _v == itr.node->last_child)
            return sibling_iterator{ nullptr };
        _v = _v->next_sibling;
    }
    return sibling_iterator{ _v };
}

//--------------------------------------------------------------------------------------//
// Iterator base

template <typename T, typename AllocatorT>
T& graph<T, AllocatorT>::iterator_base::operator*() const
{
    return node->data;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
T* graph<T, AllocatorT>::iterator_base::operator->() const
{
    return &(node->data);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::pre_order_iterator::operator==(
    const pre_order_iterator& other) const
{
    return (other.node == this->node);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::pre_order_iterator::operator!=(
    const pre_order_iterator& other) const
{
    return !(*this == other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::sibling_iterator::operator!=(const sibling_iterator& other) const
{
    return !(*this == other);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
bool
graph<T, AllocatorT>::sibling_iterator::operator==(const sibling_iterator& other) const
{
    return (other.node == this->node);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::iterator_base::begin() const
{
    if(node->first_child == nullptr)
        return end();

    return sibling_iterator{ node->first_child };
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::iterator_base::end() const
{
    return sibling_iterator{ nullptr };
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
unsigned int
graph<T, AllocatorT>::iterator_base::number_of_children() const
{
    graph_node* pos = node->first_child;
    if(pos == nullptr)
        return 0;

    unsigned int ret = 1;
    while(pos != node->last_child)
    {
        ++ret;
        pos = pos->next_sibling;
    }
    return ret;
}

//--------------------------------------------------------------------------------------//
// Pre-order iterator

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::pre_order_iterator::pre_order_iterator()
: iterator_base(nullptr)
{}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::pre_order_iterator::pre_order_iterator(graph_node* tn)
: iterator_base(tn)
{}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::pre_order_iterator::pre_order_iterator(const iterator_base& other)
: iterator_base(other.node)
{}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::pre_order_iterator::pre_order_iterator(
    const sibling_iterator& other)
: iterator_base(other.node)
{
    if(this->node == nullptr)
    {
        if(other.range_last() != nullptr)
        {
            this->node = other.range_last();
        }
        else
        {
            this->node = other.parent();
        }
        ++(*this);
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator&
graph<T, AllocatorT>::pre_order_iterator::operator++()
{
    assert(this->node != nullptr);
    // if(!this->m_skip_current_children && this->node->first_child != nullptr)
    if(this->node->first_child != nullptr)
    {
        this->node = this->node->first_child;
    }
    else
    {
        // this->m_skip_current_children = false;
        while(this->node->next_sibling == nullptr)
        {
            this->node = this->node->parent;
            if(this->node == nullptr)
                return *this;
        }
        this->node = this->node->next_sibling;
    }
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator&
graph<T, AllocatorT>::pre_order_iterator::operator--()
{
    assert(this->node != nullptr);
    if(this->node->prev_sibling)
    {
        this->node = this->node->prev_sibling;
        while(this->node->last_child)
            this->node = this->node->last_child;
    }
    else
    {
        this->node = this->node->parent;
        if(this->node == nullptr)
            return *this;
    }
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::pre_order_iterator::operator++(int)
{
    pre_order_iterator copy = *this;
    ++(*this);
    return copy;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::pre_order_iterator::operator--(int)
{
    pre_order_iterator copy = *this;
    --(*this);
    return copy;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator&
graph<T, AllocatorT>::pre_order_iterator::operator+=(unsigned int num)
{
    for(unsigned int i = 0; i < num; ++i)
        ++(*this);
    return (*this);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator&
graph<T, AllocatorT>::pre_order_iterator::operator-=(unsigned int num)
{
    for(unsigned int i = 0; i < num; ++i)
        --(*this);
    return (*this);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::pre_order_iterator::operator+(unsigned int num)
{
    auto itr = *this;
    for(unsigned int i = 0; i < num; ++i)
        ++itr;
    return itr;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::pre_order_iterator
graph<T, AllocatorT>::pre_order_iterator::operator-(unsigned int num)
{
    auto itr = *this;
    for(unsigned int i = 0; i < num; ++i)
        --itr;
    return itr;
}

//--------------------------------------------------------------------------------------//
// Sibling iterator
template <typename T, typename AllocatorT>
graph<T, AllocatorT>::sibling_iterator::sibling_iterator()
: iterator_base{}
{}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::sibling_iterator::sibling_iterator(graph_node* tn)
: iterator_base{ tn }
{}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
graph<T, AllocatorT>::sibling_iterator::sibling_iterator(const iterator_base& other)
: iterator_base{ other.node }
{}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::graph_node*
graph<T, AllocatorT>::iterator_base::parent() const
{
    if(this->node == nullptr)
        return nullptr;
    return this->node->parent;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator&
graph<T, AllocatorT>::sibling_iterator::operator++()
{
    if(this->node)
        this->node = this->node->next_sibling;
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator&
graph<T, AllocatorT>::sibling_iterator::operator--()
{
    if(this->node)
    {
        this->node = this->node->prev_sibling;
    }
    else
    {
        auto _parent = this->parent();
        this->node   = (_parent) ? _parent->last_child : nullptr;
    }
    return *this;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::sibling_iterator::operator++(int)
{
    sibling_iterator copy = *this;
    ++(*this);
    return copy;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::sibling_iterator::operator--(int)
{
    sibling_iterator copy = *this;
    --(*this);
    return copy;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator&
graph<T, AllocatorT>::sibling_iterator::operator+=(unsigned int num)
{
    for(unsigned int i = 0; i < num; ++i)
        ++(*this);
    return (*this);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator&
graph<T, AllocatorT>::sibling_iterator::operator-=(unsigned int num)
{
    for(unsigned int i = 0; i < num; ++i)
        --(*this);
    return (*this);
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::sibling_iterator::operator+(unsigned int num)
{
    auto itr = *this;
    for(unsigned int i = 0; i < num; ++i)
        ++itr;
    return itr;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::sibling_iterator
graph<T, AllocatorT>::sibling_iterator::operator-(unsigned int num)
{
    auto itr = *this;
    for(unsigned int i = 0; i < num; ++i)
        --itr;
    return itr;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
typename graph<T, AllocatorT>::graph_node*
graph<T, AllocatorT>::sibling_iterator::range_last() const
{
    return (parent()) ? parent()->last_child : nullptr;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename AllocatorT>
size_t
graph<T, AllocatorT>::size() const
{
    size_t             i   = 0;
    pre_order_iterator bit = begin();
    pre_order_iterator eit = end();
    pre_order_iterator itr = bit;
    while(itr != eit)
    {
        ++i;
        ++itr;
    }
    return i;
}

//--------------------------------------------------------------------------------------//

template <typename T>
void
print_graph_bracketed(const graph<T>& t, std::ostream& os = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T>
void
print_subgraph_bracketed(const graph<T>& t, typename graph<T>::iterator root,
                         std::ostream& os = std::cout);

//--------------------------------------------------------------------------------------//
// Iterate over all roots (the head) and print each one on a new line
// by calling printSingleRoot.
//--------------------------------------------------------------------------------------//

template <typename T>
void
print_graph_bracketed(const graph<T>& t, std::ostream& os)
{
    int head_count = t.number_of_siblings(t.begin());
    int nhead      = 0;
    for(typename graph<T>::sibling_iterator ritr = t.begin(); ritr != t.end();
        ++ritr, ++nhead)
    {
        print_subgraph_bracketed(t, ritr, os);
        if(nhead != head_count)
        {
            os << std::endl;
        }
    }
}

//--------------------------------------------------------------------------------------//
// Print everything under this root in a flat, bracketed structure.
//--------------------------------------------------------------------------------------//

template <typename T>
void
print_subgraph_bracketed(const graph<T>& t, typename graph<T>::iterator root,
                         std::ostream& os)
{
    static int _depth = 0;
    if(t.empty())
        return;

    auto        m_depth = _depth++;
    std::string indent  = {};
    for(int i = 0; i < m_depth; ++i)
        indent += "  ";

    if(t.number_of_children(root) == 0)
    {
        os << "\n" << indent << *root;
    }
    else
    {
        // parent
        os << "\n" << indent << *root;
        os << "(";
        // child1, ..., childn
        int sibling_count = t.number_of_siblings(t.begin(root));
        int nsiblings;
        typename graph<T>::sibling_iterator children;
        for(children = t.begin(root), nsiblings = 0; children != t.end(root);
            ++children, ++nsiblings)
        {
            // recursively print child
            print_subgraph_bracketed(t, children, os);
            // comma after every child except the last one
            if(nsiblings != sibling_count)
            {
                os << ", ";
            }
        }
        os << ")";
    }
    --_depth;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter = std::function<std::string(const T&)>>
void
print_graph(const tim::graph<T>& t, Formatter format, std::ostream& str = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter = std::function<std::string(const T&)>>
void
print_subgraph(const tim::graph<T>& t, Formatter format,
               typename tim::graph<T>::iterator root, std::ostream& str = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter>
void
print_graph(const tim::graph<T>& t, Formatter format, std::ostream& str)
{
    int head_count = t.number_of_siblings(t.begin());
    int nhead      = 0;
    for(typename tim::graph<T>::sibling_iterator ritr = t.begin(); ritr != t.end();
        ++ritr, ++nhead)
    {
        print_subgraph(t, format, ritr, str);
        if(nhead != head_count)
        {
            str << std::endl;
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T>
void
print_graph(const tim::graph<T>& t, std::ostream& str)
{
    auto _formatter = [](const T& obj) {
        std::stringstream ss;
        ss << obj;
        return ss.str();
    };
    int head_count = t.number_of_siblings(t.begin());
    int nhead      = 0;
    for(typename tim::graph<T>::sibling_iterator ritr = t.begin(); ritr != t.end();
        ++ritr, ++nhead)
    {
        print_subgraph(t, _formatter, ritr, str);
        if(nhead != head_count)
        {
            str << std::endl;
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter>
void
print_subgraph(const tim::graph<T>& t, Formatter format,
               typename tim::graph<T>::iterator root, std::ostream& os)
{
    if(t.empty())
        return;
    if(t.number_of_children(root) == 0)
    {
        os << format(*root);
    }
    else
    {
        // parent
        std::string str = format(*root);
        if(str.length() > 0)
            os << str << "\n";
        // child1, ..., childn
        int sibling_count = t.number_of_siblings(t.begin(root));
        int nsiblings;
        typename tim::graph<T>::sibling_iterator children;
        for(children = t.begin(root), nsiblings = 0; children != t.end(root);
            ++children, ++nsiblings)
        {
            // recursively print child
            print_subgraph(t, format, children, os);
            // comma after every child except the last one
            if(nsiblings != sibling_count)
            {
                os << "\n";
            }
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter = std::function<std::string(const T&)>>
void
print_graph_hierarchy(const tim::graph<T>& t, Formatter format,
                      std::ostream& str = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter = std::function<std::string(const T&)>>
void
print_subgraph_hierarchy(const tim::graph<T>& t, Formatter format,
                         typename tim::graph<T>::iterator root,
                         std::ostream&                    str = std::cout);

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter>
void
print_graph_hierarchy(const tim::graph<T>& t, Formatter format, std::ostream& str)
{
    int head_count = t.number_of_siblings(t.begin());
    int nhead      = 0;
    for(typename tim::graph<T>::sibling_iterator ritr = t.begin(); ritr != t.end();
        ++ritr, ++nhead)
    {
        print_subgraph_hierarchy(t, format, ritr, str);
        if(nhead != head_count)
        {
            str << std::endl;
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename T, typename Formatter>
void
print_subgraph_hierarchy(const tim::graph<T>& t, Formatter format,
                         typename tim::graph<T>::iterator root, std::ostream& os)
{
    if(t.empty())
        return;
    if(t.number_of_children(root) == 0)
    {
        os << format(*root);
    }
    else
    {
        // parent
        std::string str = format(*root);
        if(str.length() > 0)
            os << str << "\n" << std::setw(2 * (t.depth(root) + 1)) << "|_";
        // child1, ..., childn
        int sibling_count = t.number_of_siblings(t.begin(root));
        int nsiblings;
        typename tim::graph<T>::sibling_iterator children;
        for(children = t.begin(root), nsiblings = 0; children != t.end(root);
            ++children, ++nsiblings)
        {
            // recursively print child
            print_subgraph_hierarchy(t, format, children, os);
            // comma after every child except the last one
            if(nsiblings != sibling_count)
            {
                os << "\n" << std::setw(2 * (t.depth(root) + 1)) << "|_";
            }
        }
    }
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
