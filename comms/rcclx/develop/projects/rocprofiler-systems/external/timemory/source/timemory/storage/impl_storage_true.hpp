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

#include "timemory/hash/declaration.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/cleanup.hpp"
#include "timemory/operations/types/set.hpp"
#include "timemory/storage/base_storage.hpp"
#include "timemory/storage/graph.hpp"
#include "timemory/storage/graph_data.hpp"
#include "timemory/storage/impl_storage_deleter.hpp"
#include "timemory/storage/macros.hpp"
#include "timemory/storage/node.hpp"
#include "timemory/storage/storage_singleton.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace tim
{
namespace impl
{
template <typename Type>
class storage<Type, true> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    static constexpr bool has_data_v = true;

    template <typename KeyT, typename MappedT>
    using uomap_t = std::unordered_map<KeyT, MappedT>;

    using result_node    = node::result<Type>;
    using graph_node     = node::graph<Type>;
    using strvector_t    = std::vector<string_t>;
    using uintvector_t   = std::vector<hash_value_t>;
    using EmptyT         = std::tuple<>;
    using base_type      = base::storage;
    using component_type = Type;
    using this_type      = storage<Type, has_data_v>;
    using smart_pointer  = std::unique_ptr<this_type, impl::storage_deleter<this_type>>;
    using singleton_t    = singleton<this_type, smart_pointer>;
    using singleton_type = singleton_t;
    using pointer        = typename singleton_t::pointer;
    using auto_lock_t    = typename singleton_t::auto_lock_t;
    using node_type      = typename node::data<Type>::node_type;
    using stats_type     = typename node::data<Type>::stats_type;
    using result_type    = typename node::data<Type>::result_type;
    using result_array_t = std::vector<result_node>;
    using dmp_result_t   = std::vector<result_array_t>;
    using printer_t      = operation::finalize::print<Type, has_data_v>;
    using sample_array_t = std::vector<Type>;
    using graph_node_t   = graph_node;
    using graph_data_t   = graph_data<graph_node_t>;
    using graph_t        = typename graph_data_t::graph_t;
    using graph_type     = graph_t;
    using iterator       = typename graph_type::iterator;
    using const_iterator = typename graph_type::const_iterator;

    template <typename Vp>
    using secondary_data_t       = std::tuple<iterator, const std::string&, Vp>;
    using iterator_hash_submap_t = uomap_t<int64_t, iterator>;
    using iterator_hash_map_t    = uomap_t<int64_t, iterator_hash_submap_t>;

    friend class tim::manager;
    friend struct node::result<Type>;
    friend struct node::graph<Type>;
    friend struct impl::storage_deleter<this_type>;
    friend struct operation::finalize::get<Type, has_data_v>;
    friend struct operation::finalize::mpi_get<Type, has_data_v>;
    friend struct operation::finalize::upc_get<Type, has_data_v>;
    friend struct operation::finalize::dmp_get<Type, has_data_v>;
    friend struct operation::finalize::print<Type, has_data_v>;
    friend struct operation::finalize::merge<Type, has_data_v>;

public:
    // static functions
    static pointer instance();
    static pointer master_instance();
    static pointer noninit_instance();
    static pointer noninit_master_instance();

    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static singleton_t* get_singleton() { return get_storage_singleton<this_type>(); }
    static std::atomic<int64_t>& instance_count();

public:
public:
    storage();
    ~storage() override;

    storage(standalone_storage, int64_t _instance_id, std::string _label);

    storage(const this_type&) = delete;
    storage(this_type&&)      = delete;

    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

public:
    void get_shared_manager();

    void write(std::string = {}) final;
    void print() final { internal_print(); }
    void cleanup() final { operation::cleanup<Type>{}; }
    void disable() final { trait::runtime_enabled<component_type>::set(false); }
    void initialize() final;
    void finalize() final;
    void stack_clear() final;
    bool global_init() final;
    bool thread_init() final;
    bool data_init() final;
    void destroy() final;
    void deregister() final;

    const graph_data_t& data() const;
    const graph_t&      graph() const;
    int64_t             depth() const;
    graph_data_t&       data();
    graph_t&            graph();
    iterator&           current();

    void        reset();
    inline bool empty() const
    {
        return (m_graph_data_instance) ? (_data().graph().size() <= 1) : true;
    }
    inline size_t size() const
    {
        return (m_graph_data_instance) ? (_data().graph().size() - 1) : 0;
    }
    inline size_t true_size() const
    {
        if(!m_graph_data_instance)
            return 0;
        size_t _sz = _data().graph().size();
        size_t _dc = _data().dummy_count();
        return (_dc < _sz) ? (_sz - _dc) : 0;
    }
    iterator       pop();
    result_array_t get();
    dmp_result_t   mpi_get();
    dmp_result_t   upc_get();
    dmp_result_t   dmp_get();

    template <typename Tp>
    Tp& get(Tp&);
    template <typename Tp>
    Tp& mpi_get(Tp&);
    template <typename Tp>
    Tp& upc_get(Tp&);
    template <typename Tp>
    Tp& dmp_get(Tp&);

    std::shared_ptr<printer_t> get_printer() const { return m_printer; }

    // don't expose this
    // iterator_hash_map_t get_node_ids() const { return m_node_ids; }

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj);

    void ensure_init();

    iterator insert(scope::config scope_data, const Type& obj, uint64_t hash_id,
                    int64_t _tid = -1);

    // append a value to the the graph
    template <typename Vp, enable_if_t<!std::is_same<decay_t<Vp>, Type>::value, int> = 0>
    iterator append(const secondary_data_t<Vp>& _secondary);

    // append an instance to the graph
    template <typename Vp, enable_if_t<std::is_same<decay_t<Vp>, Type>::value, int> = 0>
    iterator append(const secondary_data_t<Vp>& _secondary);

    template <typename Archive>
    void serialize(Archive& ar, unsigned int version);

    void add_sample(Type&& _obj) { m_samples.emplace_back(std::forward<Type>(_obj)); }

    auto&       get_samples() { return m_samples; }
    const auto& get_samples() const { return m_samples; }

protected:
    iterator insert_tree(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                         int64_t _tid);
    iterator insert_timeline(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                             int64_t _tid);
    iterator insert_flat(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                         int64_t _tid);
    iterator insert_hierarchy(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                              bool has_head, int64_t _tid);

    void     merge();
    void     merge(this_type* itr);
    string_t get_prefix(const graph_node&);
    string_t get_prefix(iterator _node) { return get_prefix(*_node); }
    string_t get_prefix(const uint64_t& _id);

private:
    template <typename Archive>
    void do_serialize(Archive& ar);

    void internal_print();

    graph_data_t&       _data();
    const graph_data_t& _data() const
    {
        using type_t = decay_t<remove_pointer_t<decltype(this)>>;
        return const_cast<type_t*>(this)->_data();
    }

private:
    uint64_t                   m_timeline_counter    = 1;
    mutable graph_data_t*      m_graph_data_instance = nullptr;
    std::shared_ptr<printer_t> m_printer             = {};
    iterator_hash_map_t        m_node_ids            = {};
    std::unordered_set<Type*>  m_stack               = {};
    sample_array_t             m_samples             = {};
};
}  // namespace impl
}  // namespace tim
