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
#include "timemory/mpl/type_traits.hpp"
#include "timemory/operations/types/cleanup.hpp"
#include "timemory/operations/types/finalize/dmp_get.hpp"
#include "timemory/operations/types/finalize/get.hpp"
#include "timemory/operations/types/finalize/merge.hpp"
#include "timemory/operations/types/finalize/mpi_get.hpp"
#include "timemory/operations/types/finalize/upc_get.hpp"
#include "timemory/storage/base_storage.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/singleton.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace tim
{
namespace impl
{
//
//--------------------------------------------------------------------------------------//
//
//                      impl::storage<Type, false>
//                          impl::storage_false
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
class storage<Type, false> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    static constexpr bool has_data_v = false;

    using result_node    = std::tuple<>;
    using graph_node     = std::tuple<>;
    using graph_t        = std::tuple<>;
    using graph_type     = graph_t;
    using dmp_result_t   = std::vector<std::tuple<>>;
    using result_array_t = std::vector<std::tuple<>>;
    using uintvector_t   = std::vector<hash_value_t>;
    using base_type      = base::storage;
    using component_type = Type;
    using this_type      = storage<Type, has_data_v>;
    using string_t       = std::string;
    using smart_pointer  = std::unique_ptr<this_type, impl::storage_deleter<this_type>>;
    using singleton_t    = singleton<this_type, smart_pointer>;
    using singleton_type = singleton_t;
    using pointer        = typename singleton_t::pointer;
    using auto_lock_t    = typename singleton_t::auto_lock_t;
    using printer_t      = operation::finalize::print<Type, has_data_v>;

    using iterator       = void*;
    using const_iterator = const void*;

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
    storage();
    ~storage() override;

    storage(standalone_storage, int64_t _instance_id, std::string _label);

    explicit storage(const this_type&) = delete;
    explicit storage(this_type&&)      = delete;
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

    void print() final { finalize(); }
    void cleanup() final { operation::cleanup<Type>{}; }
    void stack_clear() final;
    void disable() final { trait::runtime_enabled<component_type>::set(false); }

    void initialize() final;
    void finalize() final;
    void destroy() final;
    void deregister() final;

    void                             reset() {}
    TIMEMORY_NODISCARD bool          empty() const { return true; }
    TIMEMORY_NODISCARD inline size_t size() const { return 0; }
    TIMEMORY_NODISCARD inline size_t true_size() const { return 0; }
    TIMEMORY_NODISCARD inline size_t depth() const { return 0; }

    iterator pop() { return nullptr; }
    iterator insert(int64_t, const Type&, const string_t&) { return nullptr; }

    template <typename Archive>
    void serialize(Archive&, const unsigned int)
    {}

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj);

    TIMEMORY_NODISCARD std::shared_ptr<printer_t> get_printer() const
    {
        return m_printer;
    }

protected:
    void get_shared_manager();
    void merge();
    void merge(this_type* itr);

private:
    template <typename Archive>
    void do_serialize(Archive&)
    {}

private:
    std::unordered_set<Type*>  m_stack   = {};
    std::shared_ptr<printer_t> m_printer = {};
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::instance()
{
    return get_singleton() ? get_singleton()->instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::master_instance()
{
    return get_singleton() ? get_singleton()->master_instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
}  // namespace tim
