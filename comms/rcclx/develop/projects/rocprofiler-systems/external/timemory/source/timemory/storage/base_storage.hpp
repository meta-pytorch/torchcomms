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

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/storage/macros.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

namespace tim
{
namespace base
{
class storage : public data_cleanup
{
public:
    using string_t       = std::string;
    using this_type      = storage;
    using children_map_t = std::map<int64_t, std::set<storage*>>;

public:
    storage(bool _is_master, int64_t _instance_id, std::string _label);
    storage(standalone_storage, int64_t _instance_id, std::string _label);
    ~storage() override = default;

    explicit storage(const storage&) = delete;
    explicit storage(storage&&)      = delete;
    storage& operator=(const storage&) = delete;
    storage& operator=(storage&& rhs) = delete;

    virtual void write(std::string = {}) {}  // NOLINT
    virtual void print() {}
    virtual void cleanup() {}
    virtual void stack_clear() {}
    virtual void disable() {}
    virtual void initialize() {}
    virtual void finalize() {}
    virtual bool global_init() { return false; }
    virtual bool thread_init() { return false; }
    virtual bool data_init() { return false; }
    void         destroy() override    = 0;
    void         deregister() override = 0;

    template <typename Tp, typename Vp>
    static this_type* base_instance();

public:
    const hash_map_ptr_t&   get_hash_ids() const { return m_hash_ids; }
    const hash_alias_ptr_t& get_hash_aliases() const { return m_hash_aliases; }

    hash_value_t add_hash_id(const std::string& _prefix);
    void         add_hash_id(uint64_t _lhs, uint64_t _rhs);

    bool    is_initialized() const { return m_initialized; }
    int64_t instance_id() const { return m_instance_id; }
    void    free_shared_manager();

    void               add_child(storage*, int64_t _tid = -1);
    void               remove_child(storage*, int64_t _tid = -1);
    std::set<storage*> get_children(int64_t _tid = -1);

protected:
    void add_file_output(const string_t& _category, const string_t& _label,
                         const string_t& _file);
    void add_text_output(const string_t& _label, const string_t& _file)
    {
        add_file_output("text", _label, _file);
    }
    void add_json_output(const string_t& _label, const string_t& _file)
    {
        add_file_output("json", _label, _file);
    }

    static std::atomic<int>& storage_once_flag();
    static void              stop_profiler();

protected:
    bool                      m_initialized  = false;                     // NOLINT
    bool                      m_finalized    = false;                     // NOLINT
    bool                      m_global_init  = false;                     // NOLINT
    bool                      m_thread_init  = false;                     // NOLINT
    bool                      m_data_init    = false;                     // NOLINT
    bool                      m_is_master    = false;                     // NOLINT
    bool                      m_standalone   = false;                     // NOLINT
    bool                      m_node_init    = dmp::is_initialized();     // NOLINT
    int32_t                   m_node_rank    = dmp::rank();               // NOLINT
    int32_t                   m_node_size    = dmp::size();               // NOLINT
    uint32_t                  m_thread_idx   = threading::get_id();       // NOLINT
    int64_t                   m_instance_id  = -1;                        // NOLINT
    string_t                  m_label        = {};                        // NOLINT
    mutex_t                   m_mutex        = {};                        // NOLINT
    hash_map_ptr_t            m_hash_ids     = hash::get_hash_ids();      // NOLINT
    hash_alias_ptr_t          m_hash_aliases = hash::get_hash_aliases();  // NOLINT
    std::shared_ptr<manager>  m_manager      = {};                        // NOLINT
    std::shared_ptr<settings> m_settings     = {};                        // NOLINT
    children_map_t            m_children     = {};                        // NOLINT
};
//
template <typename Tp, typename Vp>
inline storage*
storage::base_instance()
{
    using storage_type = tim::storage<Tp, Vp>;

    // thread-local variable
    static thread_local base::storage* _ret = nullptr;

    // return nullptr is disabled
    if(!trait::runtime_enabled<Tp>::get())
        return nullptr;

    // if nullptr, try to get instance
    if(_ret == nullptr)
    {
        // thread will copy the hash-table so use a lock here
        auto_lock_t _lk{ type_mutex<base::storage>(), std::defer_lock };
        if(!locking::try_lock_for_n(_lk))
            TIMEMORY_PRINT_HERE("%s", "failed to acquire base::storage lock. Potential "
                                      "data-race for hash-table may cause issues");
        _ret = static_cast<base::storage*>(storage_type::instance());
    }

    // return pointer
    return _ret;
}
//
}  // namespace base
}  // namespace tim

#if defined(TIMEMORY_STORAGE_HEADER_MODE)
#    include "timemory/storage/base_storage.cpp"
#endif
