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
 * \file timemory/manager/declaration.hpp
 * \brief The declaration for the types for manager without definitions
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/manager/macros.hpp"
#include "timemory/manager/types.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              manager
//
//--------------------------------------------------------------------------------------//
//
class TIMEMORY_VISIBILITY("default") manager
{
    template <typename... Args>
    using uomap_t = std::unordered_map<Args...>;

public:
    using this_type          = manager;
    using pointer_t          = std::shared_ptr<this_type>;
    using pointer_pair_t     = std::pair<pointer_t, pointer_t>;
    using size_type          = std::size_t;
    using string_t           = std::string;
    using comm_group_t       = std::tuple<mpi::comm_t, int32_t>;
    using mutex_t            = std::recursive_mutex;
    using auto_lock_t        = std::unique_lock<mutex_t>;
    using initializer_func_t = std::function<bool()>;
    using initializer_list_t = std::deque<initializer_func_t>;
    using finalizer_func_t   = std::function<void()>;
    using finalizer_plist_t  = std::deque<std::pair<std::string, finalizer_func_t>>;
    using finalizer_slist_t  = std::deque<std::pair<std::string, base::data_cleanup*>>;
    using finalizer_pmap_t   = std::map<int32_t, finalizer_plist_t>;
    using finalizer_smap_t   = std::map<int32_t, finalizer_slist_t>;
    using synchronize_list_t = uomap_t<string_t, uomap_t<int64_t, std::function<void()>>>;
    using finalizer_void_t   = std::multimap<void*, finalizer_func_t>;
    using settings_ptr_t     = std::weak_ptr<settings>;
    using filemap_t          = std::map<string_t, std::map<string_t, std::set<string_t>>>;
    using metadata_func_t    = std::vector<std::function<void(void*)>>;
    using metadata_info_t    = std::multimap<string_t, string_t>;
    using enum_set_t         = std::set<TIMEMORY_COMPONENT>;
    template <typename Tp>
    using enum_map_t = std::map<TIMEMORY_COMPONENT, Tp>;

public:
    // Constructor and Destructors
    manager();
    ~manager();

    manager(const manager&) = delete;
    manager(manager&&)      = delete;

    manager& operator=(const manager&) = delete;
    manager& operator=(manager&&) = delete;

    /// add functors to destroy instances based on a pointer
    template <typename Func>
    void add_cleanup(void*, Func&&);
    /// add functors to destroy instances based on a string key
    template <typename Func>
    void add_cleanup(const std::string&, Func&&);
    /// this is used for initialization
    template <typename InitFuncT>
    void add_initializer(InitFuncT&&);
    /// this is used by storage classes for finalization.
    void add_finalizer(const std::string&, base::data_cleanup*, bool, int32_t = 0);
    /// remove a cleanup functor
    void remove_cleanup(void*);
    /// remove a cleanup functor
    void remove_cleanup(const std::string&);
    /// remove a finalizer functor
    std::vector<finalizer_func_t> remove_finalizer(const std::string&);
    /// execute a cleanup based on a key
    void cleanup(const std::string&);
    void cleanup();
    void initialize();
    void finalize();
    bool is_initialized() const { return m_is_initialized; }
    bool is_finalized() const { return m_is_finalized; }

    void add_file_output(const string_t& _category, const string_t& _label,
                         const string_t& _file);
    void add_text_output(const string_t& _label, const string_t& _file);
    void add_json_output(const string_t& _label, const string_t& _file);

    /// Set to 0 for yes if other output, -1 for never, or 1 for yes
    void set_write_metadata(short v) { m_write_metadata = v; }
    /// Print metadata to filename
    void write_metadata(const std::string&, const char* = "",
                        settings::compose_filename_config _cfg = {});
    /// Write metadata to ostream
    std::ostream& write_metadata(std::ostream&);
    /// Updates settings, rank, output prefix, etc.
    void update_metadata_prefix();
    /// Get the dmp rank. This is stored to avoid having to do MPI/UPC++ query after
    /// finalization has been called
    int32_t get_rank() const { return m_rank; }
    /// Query whether finalization is currently occurring
    bool is_finalizing() const { return m_is_finalizing; }
    /// Sets whether finalization is currently occuring
    void is_finalizing(bool v) { m_is_finalizing = v; }
    /// Add number of component output data entries. If this value is zero, metadata
    /// output is suppressed unless \ref tim::manager::set_write_metadata was assigned a
    /// value of 1
    void add_entries(uint64_t n) { m_num_entries += n; }

    /// Add function for synchronizing data in threads
    void add_synchronization(const std::string&, int64_t, std::function<void()>);
    /// Remove function for synchronizing data in threads
    void remove_synchronization(const std::string&, int64_t);
    /// Synchronizes thread-data for storage
    void synchronize();

public:
    /// Get a shared pointer to the instance for the current thread
    static pointer_t instance() TIMEMORY_VISIBILITY("default");
    /// Get a shared pointer to the instance on the primary thread
    static pointer_t master_instance() TIMEMORY_VISIBILITY("default");
    /// Get the number of instances that are currently allocated. This is decremented
    /// during the destructor of each manager instance, unlike \ref
    /// tim::manager::get_thread_count()
    static int32_t total_instance_count() { return f_manager_instance_count().load(); }
    /// Enable setting std::exit callback
    static void use_exit_hook(bool val) { f_use_exit_hook() = val; }
    /// The exit hook function
    static void exit_hook();
    /// This effectively provides the total number of threads which collected data.
    /// It is only "decremented" when the last manager instance has been deleted, at which
    /// point it is set to zero.
    static int32_t get_thread_count() { return f_thread_counter().load(); }
    /// Return whether this is the main thread
    static bool get_is_main_thread();

    /// Add a metadata entry of a non-string type. If this fails to serialize, either
    /// include either the approiate header from timemory/tpls/cereal/cereal/types or
    /// provides a serialization function for the type.
    template <typename Tp>
    static void add_metadata(const std::string&, const Tp&);
    template <typename FuncT>
    static auto add_metadata(FuncT&& _func)
        -> decltype(_func(std::declval<cereal::PrettyJSONOutputArchive&>()), void());
    /// Add a metadata entry of a const character array. This only exists to avoid
    /// the template function from serializing the pointer.
    static void add_metadata(const std::string&, const char*);
    /// Add a metadata entry of a string
    static void add_metadata(const std::string&, const std::string&);

private:
    template <typename Tp>
    void do_init_storage();

    template <typename Tp>
    void do_print_storage(const enum_set_t& = {});

    template <typename Tp>
    void do_clear(const enum_set_t& = {});

    template <typename Arch, typename Tp>
    void do_serialize(Arch& ar, const enum_set_t& = {});

    template <typename Tp>
    void do_size(uint64_t& _sz);  // total size for all components

    template <typename Tp>
    void do_size(enum_map_t<uint64_t>& _sz);  // size for individual components

    //----------------------------------------------------------------------------------//
    // used to expand a tuple in settings
    //
    template <typename... Types>
    struct filtered_get_storage
    {
        static void        initialize(pointer_t _manager = {});
        static std::string serialize(pointer_t _manager = {}, const enum_set_t& = {});
        static void        clear(pointer_t _manager = {}, const enum_set_t& = {});
        static void        print(pointer_t _manager = {}, const enum_set_t& = {});
        static uint64_t    size(pointer_t _manager = {});
        static enum_map_t<uint64_t> size(pointer_t _manager, const enum_set_t&);

        static std::string serialize(const enum_set_t& _types)
        {
            return serialize(pointer_t{ nullptr }, _types);
        }

        static void clear(const enum_set_t& _types)
        {
            clear(pointer_t{ nullptr }, _types);
        }

        static void print(const enum_set_t& _types)
        {
            print(pointer_t{ nullptr }, _types);
        }

        static enum_map_t<uint64_t> size(const enum_set_t& _types)
        {
            return size(pointer_t{ nullptr }, _types);
        }
    };

    //----------------------------------------------------------------------------------//
    //
    template <template <typename...> class Tuple, typename... Types>
    struct filtered_get_storage<Tuple<Types...>> : public filtered_get_storage<Types...>
    {
        using base_type = filtered_get_storage<Types...>;
        using base_type::clear;
        using base_type::initialize;
        using base_type::print;
        using base_type::serialize;
        using base_type::size;
    };

public:
    //----------------------------------------------------------------------------------//
    /// This is used to apply/query storage data for multiple component types.
    ///
    /// \code{.cpp}
    /// using types = tim::available_types_t;   // type-list of all enumerated types
    /// manager_t::get_storage<types>::clear(); // clear storage for all enumerated types
    /// \endcode
    template <typename... Types>
    struct get_storage : public filtered_get_storage<mpl::implemented_t<Types...>>
    {
        using base_type = filtered_get_storage<mpl::implemented_t<Types...>>;
        using base_type::clear;
        using base_type::initialize;
        using base_type::print;
        using base_type::serialize;
        using base_type::size;
    };

    //----------------------------------------------------------------------------------//
    /// Overload for a tuple/type-list
    template <template <typename...> class Tuple, typename... Types>
    struct get_storage<Tuple<Types...>>
    : public filtered_get_storage<mpl::implemented_t<Types...>>
    {
        using base_type = filtered_get_storage<mpl::implemented_t<Types...>>;
        using base_type::clear;
        using base_type::initialize;
        using base_type::print;
        using base_type::serialize;
        using base_type::size;
    };

private:
    template <typename... Types>
    friend struct get_storage;

    template <typename... Types>
    friend struct filtered_get_storage;

public:
    /// Get the instance ID for this manager instance
    int32_t instance_count() const { return m_instance_count; }
    /// Get the thread-index for this manager instance
    int64_t get_tid() const { return m_thread_index; }

protected:
    // static comm_group_t get_communicator_group();

protected:
    // protected functions
    TIMEMORY_NODISCARD string_t get_prefix() const;
    void                        internal_write_metadata(const char* = "");

private:
    /// notifies that it is finalizing
    bool            m_is_initialized   = false;
    bool            m_is_finalizing    = false;
    bool            m_is_finalized     = false;
    short           m_write_metadata   = 0;
    int32_t         m_instance_count   = 0;
    int32_t         m_rank             = 0;
    uint64_t        m_num_entries      = 0;
    int64_t         m_metadata_entries = 0;
    int64_t         m_thread_index     = threading::get_id();
    std::thread::id m_thread_id        = threading::get_tid();
    string_t        m_metadata_prefix  = {};
    mutex_t         m_mutex;
    /// increment the shared_ptr count here to ensure these instances live
    /// for the entire lifetime of the manager instance
    hash_map_ptr_t     m_hash_ids           = get_hash_ids();
    hash_alias_ptr_t   m_hash_aliases       = get_hash_aliases();
    initializer_list_t m_initializers       = {};
    finalizer_plist_t  m_finalizer_cleanups = {};
    finalizer_pmap_t   m_master_cleanup     = {};
    finalizer_pmap_t   m_worker_cleanup     = {};
    finalizer_smap_t   m_master_finalizers  = {};
    finalizer_smap_t   m_worker_finalizers  = {};
    finalizer_void_t   m_pointer_fini       = {};
    synchronize_list_t m_synchronize        = {};
    filemap_t          m_output_files       = {};
    settings_ptr_t     m_settings           = settings::shared_instance();

private:
    struct persistent_data
    {
        persistent_data() = default;
        ~persistent_data()
        {
            // make sure the manager is deleted before the settings
            master_instance.reset();
        }

        persistent_data(const persistent_data&) = delete;
        persistent_data(persistent_data&&)      = delete;
        persistent_data& operator=(const persistent_data&) = delete;
        persistent_data& operator=(persistent_data&&) = delete;

        std::atomic<int32_t>      instance_count{ 0 };
        std::atomic<int32_t>      thread_count{ 0 };
        bool                      use_exit_hook = true;
        pointer_t                 master_instance;
        int64_t                   metadata_count = 0;
        metadata_func_t           func_metadata  = {};
        metadata_info_t           info_metadata  = {};
        std::shared_ptr<settings> config         = settings::shared_instance();
    };

    /// single instance of all the global static data
    static persistent_data& f_manager_persistent_data();
    /// number of timing manager instances
    static std::atomic<int32_t>& f_manager_instance_count()
    {
        return f_manager_persistent_data().instance_count;
    }
    /// num-threads based on number of managers created
    static std::atomic<int32_t>& f_thread_counter()
    {
        return f_manager_persistent_data().thread_count;
    }
    /// suppresses the exit hook during termination
    static bool& f_use_exit_hook() { return f_manager_persistent_data().use_exit_hook; }
    static auto  f_settings() { return f_manager_persistent_data().config; }
    static auto  f_debug() { return f_settings() ? false : f_settings()->get_debug(); }
    static auto  f_verbose() { return f_settings() ? 0 : f_settings()->get_verbose(); }

public:
    /// This function stores the primary manager instance for the application
    static void set_persistent_master(pointer_t _pinst)
    {
        tim::manager::f_manager_persistent_data().master_instance = std::move(_pinst);
    }
};
//
//----------------------------------------------------------------------------------//
//
template <typename Func>
void
manager::add_cleanup(void* _key, Func&& _func)
{
    // insert into map
    m_pointer_fini.insert({ _key, std::forward<Func>(_func) });
}
//
//----------------------------------------------------------------------------------//
//
template <typename Func>
void
manager::add_cleanup(const std::string& _key, Func&& _func)
{
    // ensure there are no duplicates
    remove_cleanup(_key);
    // insert into map
    m_finalizer_cleanups.emplace_back(_key, std::forward<Func>(_func));
}
//
//----------------------------------------------------------------------------------//
//
template <typename InitFuncT>
void
manager::add_initializer(InitFuncT&& _init_func)
{
    auto_lock_t _lk{ m_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    m_initializers.emplace_back(std::forward<InitFuncT>(_init_func));
}
//
//----------------------------------------------------------------------------------//
//
inline void
manager::add_finalizer(const std::string& _key, base::data_cleanup* _storage,
                       bool _is_master, int32_t _priority)
{
    m_metadata_prefix = settings::get_global_output_prefix();

    if(m_write_metadata == 0)
        m_write_metadata = 1;

    auto& _finalizer_target = (_is_master) ? m_master_finalizers : m_worker_finalizers;

    _finalizer_target[_priority].emplace_back(_key, _storage);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::add_metadata(const std::string& _key, const Tp& _value)
{
    auto_lock_t _lk(type_mutex<manager>());
    ++f_manager_persistent_data().metadata_count;
    f_manager_persistent_data().func_metadata.push_back([_key, _value](void* _varchive) {
        if(!_varchive)
            return;
        auto* ar = static_cast<cereal::PrettyJSONOutputArchive*>(_varchive);
        (*ar)(cereal::make_nvp(_key.c_str(), _value));
    });
}
//
//--------------------------------------------------------------------------------------//
//
template <typename FuncT>
auto
manager::add_metadata(FuncT&& _func)
    -> decltype(_func(std::declval<cereal::PrettyJSONOutputArchive&>()), void())
{
    auto_lock_t _lk(type_mutex<manager>());
    ++f_manager_persistent_data().metadata_count;
    f_manager_persistent_data().func_metadata.push_back([_func](void* _varchive) {
        if(!_varchive)
            return;
        auto* ar = static_cast<cereal::PrettyJSONOutputArchive*>(_varchive);
        _func(*ar);
    });
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_init_storage()
{
    using storage_type = typename Tp::storage_type;

    static thread_local auto tmp = [&]() {
        if(!component::state<Tp>::has_storage())
        {
            auto ret = storage_type::instance();
            if(ret)
                ret->initialize();

            if(f_debug())
            {
                printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
                       demangle<Tp>().c_str(), (void*) ret,
                       (component::state<Tp>::has_storage()) ? "true" : "false",
                       (ret) ? ((ret->empty()) ? "true" : "false") : "false");
            }
        }
        return true;
    }();
    tim::consume_parameters(tmp);
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_print_storage(const enum_set_t& _types)
{
    using storage_type = typename Tp::storage_type;

    if(!_types.empty() && _types.count(component::properties<Tp>{}()) == 0)
        return;

    auto ret = storage_type::noninit_instance();
    if(ret && !ret->empty())
        ret->print();

    if(f_debug())
    {
        printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
               demangle<Tp>().c_str(), (void*) ret,
               (component::state<Tp>::has_storage()) ? "true" : "false",
               (ret) ? ((ret->empty()) ? "true" : "false") : "false");
    }
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_clear(const enum_set_t& _types)
{
    using storage_type = typename Tp::storage_type;

    if(!_types.empty() && _types.count(component::properties<Tp>{}()) == 0)
        return;

    auto ret = storage_type::noninit_instance();
    if(ret)
        ret->reset();

    if(f_debug())
    {
        printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
               demangle<Tp>().c_str(), (void*) ret,
               (component::state<Tp>::has_storage()) ? "true" : "false",
               (ret) ? ((ret->empty()) ? "true" : "false") : "false");
    }
}
//
//----------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
manager::do_serialize(Archive& ar, const enum_set_t& _types)
{
    using storage_type = typename Tp::storage_type;

    if(!_types.empty() && _types.count(component::properties<Tp>{}()) == 0)
        return;

    auto ret = storage_type::noninit_instance();
    if(ret && !ret->empty())
        ret->do_serialize(ar);

    if(f_debug())
    {
        printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
               demangle<Tp>().c_str(), (void*) ret,
               (component::state<Tp>::has_storage()) ? "true" : "false",
               (ret) ? ((ret->empty()) ? "true" : "false") : "false");
    }
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_size(uint64_t& _sz)
{
    auto label         = tim::demangle<Tp>();
    using storage_type = typename Tp::storage_type;
    label += std::string(" (") + tim::demangle<storage_type>() + ")";

    auto ret = storage_type::noninit_instance();
    if(ret && !ret->empty())
        _sz += ret->size();

    if(f_debug())
    {
        printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
               demangle<Tp>().c_str(), (void*) ret,
               (component::state<Tp>::has_storage()) ? "true" : "false",
               (ret) ? ((ret->empty()) ? "true" : "false") : "false");
    }
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_size(enum_map_t<uint64_t>& _sz)
{
    using storage_type = typename Tp::storage_type;

    auto itr = _sz.find(component::properties<Tp>{}());
    if(itr == _sz.end())
        return;

    auto ret = storage_type::noninit_instance();
    if(ret && !ret->empty())
        itr->second = ret->true_size();
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
manager::filtered_get_storage<Types...>::initialize(pointer_t _manager)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return;
    TIMEMORY_FOLD_EXPRESSION(_manager->do_init_storage<Types>());
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
std::string
manager::filtered_get_storage<Types...>::serialize(pointer_t         _manager,
                                                   const enum_set_t& _types)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return "";
    std::stringstream ss;
    {
        using archive_type = trait::output_archive_t<manager>;
        using policy_type  = policy::output_archive_t<manager>;
        auto oa            = policy_type::get(ss);
        oa->setNextName("timemory");
        oa->startNode();
        {
            TIMEMORY_FOLD_EXPRESSION(
                _manager->do_serialize<archive_type, Types>(*oa, _types));
        }
        oa->finishNode();
    }
    return ss.str();
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
manager::filtered_get_storage<Types...>::clear(pointer_t         _manager,
                                               const enum_set_t& _types)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return;
    TIMEMORY_FOLD_EXPRESSION(_manager->do_clear<Types>(_types));
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
manager::filtered_get_storage<Types...>::print(pointer_t         _manager,
                                               const enum_set_t& _types)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return;
    TIMEMORY_FOLD_EXPRESSION(_manager->do_print_storage<Types>(_types));
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
uint64_t
manager::filtered_get_storage<Types...>::size(pointer_t _manager)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return 0;
    uint64_t _sz = 0;
    TIMEMORY_FOLD_EXPRESSION(_manager->do_size<Types>(_sz));
    return _sz;
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
manager::enum_map_t<uint64_t>
manager::filtered_get_storage<Types...>::size(pointer_t         _manager,
                                              const enum_set_t& _types)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();

    enum_map_t<uint64_t> _sz{};
    if(_manager)
    {
        for(const auto& itr : _types)
            _sz.insert({ itr, 0 });
        TIMEMORY_FOLD_EXPRESSION(_manager->do_size<Types>(_sz));
    }
    return _sz;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
