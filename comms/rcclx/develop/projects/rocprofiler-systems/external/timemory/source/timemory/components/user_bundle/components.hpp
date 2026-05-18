//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/**
 * \file timemory/components/user_bundle/components.hpp
 * \brief Implementation of the user_bundle component(s)
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/user_bundle/backends.hpp"
#include "timemory/components/user_bundle/types.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/demangle.hpp"

#include <algorithm>
#include <functional>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

//======================================================================================//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace env
{
//
using user_bundle_spec_t = std::function<std::string()>;
//
using user_bundle_variables_t =
    std::unordered_map<size_t, std::vector<user_bundle_spec_t>>;
//
//--------------------------------------------------------------------------------------//
/// non-static so that projects can globally change this for their project/API
template <typename ApiT>
inline user_bundle_variables_t& get_user_bundle_variables(ApiT)
{
    static user_bundle_variables_t _instance{};
    return _instance;
}
//
#if defined(TIMEMORY_USER_BUNDLE_COMPONENT_HEADER_MODE)
/// static so that projects cannot globally change this
static inline user_bundle_variables_t& get_user_bundle_variables(TIMEMORY_API);
static inline user_bundle_variables_t& get_user_bundle_variables(project::kokkosp);
//
inline std::vector<TIMEMORY_COMPONENT>
get_bundle_components(const std::vector<user_bundle_spec_t>& _priority);
#else
/// static so that projects cannot globally change this
user_bundle_variables_t& get_user_bundle_variables(TIMEMORY_API);
user_bundle_variables_t& get_user_bundle_variables(project::kokkosp);
//
std::vector<TIMEMORY_COMPONENT>
get_bundle_components(const std::vector<user_bundle_spec_t>& _priority);
#endif
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Api, typename AltApi = Api>
void
initialize_bundle(AltApi _api = AltApi{})
{
    using user_bundle_type = component::user_bundle<Idx, Api>;
    auto& variables        = env::get_user_bundle_variables(_api);
    auto  itr              = variables.find(Idx);
    if(itr != variables.end())
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(
            (settings::instance()) ? (settings::instance()->get_debug()) : false,
            "getting user bundle components for type %s (%s)",
            demangle<user_bundle_type>().c_str(), user_bundle_type::label().c_str());

        auto _enum = env::get_bundle_components(itr->second);
        tim::configure<user_bundle_type>(_enum);
    }
}
//
}  // namespace env
//
//--------------------------------------------------------------------------------------//
//
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                                  USER BUNDLE
//
//--------------------------------------------------------------------------------------//
//
namespace internal
{
struct user_bundle
{
public:
    using mutex_t        = std::mutex;
    using lock_t         = std::unique_lock<mutex_t>;
    using opaque_array_t = std::vector<opaque>;
    using typeid_vec_t   = std::vector<size_t>;
    using typeid_set_t   = std::set<size_t>;

    static std::string label();
    static std::string description();
    static void        record() {}

public:
    user_bundle(scope::config _cfg, typeid_vec_t _typeids, opaque_array_t _opaque_arr,
                const char* _prefix = nullptr);
    ~user_bundle();
    user_bundle(const user_bundle& rhs);
    user_bundle(user_bundle&& rhs) noexcept;
    user_bundle& operator=(const user_bundle& rhs);
    user_bundle& operator=(user_bundle&& rhs) noexcept;

public:
    ///  Configure the tool for a specific component
    static void configure(opaque_array_t& _data, typeid_vec_t& _typeids, mutex_t& _mtx,
                          opaque&& obj, std::set<size_t>&& _inp);

    ///  Explicitly clear the previous configurations
    static void reset(opaque_array_t& _data, typeid_vec_t& _typeids, mutex_t& _mtx);

public:
    void          setup();
    void          push();
    void          sample();
    void          start();
    void          stop();
    void          pop();
    void          get(void*& ptr, size_t _hash) const;
    void          set_prefix(const char* _prefix);
    void          set_scope(const scope::config& val);
    void          update_statistics(bool _v) const;
    size_t        size() const { return m_bundle.size(); }
    const char*   get_prefix() const { return m_prefix; }
    scope::config get_scope() const { return m_scope; }

public:
    //  Configure the tool for a specific component
    void insert(opaque&& obj, typeid_set_t&& _typeids);

protected:
    bool           m_setup   = false;
    scope::config  m_scope   = {};
    const char*    m_prefix  = nullptr;
    typeid_vec_t   m_typeids = {};
    opaque_array_t m_bundle  = {};

protected:
    static bool contains(size_t _val, const typeid_vec_t& _targ);
};
}  // namespace internal
//
template <size_t Idx, typename Tag>
struct user_bundle
: public base<user_bundle<Idx, Tag>, void>
, concepts::runtime_configurable
, private internal::user_bundle
{
public:
    static constexpr auto index = Idx;
    using tag_type              = Tag;
    using mutex_t               = std::mutex;
    using start_func_t          = std::function<void*(const std::string&, scope::config)>;
    using stop_func_t           = std::function<void(void*)>;
    using get_func_t            = std::function<void(void*, void*&, size_t)>;
    using delete_func_t         = std::function<void(void*)>;
    using opaque_array_t        = std::vector<opaque>;
    using typeid_vec_t          = std::vector<size_t>;
    using typeid_set_t          = std::set<size_t>;

    using value_type   = void;
    using this_type    = user_bundle<Idx, Tag>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

    static size_t bundle_size() { return get_data().size(); }
    static void   global_init(bool _preinit = false) TIMEMORY_VISIBILITY("default");
    static void   global_init(storage_type*) { global_init(false); }

    using internal::user_bundle::description;
    using internal::user_bundle::label;
    using internal::user_bundle::record;

private:
    using internal::user_bundle::m_bundle;
    using internal::user_bundle::m_prefix;
    using internal::user_bundle::m_scope;
    using internal::user_bundle::m_setup;
    using internal::user_bundle::m_typeids;

public:
    //----------------------------------------------------------------------------------//
    //  Captures the statically-defined data so these can be changed without
    //  affecting this instance
    //
    user_bundle()
    : internal::user_bundle{ scope::get_default(), get_typeids(),
                             (persistent_init(), get_data()) }
    {}

    explicit user_bundle(const char* _prefix, scope::config _scope = scope::get_default())
    : internal::user_bundle{ _scope, get_typeids(), (persistent_init(), get_data()),
                             _prefix }
    {}

    user_bundle(const char* _prefix, const opaque_array_t& _bundle_vec,
                const typeid_vec_t& _typeids, scope::config _scope = scope::get_default())
    : internal::user_bundle{ _scope, _typeids, _bundle_vec, _prefix }
    {}

    user_bundle(const char* _prefix, const opaque_array_t& _bundle_vec,
                const typeid_set_t& _typeids, scope::config _scope = scope::get_default())
    : internal::user_bundle{ _scope, typeid_vec_t{}, _bundle_vec, _prefix }
    {
        m_typeids.reserve(_typeids.size());
        for(const auto& itr : _typeids)
            m_typeids.emplace_back(itr);
    }

    ~user_bundle()                          = default;
    user_bundle(const user_bundle&)         = default;
    user_bundle(user_bundle&& rhs) noexcept = default;

    user_bundle& operator=(const user_bundle& rhs) = default;
    user_bundle& operator=(user_bundle&& rhs) noexcept = default;

public:
    //  Configure the tool for a specific component
    static void configure(opaque&& obj, std::set<size_t>&& _typeids)
    {
        internal::user_bundle::configure(get_data(), get_typeids(), get_lock(),
                                         std::forward<opaque>(obj),
                                         std::forward<std::set<size_t>>(_typeids));
    }

    template <typename Type, typename... Types, typename... Args>
    static void configure(Args&&... args)
    {
        this_type::configure(factory::get_opaque<Type>(std::forward<Args>(args)...),
                             factory::get_typeids<Type>());

        TIMEMORY_FOLD_EXPRESSION(
            this_type::configure(factory::get_opaque<Types>(std::forward<Args>(args)...),
                                 factory::get_typeids<Types>()));
    }

    //  Explicitly clear the previous configurations
    static void reset()
    {
        TIMEMORY_CONDITIONAL_PRINT_HERE(settings::verbose() > 3 || settings::debug(),
                                        "Resetting %s", demangle<this_type>().c_str());
        internal::user_bundle::reset(get_data(), get_typeids(), get_lock());
    }

public:
    using internal::user_bundle::get_prefix;
    using internal::user_bundle::get_scope;
    using internal::user_bundle::insert;
    using internal::user_bundle::pop;
    using internal::user_bundle::push;
    using internal::user_bundle::sample;
    using internal::user_bundle::set_prefix;
    using internal::user_bundle::set_scope;
    using internal::user_bundle::setup;
    using internal::user_bundle::size;
    using internal::user_bundle::start;
    using internal::user_bundle::stop;
    using internal::user_bundle::update_statistics;

    using base_type::operator+=;
    using base_type::operator-=;
    using base_type::get_depth_change;
    using base_type::get_is_flat;
    using base_type::get_is_invalid;
    using base_type::get_is_on_stack;
    using base_type::get_is_running;
    using base_type::get_is_transient;
    using base_type::get_iterator;
    using base_type::get_laps;
    using base_type::get_opaque;
    using base_type::set_depth_change;
    using base_type::set_is_flat;
    using base_type::set_is_invalid;
    using base_type::set_is_on_stack;
    using base_type::set_is_running;
    using base_type::set_is_transient;
    using base_type::set_iterator;
    using base_type::set_laps;
    using base_type::set_started;
    using base_type::set_stopped;

    void clear()
    {
        if(base_type::get_is_running())
            stop();
        m_typeids.clear();
        m_bundle.clear();
        m_setup = false;
    }

    template <typename T>
    T* get()
    {
        auto  _typeid_hash = typeid_hash<T>();
        void* void_ptr     = nullptr;
        for(auto& itr : m_bundle)
        {
            itr.get(void_ptr, _typeid_hash);
            if(void_ptr)
                return static_cast<T*>(void_ptr);
        }
        return static_cast<T*>(void_ptr);
    }

    void get() const {}

    void get(void*& ptr, size_t _typeid_hash) const
    {
        base_type::get(ptr, _typeid_hash);
        internal::user_bundle::get(ptr, _typeid_hash);
    }

public:
    template <typename Type, typename... Types, typename... Args>
    void insert(Args... args)
    {
        this->insert(factory::get_opaque<Type>(args...), factory::get_typeids<Type>());
        TIMEMORY_FOLD_EXPRESSION(this->insert(factory::get_opaque<Types>(args...),
                                              factory::get_typeids<Types>()));
    }

private:
    static bool persistent_init();
    struct persistent_data;
    static persistent_data& get_persistent_data() TIMEMORY_VISIBILITY("default");

    struct persistent_data
    {
        volatile bool             m_init    = false;
        bool                      m_preinit = false;
        mutex_t                   m_lock;
        opaque_array_t            m_data     = {};
        typeid_vec_t              m_typeids  = {};
        std::shared_ptr<settings> m_settings = settings::shared_instance();

        bool init(bool _preinit = false)
        {
            if(!m_init)
            {
                if(_preinit)
                    m_preinit = true;
                if(m_settings && m_settings->get_initialized())
                {
                    if(m_preinit)
                        reset();
                    m_init = true;  // comma operator
                }
                if(m_init || _preinit)
                    env::initialize_bundle<Idx, Tag>();
            }
            return m_init;
        }
    };

public:
    /// template instantiation-specific static opaque array
    static opaque_array_t& get_data() { return get_persistent_data().m_data; }

    /// template instantiation-specific static set of type identifiers
    static typeid_vec_t& get_typeids() { return get_persistent_data().m_typeids; }

    /// template instantiation-specific static mutex
    static mutex_t& get_lock() { return get_persistent_data().m_lock; }
};
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Tag>
void
user_bundle<Idx, Tag>::global_init(bool _preinit)
{
    if(settings::verbose() > 2 || settings::debug())
        TIMEMORY_PRINT_HERE("Global initialization of %s", demangle<this_type>().c_str());
    get_persistent_data().init(_preinit);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Tag>
typename user_bundle<Idx, Tag>::persistent_data&
user_bundle<Idx, Tag>::get_persistent_data()
{
    static persistent_data _instance{};
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Tag>
bool
user_bundle<Idx, Tag>::persistent_init()
{
    return get_persistent_data().init();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//

#if defined(TIMEMORY_USER_BUNDLE_COMPONENT_HEADER_MODE)
#    include "timemory/components/user_bundle/components.cpp"
#endif
