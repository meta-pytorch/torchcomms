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
 * \file timemory/components/ompt/components.hpp
 * \brief Implementation of the ompt component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/utility/demangle.hpp"
//
#include "timemory/components/data_tracker/components.hpp"
#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/types.hpp"
//
#include "timemory/operations/types/node.hpp"
#include "timemory/operations/types/start.hpp"
#include "timemory/operations/types/stop.hpp"
#include "timemory/operations/types/store.hpp"
//
//======================================================================================//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Api>
struct ompt_handle
: public base<ompt_handle<Api>, void>
, private policy::instance_tracker<ompt_handle<Api>>
{
    using api_type     = Api;
    using this_type    = ompt_handle<api_type>;
    using value_type   = void;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using toolset_type = typename trait::ompt_handle<api_type>::type;
    using tracker_type = policy::instance_tracker<this_type>;

    static std::string label() { return "ompt_handle"; }
    static std::string description()
    {
        return std::string(
                   "Control switch for enabling/disabling OpenMP tools defined by the ") +
               demangle<api_type>() + " tag";
    }

    static auto& get_initializer()
    {
        static std::function<void()> _instance = []() {};
        return _instance;
    }

    static void configure()
    {
        static int32_t _once = 0;
        if(_once++ == 0)
            this_type::get_initializer()();
    }

    static void preinit()
    {
        static thread_local auto _data_tracker =
            tim::storage_initializer::get<ompt_data_tracker<api_type>>();
        consume_parameters(_data_tracker);
    }

    static void global_init()
    {
        // if handle gets initialized (i.e. used), it indicates we want to disable
        trait::runtime_enabled<toolset_type>::set(false);
        configure();
    }

    static void global_finalize() { trait::runtime_enabled<toolset_type>::set(false); }

    static void thread_finalize()
    {
        if(get_persistent_data().m_cleanup)
            get_persistent_data().m_cleanup();
    }

    void start()
    {
#if defined(TIMEMORY_USE_OMPT)
        tracker_type::start();
        if(m_tot == 0)
            trait::runtime_enabled<toolset_type>::set(true);
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_OMPT)
        tracker_type::stop();
        if(m_tot == 0)
            trait::runtime_enabled<toolset_type>::set(false);
#endif
    }

private:
    using tracker_type::m_tot;

public:
    void set_prefix(const std::string& _prefix)
    {
        if(_prefix.empty())
            return;
        tim::auto_lock_t lk(get_persistent_data().m_mutex);
        get_persistent_data().m_prefix = _prefix + "/";
    }

    static std::string get_prefix()
    {
        tim::auto_lock_t lk(get_persistent_data().m_mutex);
        return get_persistent_data().m_prefix;
    }

    template <typename FuncT>
    static void add_cleanup(FuncT&& _current)
    {
        auto _former                    = get_persistent_data().m_cleanup;
        get_persistent_data().m_cleanup = [_former, _current]() {
            _former();
            _current();
        };
    }

private:
    struct persistent_data
    {
        mutex_t               m_mutex;
        std::string           m_prefix  = {};
        std::function<void()> m_cleanup = []() {};
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Api>
struct ompt_data_tracker : public base<ompt_data_tracker<Api>, void>
{
    using api_type     = Api;
    using this_type    = ompt_data_tracker<api_type>;
    using value_type   = void;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;

    using tracker_t = ompt_data_tracker_t;

    static std::string label() { return "ompt_data_tracker"; }
    static std::string description()
    {
        return std::string("OpenMP tools data tracker ") + demangle<api_type>();
    }

    static void preinit()
    {
        static thread_local auto _tracker_storage = storage_initializer::get<tracker_t>();
        consume_parameters(_tracker_storage);
    }

    static void global_init()
    {
        preinit();
        tracker_t::label() = "ompt_data_tracker";
    }

    void start() {}
    void stop() {}

    template <typename... Args>
    void store(Args&&...)
    {
        apply_store<tracker_t>(std::plus<size_t>{}, 1);
    }

    void store(ompt_id_t target_id, ompt_id_t host_op_id, ompt_target_data_op_t optype,
               void* host_addr, void* device_addr, size_t bytes)
    {
        apply_store<tracker_t>(std::plus<size_t>{}, bytes);
        consume_parameters(target_id, host_op_id, optype, host_addr, device_addr);
    }

    void store(ompt_id_t target_id, unsigned int nitems, void** host_addr,
               void** device_addr, const size_t* bytes, unsigned int* mapping_flags)
    {
        size_t _tot = 0;
        for(unsigned int i = 0; i < nitems; ++i)
            _tot += bytes[i];
        apply_store<tracker_t>(std::plus<size_t>{}, _tot);
        consume_parameters(target_id, host_addr, device_addr, mapping_flags);
        // auto _prefix = tim::get_hash_identifier(m_prefix_hash);
    }

public:
    void set_prefix(uint64_t _prefix_hash) { m_prefix_hash = _prefix_hash; }
    void set_scope(scope::config _scope) { m_scope_config = _scope; }

private:
    template <typename Tp, typename... Args>
    void apply_store(Tp& _obj, Args&&... args)
    {
        operation::push_node<Tp>(_obj, m_scope_config, m_prefix_hash);
        operation::start<Tp> _start(_obj);
        operation::store<Tp>(_obj, std::forward<Args>(args)...);
        operation::stop<Tp>     _stop(_obj);
        operation::pop_node<Tp> _pop(_obj);
    }

    template <typename Tp, typename... Args>
    void apply_store(Args&&... args)
    {
        Tp _obj;
        apply_store(_obj, std::forward<Args>(args)...);
    }

private:
    uint64_t      m_prefix_hash  = 0;
    scope::config m_scope_config = scope::get_default();
};
//
//--------------------------------------------------------------------------------------//
//

}  // namespace component
}  // namespace tim
//
//======================================================================================//
