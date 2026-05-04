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

#ifndef TIMEMORY_STORAGE_IMPL_STORAGE_FALSE_CPP_
#define TIMEMORY_STORAGE_IMPL_STORAGE_FALSE_CPP_

#include "timemory/storage/impl_storage_false.hpp"

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/data/stream.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/operations/types/fini.hpp"
#include "timemory/operations/types/init.hpp"
#include "timemory/operations/types/node.hpp"
#include "timemory/operations/types/start.hpp"
#include "timemory/operations/types/stop.hpp"
#include "timemory/plotting/declaration.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/storage/macros.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/demangle.hpp"

#include <atomic>
#include <cstddef>
#include <fstream>
#include <functional>
#include <memory>
#include <regex>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace tim
{
namespace impl
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, false>::storage()
: base_type(singleton_t::is_main_thread(), instance_count()++, demangle<Type>())
{
    TIMEMORY_CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "constructing %s",
                                    m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() >= 4, 16);
    get_shared_manager();
    component::state<Type>::has_storage() = true;

    if(operation::get_storage<Type>{}(m_thread_idx) == nullptr)
    {
        using value_type = typename trait::collects_data<Type>::type;
        operation::set_storage<Type>{}(static_cast<tim::storage<Type, value_type>*>(this),
                                       m_thread_idx);
    }

    if(!m_is_master && master_instance())
        master_instance()->add_child(this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, false>::storage(standalone_storage, int64_t _instance_id,
                              std::string _label)
: base_type(standalone_storage{}, _instance_id, std::move(_label))
{
    TIMEMORY_CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "constructing %s",
                                    m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() >= 4, 16);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, false>::~storage()
{
    if(!m_standalone)
        component::state<Type>::has_storage() = false;

    auto _debug = (m_settings) ? m_settings->get_debug() : true;

    if(!m_standalone)
    {
        if(!m_is_master)
        {
            auto _main_instance = singleton_t::master_instance();
            if(_main_instance && _main_instance != this)
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE(_debug,
                                                "[%s|%li]> merging into primary instance",
                                                m_label.c_str(), (long) m_instance_id);
                _main_instance->merge(this);
                _main_instance->remove_child(this);
            }
            else
            {
                TIMEMORY_CONDITIONAL_PRINT_HERE(
                    _debug,
                    "[%s|%li]> skipping merge into non-existent primary "
                    "instance",
                    m_label.c_str(), (long) m_instance_id);
            }
        }
    }

    if(operation::get_storage<Type>{}(m_thread_idx) == this)
        operation::set_storage<Type>{}(nullptr, m_thread_idx);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::stack_clear()
{
    if(!m_stack.empty() && m_settings && m_settings->get_stack_clearing())
    {
        std::unordered_set<Type*> _stack = m_stack;
        for(auto& itr : _stack)
            operation::stop<Type>{ *itr };
    }
    m_stack.clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::initialize()
{
    if(m_initialized)
        return;

    TIMEMORY_CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "initializing %s",
                                    m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() >= 4, 16);

    m_initialized = true;

    using init_t = operation::init<Type>;
    using ValueT = typename trait::collects_data<Type>::type;
    auto upcast  = static_cast<tim::storage<Type, ValueT>*>(this);

    if(!m_is_master)
    {
        init_t(upcast, operation::mode_constant<operation::init_mode::thread>{});
    }
    else
    {
        init_t(upcast, operation::mode_constant<operation::init_mode::global>{});
        init_t(upcast, operation::mode_constant<operation::init_mode::thread>{});
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::finalize()
{
    if(m_finalized)
        return;

    if(!m_initialized)
        return;

    TIMEMORY_CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "finalizing %s",
                                    m_label.c_str());

    using fini_t = operation::fini<Type>;
    using ValueT = typename trait::collects_data<Type>::type;
    auto upcast  = static_cast<tim::storage<Type, ValueT>*>(this);

    m_finalized = true;
    manager::instance()->is_finalizing(true);
    if(!m_is_master)
    {
        worker_is_finalizing() = true;
        fini_t(upcast, operation::mode_constant<operation::fini_mode::thread>{});
    }
    else
    {
        master_is_finalizing() = true;
        worker_is_finalizing() = true;
        fini_t(upcast, operation::mode_constant<operation::fini_mode::thread>{});
        fini_t(upcast, operation::mode_constant<operation::fini_mode::global>{});
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::stack_pop(Type* obj)
{
    auto itr = m_stack.find(obj);
    if(itr != m_stack.end())
    {
        m_stack.erase(itr);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::merge()
{
    auto_lock_t _lk{ base_type::m_mutex };

    for(auto itr : base_type::get_children())
    {
        auto _v = dynamic_cast<this_type*>(itr);
        if(_v)
            merge(_v);
    }

    stack_clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::merge(this_type* itr)
{
    if(itr)
        operation::finalize::merge<Type, false>(*this, *itr);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::get_shared_manager()
{
    // only perform this operation when not finalizing
    if(!this_type::is_finalizing())
    {
        if(!m_manager)
            return;
        if(m_manager->is_finalizing())
            return;

        auto       _label = demangle(Type::label());
        std::regex _namespace_re{ "^(tim::[a-z_]+::|tim::)([a-z].*)" };
        if(std::regex_search(_label, _namespace_re))
            _label = std::regex_replace(_label, _namespace_re, "$2");
        // replace spaces with underscores
        auto _pos = std::string::npos;
        while((_pos = _label.find_first_of(" -")) != std::string::npos)
            _label = _label.replace(_pos, 1, "_");
        // convert to upper-case
        for(auto& itr : _label)
            itr = toupper(itr);
        // handle any remaining brackets or colons
        for(auto itr : { ':', '<', '>' })
        {
            while((_pos = _label.find(itr)) != std::string::npos)
                _pos = _label.erase(_pos, 1).find(itr);
        }
        std::stringstream env_var;
        env_var << TIMEMORY_SETTINGS_PREFIX << _label << "_ENABLED";
        auto _enabled = tim::get_env<bool>(env_var.str(), true);
        if(_enabled != trait::runtime_enabled<Type>::get())
            trait::runtime_enabled<Type>::set(_enabled);

        m_manager->add_finalizer(demangle<Type>(), this, m_is_master,
                                 trait::fini_priority<Type>::value);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::deregister()
{
    // if the finalizer has not run yet, remove it and invoke directly
    if(m_manager)
    {
        auto _finalizers = m_manager->remove_finalizer(demangle<Type>());
        for(auto& itr : _finalizers)
            itr();
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::destroy()
{
    auto _debug_v = (m_settings) ? m_settings->get_debug() : false;
    auto _verb_v  = (m_settings) ? m_settings->get_verbose() : 0;
    if(_debug_v || _verb_v >= 3)
        TIMEMORY_PRINT_HERE("Destroying storage for %s", demangle<Type>().c_str());

    operation::cleanup<Type>{};

    if(!m_is_master)
    {
        auto _main_instance = master_instance();
        if(_main_instance)
            _main_instance->merge(this);
    }
    else
    {
        stack_clear();
        print();
    }

    if(m_is_master)
        trait::runtime_enabled<Type>::set(false);

    if(_debug_v || _verb_v >= 3)
        TIMEMORY_PRINT_HERE("Storage destroyed for  %s", demangle<Type>().c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::noninit_instance()
{
    return get_singleton() ? get_singleton()->instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::noninit_master_instance()
{
    return get_singleton() ? get_singleton()->master_instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
storage<Type, false>::master_is_finalizing()
{
    static bool _instance = false;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
storage<Type, false>::worker_is_finalizing()
{
    static thread_local bool _instance = master_is_finalizing();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, false>::is_finalizing()
{
    return worker_is_finalizing() || master_is_finalizing();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::atomic<int64_t>&
storage<Type, false>::instance_count()
{
    static std::atomic<int64_t> _counter{ 0 };
    return _counter;
}

}  // namespace impl
}  // namespace tim
#endif
