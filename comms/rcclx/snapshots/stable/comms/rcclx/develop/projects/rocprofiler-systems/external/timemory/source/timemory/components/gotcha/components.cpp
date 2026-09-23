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

#ifndef TIMEMORY_COMPONENTS_GOTCHA_COMPONENTS_CPP_
#define TIMEMORY_COMPONENTS_GOTCHA_COMPONENTS_CPP_

#include "timemory/defines.h"

#include <type_traits>

#if !defined(TIMEMORY_COMPONENTS_GOTCHA_COMPONENTS_HPP_)
#    include "timemory/components/gotcha/components.hpp"
#endif

#include "timemory/backends/gotcha.hpp"
#include "timemory/components/gotcha/backends.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"

#include <cstddef>
#include <string>

#if defined(TIMEMORY_NVCC_COMPILER)
// 187: '=' should possibly be '==' warning
#    pragma nv_diag_suppress 187
#endif

namespace tim
{
namespace component
{
//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
std::string
gotcha<Nt, BundleT, DiffT>::label()
{
    return "gotcha";
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
std::string
gotcha<Nt, BundleT, DiffT>::description()
{
    return "Generates GOTCHA wrappers which can be used to wrap or replace "
           "dynamically linked function calls";
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
typename gotcha<Nt, BundleT, DiffT>::get_initializer_t&
gotcha<Nt, BundleT, DiffT>::get_initializer()
{
    return get_persistent_data().m_initializer;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
typename gotcha<Nt, BundleT, DiffT>::get_select_list_t&
gotcha<Nt, BundleT, DiffT>::get_permit_list()
{
    return get_persistent_data().m_permit_list;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
typename gotcha<Nt, BundleT, DiffT>::get_select_list_t&
gotcha<Nt, BundleT, DiffT>::get_reject_list()
{
    return get_persistent_data().m_reject_list;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
bool&
gotcha<Nt, BundleT, DiffT>::get_default_ready()
{
    static bool _instance = true;
    return _instance;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
void
gotcha<Nt, BundleT, DiffT>::add_global_suppression(const std::string& func)
{
    get_suppresses().insert(func);
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
auto
gotcha<Nt, BundleT, DiffT>::get_ready()
{
    std::array<std::pair<bool, bool>, Nt> _ready;
    for(size_t i = 0; i < Nt; ++i)
        _ready.at(i) = { get_data().at(i).filled, get_data().at(i).ready };
    return _ready;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
auto
gotcha<Nt, BundleT, DiffT>::set_ready(bool val)
{
    for(size_t i = 0; i < Nt; ++i)
    {
        if(get_data().at(i).filled)
            get_data().at(i).ready = val;
    }
    return get_ready();
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
auto
gotcha<Nt, BundleT, DiffT>::set_ready(const std::array<bool, Nt>& values)
{
    for(size_t i = 0; i < Nt; ++i)
    {
        if(get_data().at(i).filled)
            get_data().at(i).ready = values.at(i);
    }
    return get_ready();
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
bool
gotcha<Nt, BundleT, DiffT>::construct(const std::string& _func, int _priority,
                                      const std::string& _tool)
{
    if(_func.empty())
        return false;

    gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

    init_storage<bundle_type>(0);

    static_assert(N < Nt, "Error! N must be less than Nt!");
    auto& _data = get_data()[N];

    assert(_data.index == N);

    if(!is_permitted<N, Ret, Args...>(_func))
        return false;

    if(_data.debug == nullptr)
        _data.debug = &settings::debug();

    if(!_data.filled)
    {
        auto _label = demangle(_func);

        // ensure the hash to string pairing is stored
        storage_type::instance()->add_hash_id(_func);
        storage_type::instance()->add_hash_id(_label);

        if(!_tool.empty() && _label.find(_tool + "/") != 0)
        {
            _label = _tool + "/" + _label;
            while(_label.find("//") != std::string::npos)
                _label.erase(_label.find("//"), 1);
        }

        // ensure the hash to string pairing is stored
        storage_type::instance()->add_hash_id(_label);

        if(_data.verbose == 0)
            _data.verbose = backend::gotcha::get_verbose();

        _data.filled   = true;
        _data.priority = _priority;
        _data.tool_id  = _label;
        _data.wrap_id  = _func;
        _data.ready    = get_default_ready();

        if(get_suppresses().find(_func) != get_suppresses().end())
        {
            _data.suppression = &gotcha_suppression::get();
            //_data.ready       = false;
        }

        _data.constructor = [_func, _priority, _tool]() {
            this_type::construct<N, Ret, Args...>(_func, _priority, _tool);
        };
        _data.destructor = []() { this_type::revert<N>(); };
        _data.binding    = std::move(construct_binder<N, Ret, Args...>(_data.wrap_id));
        error_t ret_wrap = backend::gotcha::wrap(_data.binding, _data.tool_id.c_str());
        check_error<N>(ret_wrap, "binding");

        using func_t = Ret (*)(Args...);
        operation::set_data<operator_type>{}(_data);
        operation::set_data<operator_type>{}(
            reinterpret_cast<func_t>(gotcha_get_wrappee(_data.wrappee)));
        operation::set_data<bundle_type>{}(_data);
        operation::set_data<bundle_type>{}(
            reinterpret_cast<func_t>(gotcha_get_wrappee(_data.wrappee)));
    }

    if(!_data.is_active)
    {
        _data.is_active  = true;
        error_t ret_prio = backend::gotcha::set_priority(_data.tool_id, _data.priority);
        check_error<N>(ret_prio, "set priority");
    }

    if(!_data.ready)
        revert<N>();

    return _data.filled;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
bool
gotcha<Nt, BundleT, DiffT>::configure(const gotcha_config<N, Ret, Args...>& _cfg)
{
    for(auto&& itr : _cfg.names)
    {
        if(construct<N, Ret, Args...>(itr, _cfg.priority, _cfg.tool))
            return true;
    }
    return false;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
bool
gotcha<Nt, BundleT, DiffT>::configure(const std::string& _func, int _priority,
                                      const std::string& _tool)
{
    return configure(gotcha_config<N, Ret, Args...>{ _func, _priority, _tool });
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
bool
gotcha<Nt, BundleT, DiffT>::configure(const std::vector<std::string>& _funcs,
                                      int _priority, const std::string& _tool)
{
    return configure(gotcha_config<N, Ret, Args...>{ _funcs, _priority, _tool });
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N>
bool
gotcha<Nt, BundleT, DiffT>::revert()
{
    gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

    static_assert(N < Nt, "Error! N must be less than Nt!");
    auto& _data = get_data()[N];

    if(_data.filled && _data.is_active)
    {
        _data.is_active = false;
        if(get_suppresses().find(_data.tool_id) != get_suppresses().end())
        {
            _data.ready = false;
        }
        else
        {
            _data.ready = get_default_ready();
        }
    }

    return _data.filled;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
auto
gotcha<Nt, BundleT, DiffT>::get_info()
{
    std::array<size_t, 5> _info{};
    _info.fill(0);
    for(auto& itr : get_data())
    {
        _info.at(0) += (itr.ready) ? 1 : 0;
        _info.at(1) += (itr.filled) ? 1 : 0;
        _info.at(2) += (itr.is_active) ? 1 : 0;
        _info.at(3) += (itr.is_finalized) ? 1 : 0;
        _info.at(4) += (itr.suppression && !(*itr.suppression)) ? 1 : 0;
    }
    return _info;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
void
gotcha<Nt, BundleT, DiffT>::configure()
{
    // do not release lock until fully constructed
    locking::spin_lock lk(get_mutex(), std::defer_lock);
    if(!lk.owns_lock())
        lk.lock();

    if(!is_configured())
    {
        is_configured() = true;
        auto& _init     = get_initializer();
        _init();
        operation::set_data<operator_type>{}(get_data());
        operation::set_data<bundle_type>{}(get_data());
        for(auto& itr : get_data())
        {
            if(itr.is_finalized)
            {
                itr.is_finalized = false;
                itr.constructor();
            }
        }
    }
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
void
gotcha<Nt, BundleT, DiffT>::disable()
{
    // do not release lock until fully destroyed
    locking::spin_lock lk(get_mutex(), std::defer_lock);
    if(!lk.owns_lock())
        lk.lock();

    if(is_configured())
    {
        is_configured() = false;
        for(auto& itr : get_data())
        {
            if(!itr.is_finalized)
            {
                itr.is_finalized = true;
                itr.destructor();
            }
        }
    }
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
void
gotcha<Nt, BundleT, DiffT>::global_finalize()
{
    while(get_started() > 0)
        --get_started();
    while(get_thread_started() > 0)
        --get_thread_started();
    disable();
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
void
gotcha<Nt, BundleT, DiffT>::thread_init()
{
    auto& _data = get_data();
    for(size_t i = 0; i < Nt; ++i)
        _data[i].ready = (_data[i].filled && get_default_ready());
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
gotcha_data*
gotcha<Nt, BundleT, DiffT>::at(size_t _idx)
{
    return (_idx < get_data().size()) ? &get_data().at(_idx) : nullptr;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
void
gotcha<Nt, BundleT, DiffT>::start()
{
    if(storage_type::is_finalizing())
        return;

    auto _n = get_started()++;
    auto _t = get_thread_started()++;

    // this ensures that if started from multiple threads, all threads synchronize
    // before
    if(_t == 0 && !is_configured())
        configure();

    if(_n == 0)
    {
        configure();
        for(auto& itr : get_data())
        {
            if(!itr.is_finalized)
                itr.constructor();
        }
    }

    if(_t == 0)
    {
        auto& _data = get_data();
        for(size_t i = 0; i < Nt; ++i)
            _data[i].ready = _data[i].filled;
    }
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
void
gotcha<Nt, BundleT, DiffT>::stop()
{
    auto _n = --get_started();
    auto _t = --get_thread_started();

    if(_t == 0)
    {
        auto& _data = get_data();
        for(size_t i = 0; i < Nt; ++i)
            _data[i].ready = false;
    }

    if(_n == 0)
    {
        for(auto& itr : get_data())
        {
            if(!itr.is_finalized)
                itr.destructor();
        }
    }
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
void
gotcha<Nt, BundleT, DiffT>::instrument<N, Ret, Args...>::generate(
    const std::string& _func, const std::string& _tool, int _priority)
{
    this_type::configure(gotcha_config<N, Ret, Args...>{ _func, _priority, _tool });
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
void
gotcha<Nt, BundleT, DiffT>::gotcha_factory(const std::string& _func,
                                           const std::string& _tool, int _priority)
{
    instrument<N, Ret, Args...>::generate(_func, _tool, _priority);
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
bool
gotcha<Nt, BundleT, DiffT>::is_permitted(const std::string& _func)
{
    // if instruments are being used, we need to restrict using GOTCHAs around
    // certain MPI functions which can cause deadlocks. However, allow
    // these GOTCHA components which serve as function replacements to
    // wrap these functions
    if(std::is_same<operator_type, void>::value &&
       (_func.find("MPI_") != std::string::npos ||
        _func.find("mpi_") != std::string::npos))
    {
        static auto mpi_reject_list = { "MPI_Pcontrol", "MPI_T_init_thread",
                                        "MPI_Abort" };

        auto tofortran = [](std::string _fort) {
            for(auto& itr : _fort)
                itr = tolower(itr);
            if(!_fort.empty() && _fort[_fort.length() - 1] != '_')
                _fort += "_";
            return _fort;
        };

        // if function matches a reject_listed entry, do not construct wrapper
        for(const auto& itr : mpi_reject_list)
        {
            if(_func == itr || _func == tofortran(itr))
            {
                if(backend::gotcha::get_verbose() >= 1)
                {
                    TIMEMORY_PRINTF(stderr,
                                    "[gotcha] Skipping gotcha binding for %s...\n",
                                    _func.c_str());
                }
                return false;
            }
        }
    }

    const select_list_t& _permit_list = get_permit_list()();
    const select_list_t& _reject_list = get_reject_list()();

    // if function matches a reject_listed entry, do not construct wrapper
    if(_reject_list.count(_func) > 0)
    {
        if(backend::gotcha::get_verbose() >= 1)
        {
            TIMEMORY_PRINTF(stderr,
                            "[gotcha] GOTCHA binding for function '%s' is in reject "
                            "list...\n",
                            _func.c_str());
        }
        return false;
    }

    // if a permit_list was provided, then do not construct wrapper if not in permit
    // list
    if(!_permit_list.empty())
    {
        if(_permit_list.count(_func) == 0)
        {
            if(backend::gotcha::get_verbose() >= 3)
            {
                TIMEMORY_PRINTF(
                    stderr,
                    "[gotcha] GOTCHA binding for function '%s' is not in permit "
                    "list...\n",
                    _func.c_str());
            }
            return false;
        }
    }

    return true;
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N>
void
gotcha<Nt, BundleT, DiffT>::check_error(error_t _ret, const std::string& _prefix)
{
    auto& _data = get_data()[N];
    if(_ret != GOTCHA_SUCCESS && _data.verbose >= 0)
    {
        std::stringstream msg;
        msg << "[" << TIMEMORY_PROJECT_NAME << "][gotcha] " << _prefix << " at index '"
            << N << "' for function '" << _data.wrap_id << "' returned error code "
            << static_cast<int>(_ret) << ": " << backend::gotcha::get_error(_ret) << "\n";
        log::stream(std::cerr, log::color::warning()) << msg.str();
    }
    else if(_data.verbose >= 3)
    {
#if defined(TIMEMORY_USE_GOTCHA)
        std::stringstream msg;
        msg << "[" << TIMEMORY_PROJECT_NAME << "][gotcha] " << _prefix << " :: "
            << "wrapped: " << _data.wrap_id << ", label: " << _data.tool_id << "\n";
        log::stream(std::cerr, log::color::info()) << msg.str();
#endif
    }
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
Ret
gotcha<Nt, BundleT, DiffT>::wrap(Args... _args)
{
    static_assert(N < Nt, "Error! N must be less than Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
    using func_t = Ret (*)(Args...);

    auto& _data = get_data()[N];
    auto  _func = reinterpret_cast<func_t>(gotcha_get_wrappee(_data.wrappee));

    static_assert(std::is_void<operator_type>::value, "operator_type should be void!");

    constexpr bool _prevent_reentry =
        trait::gotcha_trait<trait::prevent_reentry, this_type, N>::value;
    constexpr bool _static_data =
        trait::gotcha_trait<trait::static_data, this_type, N>::value;

    if(!_data.is_active || _data.is_finalized)
        return (*_func)(_args...);

    if constexpr(!_prevent_reentry)
    {
        if constexpr(!std::is_void<Ret>::value)
        {
            if constexpr(_static_data)
            {
                static bool        _once = false;
                static bundle_type _bundle =
                    operation::construct<bundle_type>{}(_data.tool_id);
                if(!_once && (_once = true))
                    operation::set_data<bundle_type>{}(_bundle, get_data()[N]);

                _bundle.construct(_args...);
                _bundle.start();
                _bundle.audit(audit::incoming{}, _args...);
                Ret _ret = invoke<bundle_type>(std::forward<gotcha_data>(_data), _bundle,
                                               _func, std::forward<Args>(_args)...);
                _bundle.audit(audit::outgoing{}, _ret);
                _bundle.stop();
                return _ret;
            }
            else
            {
                bundle_type _bundle{ _data.tool_id };
                _bundle.construct(_args...);
                _bundle.start(_data);
                _bundle.store(_data);
                _bundle.audit(_data, audit::incoming{}, _args...);
                Ret _ret = invoke<bundle_type>(std::forward<gotcha_data>(_data), _bundle,
                                               _func, std::forward<Args>(_args)...);
                _bundle.audit(_data, audit::outgoing{}, _ret);
                _bundle.stop(_data);
                return _ret;
            }
        }
        else
        {
            if constexpr(_static_data)
            {
                static bool        _once = false;
                static bundle_type _bundle =
                    operation::construct<bundle_type>{}(_data.tool_id);
                if(!_once && (_once = true))
                    operation::set_data<bundle_type>{}(_bundle, get_data()[N]);

                _bundle.construct(_args...);
                _bundle.start();
                _bundle.audit(audit::incoming{}, _args...);
                invoke<bundle_type>(std::forward<gotcha_data>(_data), _bundle, _func,
                                    std::forward<Args>(_args)...);
                _bundle.audit(audit::outgoing{});
                _bundle.stop();
            }
            else
            {
                bundle_type _bundle{ _data.tool_id };
                _bundle.construct(_args...);
                _bundle.start(_data);
                _bundle.store(_data);
                _bundle.audit(_data, audit::incoming{}, _args...);
                invoke<bundle_type>(std::forward<gotcha_data>(_data), _bundle, _func,
                                    std::forward<Args>(_args)...);
                _bundle.audit(_data, audit::outgoing{});
                _bundle.stop(_data);
            }
        }
    }
    else
    {
        (void) _static_data;
        // protects against TLS calling malloc when malloc is wrapped
        static bool _protect_tls_alloc = false;

        if(_protect_tls_alloc)
            return (*_func)(_args...);

        if constexpr(!std::is_void<Ret>::value)
        {
            _protect_tls_alloc = true;
            auto _suppress =
                gotcha_suppression::get() || (_data.suppression && *_data.suppression);
            _protect_tls_alloc = false;

            if(!_data.ready || _suppress)
            {
                _protect_tls_alloc                  = true;
                static thread_local bool _recursive = false;
                _protect_tls_alloc                  = false;
                if(!_recursive && _data.debug && *_data.debug)
                {
                    _recursive = true;
                    auto _tid  = threading::get_id();
                    TIMEMORY_PRINTF(
                        stderr,
                        "[T%i][%s] %s is either not ready (ready=%s) or is globally "
                        "suppressed (suppressed=%s)\n",
                        (int) _tid, __FUNCTION__, _data.tool_id.c_str(),
                        (_data.ready) ? "true" : "false", (_suppress) ? "true" : "false");
                    fflush(stderr);
                    _recursive = false;
                }
                return (*_func)(_args...);
            }

            bool did_data_toggle = false;
            bool did_glob_toggle = false;

            // make sure the function is not recursively entered
            // (important for allocation-based wrappers)
            _data.ready = false;
            toggle_suppress_on(_data.suppression, did_data_toggle);

            // bundle_type is always: component_{tuple,list,bundle}
            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            //
            bundle_type _bundle{ _data.tool_id };
            _bundle.construct(_args...);
            _bundle.start(_data);
            _bundle.store(_data);
            _bundle.audit(_data, audit::incoming{}, _args...);
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            _data.ready = true;
            Ret _ret    = invoke<bundle_type>(std::forward<gotcha_data>(_data), _bundle,
                                           _func, std::forward<Args>(_args)...);
            _data.ready = false;

            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            _bundle.audit(_data, audit::outgoing{}, _ret);
            _bundle.stop(_data);
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            // allow re-entrance into wrapper
            toggle_suppress_off(_data.suppression, did_data_toggle);
            _data.ready = true;

            return _ret;
        }
        else
        {
            _protect_tls_alloc = true;
            auto _suppress =
                gotcha_suppression::get() || (_data.suppression && *_data.suppression);
            _protect_tls_alloc = false;

            if(!_data.ready || _suppress)
            {
                _protect_tls_alloc                  = true;
                static thread_local bool _recursive = false;
                _protect_tls_alloc                  = false;
                if(!_recursive && _data.debug && *_data.debug)
                {
                    _recursive = true;
                    auto _tid  = threading::get_id();
                    TIMEMORY_PRINTF(
                        stderr,
                        "[T%i][%s] %s is either not ready (ready=%s) or is globally "
                        "suppressed (suppressed=%s)\n",
                        (int) _tid, __FUNCTION__, _data.tool_id.c_str(),
                        (_data.ready) ? "true" : "false", (_suppress) ? "true" : "false");
                    fflush(stderr);
                    _recursive = false;
                }
                (*_func)(_args...);
                return;
            }

            bool did_data_toggle = false;
            bool did_glob_toggle = false;

            // make sure the function is not recursively entered
            // (important for allocation-based wrappers)
            _data.ready = false;
            toggle_suppress_on(_data.suppression, did_data_toggle);
            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);

            //
            bundle_type _bundle{ _data.tool_id };
            _bundle.construct(_args...);
            _bundle.start(_data);
            _bundle.store(_data);
            _bundle.audit(_data, audit::incoming{}, _args...);
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            _data.ready = true;
            invoke<bundle_type>(std::forward<gotcha_data>(_data), _bundle, _func,
                                std::forward<Args>(_args)...);
            _data.ready = false;

            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            _bundle.audit(_data, audit::outgoing{});
            _bundle.stop(_data);

            // allow re-entrance into wrapper
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);
            toggle_suppress_off(_data.suppression, did_data_toggle);
            _data.ready = true;
        }
    }
#else
    consume_parameters(_args...);
    TIMEMORY_PRINT_HERE("%s", "should not be here!");
#endif
    if constexpr(!std::is_void<Ret>::value)
        return Ret{};
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
Ret
gotcha<Nt, BundleT, DiffT>::replace_func(Args... _args)
{
    static_assert(N < Nt, "Error! N must be less than Nt!");
    static_assert(components_size == 0, "Error! Number of components must be zero!");

#if defined(TIMEMORY_USE_GOTCHA)
    using func_t       = Ret (*)(Args...);
    using replace_type = tim::component_tuple<operator_type>;

    static_assert(!std::is_void<operator_type>::value, "operator_type cannot be void!");

    constexpr bool _prevent_reentry =
        trait::gotcha_trait<trait::prevent_reentry, this_type, N>::value;
    constexpr bool _static_data =
        trait::gotcha_trait<trait::static_data, this_type, N>::value;

    auto& _data = get_data()[N];
    auto  _func = reinterpret_cast<func_t>(gotcha_get_wrappee(_data.wrappee));

    if(!_data.is_active || _data.is_finalized)
        return (*_func)(_args...);

    if constexpr(!std::is_void<Ret>::value)
    {
        if constexpr(_prevent_reentry)
        {
            if(!_data.ready)
                return (*_func)(_args...);
            _data.ready = false;
        }
        Ret _ret{};
        if constexpr(_static_data)
        {
            static replace_type _bundle{ _data.tool_id };
            _ret = invoke<replace_type>(std::forward<gotcha_data>(_data), _bundle, _func,
                                        std::forward<Args>(_args)...);
        }
        else
        {
            replace_type _bundle{};
            _ret = invoke<replace_type>(std::forward<gotcha_data>(_data), _bundle, _func,
                                        std::forward<Args>(_args)...);
        }
        if constexpr(_prevent_reentry)
            _data.ready = true;
        return _ret;
    }
    else
    {
        if(!_data.ready)
            return (*_func)(_args...);

        if constexpr(_prevent_reentry)
            _data.ready = false;
        if constexpr(_static_data)
        {
            static replace_type _bundle{ _data.tool_id };
            invoke<replace_type>(std::forward<gotcha_data>(_data), _bundle, _func,
                                 std::forward<Args>(_args)...);
        }
        else
        {
            replace_type _bundle{};
            invoke<replace_type>(std::forward<gotcha_data>(_data), _bundle, _func,
                                 std::forward<Args>(_args)...);
        }
        if constexpr(_prevent_reentry)
            _data.ready = true;
    }
#else
    consume_parameters(_args...);
    TIMEMORY_PRINT_HERE("%s", "should not be here!");
    if constexpr(!std::is_void<Ret>::value)
        return Ret{};
#endif
}

//----------------------------------------------------------------------------------//

template <size_t Nt, typename BundleT, typename DiffT>
template <size_t N, typename Ret, typename... Args>
Ret
gotcha<Nt, BundleT, DiffT>::fast_func(Args... _args)
{
    static_assert(N < Nt, "Error! N must be less than Nt!");
    static_assert(components_size == 0, "Error! Number of components must be zero!");

#if defined(TIMEMORY_USE_GOTCHA)
    using func_t = Ret (*)(Args...);
    constexpr bool _static_data =
        trait::gotcha_trait<trait::static_data, this_type, N>::value;
    constexpr bool _prevent_reentry =
        trait::gotcha_trait<trait::prevent_reentry, this_type, N>::value;

    static auto _func =
        reinterpret_cast<func_t>(gotcha_get_wrappee(get_data()[N].wrappee));

    if constexpr(_prevent_reentry)
    {
        static auto& _data = get_data()[N];
        if(!_data.is_active)
            return (*_func)(_args...);
    }

    if constexpr(_static_data)
    {
        using static_data_type = policy::static_data<operator_type, this_type>;

        decltype(auto) _v =
            static_data_type{}(std::integral_constant<size_t, N>{}, get_data()[N]);
        return std::invoke(_v, _func, _args...);
    }
    else
    {
        if constexpr(std::is_invocable<operator_type, gotcha_data, func_t,
                                       Args...>::value)
        {
            return operator_type{}(get_data()[N], _func, _args...);
        }
        else if constexpr(std::is_invocable<operator_type,
                                            std::integral_constant<size_t, N>, func_t,
                                            Args...>::value)
        {
            return operator_type{}(std::integral_constant<size_t, N>{}, _func, _args...);
        }
        else
        {
            return operator_type{}(_func, _args...);
        }
    }
#else
    consume_parameters(_args...);
    TIMEMORY_PRINT_HERE("%s", "should not be here!");
    return Ret();
#endif
}

//----------------------------------------------------------------------------------//

}  // namespace component
}  // namespace tim

#endif
