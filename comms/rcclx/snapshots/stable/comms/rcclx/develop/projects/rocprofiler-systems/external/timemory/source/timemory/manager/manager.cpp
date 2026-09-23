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

// maybe included directly in header-only mode but pragma once will cause warnings
#ifndef TIMEMORY_MANAGER_MANAGER_CPP_
#define TIMEMORY_MANAGER_MANAGER_CPP_

#include "timemory/manager/manager.hpp"

#include "timemory/backends/process.hpp"
#include "timemory/defines.h"
#include "timemory/log/logger.hpp"
#include "timemory/manager/macros.hpp"
#include "timemory/manager/types.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/process/threading.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/signals.hpp"

//--------------------------------------------------------------------------------------//
//
//          if not using extern and not compiling manager library, everything
//          in this file gets excluded by the pre-processor
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_MANAGER_SOURCE) || defined(TIMEMORY_MANAGER_INLINE)
//
//--------------------------------------------------------------------------------------//
//
#    include "timemory/api.hpp"
#    include "timemory/backends/cpu.hpp"
#    include "timemory/backends/threading.hpp"
#    include "timemory/mpl/policy.hpp"
#    include "timemory/mpl/type_traits.hpp"
#    include "timemory/operations/types/file_output_message.hpp"
#    include "timemory/operations/types/finalize/ctest_notes.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/utility/macros.hpp"
#    include "timemory/version.h"

#    include <algorithm>
#    include <atomic>
#    include <cctype>
#    include <fstream>
#    include <iosfwd>
#    include <locale>
#    include <memory>
#    include <string>
#    include <utility>
#    include <vector>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              manager
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE
manager::manager()
: m_instance_count(f_manager_instance_count()++)
, m_rank(dmp::rank())
, m_thread_index(threading::get_id())
, m_thread_id(std::this_thread::get_id())
, m_hash_ids(get_hash_ids())
, m_hash_aliases(get_hash_aliases())
{
    f_thread_counter()++;

    static bool _once = [=] {
        auto _launch_date =
            get_local_datetime("%D", settings::get_launch_time(TIMEMORY_API{}));
        auto _launch_time =
            get_local_datetime("%H:%M", settings::get_launch_time(TIMEMORY_API{}));
        auto _cpu_info = cpu::get_info();
        auto _user     = get_env<std::string>("USER", "nobody");

        for(auto&& itr : { "SHELL", "HOME", "PWD" })
        {
            auto _var = get_env<std::string>(itr, "");
            if(!_var.empty())
                add_metadata(itr, _var);
        }

        add_metadata("USER", _user);
        add_metadata("LAUNCH_DATE", _launch_date);
        add_metadata("LAUNCH_TIME", _launch_time);
        add_metadata("TIMEMORY_API", demangle<TIMEMORY_API>());

#    if defined(TIMEMORY_VERSION_STRING)
        add_metadata("TIMEMORY_VERSION", TIMEMORY_VERSION_STRING);
#    endif

#    if defined(TIMEMORY_GIT_DESCRIBE)
        add_metadata("TIMEMORY_GIT_DESCRIBE", TIMEMORY_GIT_DESCRIBE);
#    endif

#    if defined(TIMEMORY_GIT_REVISION)
        add_metadata("TIMEMORY_GIT_REVISION", TIMEMORY_GIT_REVISION);
#    endif

        auto _cpu_features = delimit(_cpu_info.features, " \t");
        add_metadata("CPU_MODEL", _cpu_info.model);
        add_metadata("CPU_VENDOR", _cpu_info.vendor);
        add_metadata("CPU_FREQUENCY", _cpu_info.frequency);
        add_metadata("CPU_FEATURES", _cpu_features);
        add_metadata("HW_CONCURRENCY", threading::affinity::hw_concurrency());
        add_metadata("HW_PHYSICAL_CPU", threading::affinity::hw_physicalcpu());

        for(int i = 0; i < 3; ++i)
        {
            auto              _cache_size = cpu::cache_size::get(i + 1);
            std::stringstream _cache_lvl;
            _cache_lvl << "HW_L" << (i + 1) << "_CACHE_SIZE";
            add_metadata(_cache_lvl.str(), _cache_size);
        }

        auto_lock_t _lk{ type_mutex<manager>() };
        // set to zero to we can track if user added metadata
        f_manager_persistent_data().metadata_count = 0;
        return true;
    }();
    consume_parameters(_once);

    if(settings::cpu_affinity())
        threading::affinity::set();
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE
manager::~manager()
{
    auto _remain = --f_manager_instance_count();
    bool _last   = ((get_shared_ptr_pair<this_type, TIMEMORY_API>() != nullptr &&
                   this == get_shared_ptr_pair<this_type, TIMEMORY_API>()->first.get()) ||
                  _remain == 0 || m_instance_count == 0);

    if(_last)
        f_thread_counter().store(0, std::memory_order_relaxed);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::cleanup(const std::string& key)
{
    if(f_debug())
        TIMEMORY_PRINT_HERE("cleaning %s", key.c_str());

    auto _cleanup = [&](auto& _functors) {
        auto _orig = _functors.size();
        auto itr   = _functors.begin();
        for(; itr != _functors.end(); ++itr)
        {
            if(itr->first == key)
                break;
        }

        if(itr != _functors.end())
        {
            itr->second();
            _functors.erase(itr);
        }

        if(f_debug())
            TIMEMORY_PRINT_HERE("%s [size: %i]", "cleaned",
                                (int) (_orig - _functors.size()));
    };

    _cleanup(m_finalizer_cleanups);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::cleanup()
{
    m_is_finalizing = true;
    auto _orig_sz   = m_finalizer_cleanups.size();

    if(f_debug())
    {
        TIMEMORY_PRINT_HERE("%s [size: %i]", "cleaning",
                            (int) m_finalizer_cleanups.size());
    }

    auto _cleanup = [](finalizer_plist_t& _functors) {
        // reverse to delete the most recent additions first
        std::reverse(_functors.begin(), _functors.end());
        // invoke all the functions
        for(auto& itr : _functors)
            itr.second();
        // remove all these functors
        _functors.clear();
    };

    _cleanup(m_finalizer_cleanups);

    if(f_debug())
    {
        TIMEMORY_PRINT_HERE("%s [size: %i]", "cleaned",
                            (int) (_orig_sz - m_finalizer_cleanups.size()));
    }
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::initialize()
{
    if(m_is_initialized)
        return;
    m_is_initialized = true;
    auto_lock_t _lk{ m_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();

    // if no initializers on child thread, then copy from main instance
    if(m_initializers.empty() && threading::get_id() > 0 && master_instance() &&
       this != master_instance().get())
        m_initializers = master_instance()->m_initializers;

    m_initializers.erase(std::remove_if(m_initializers.begin(), m_initializers.end(),
                                        [](auto& itr) { return itr(); }),
                         m_initializers.end());
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::finalize()
{
    auto _settings = m_settings.lock();
    if(!_settings || _settings->get_finalized())
        return;

    auto _print_size = [&](const char* _msg) {
        if(f_debug())
        {
            auto _get_sz = [](auto& _mdata) {
                int _val = 0;
                for(auto& itr : _mdata)
                    _val += itr.second.size();
                return _val;
            };
            TIMEMORY_PRINT_HERE("%s [master: %i/%i, worker: %i/%i, other: %i]", _msg,
                                _get_sz(m_master_cleanup), _get_sz(m_master_finalizers),
                                _get_sz(m_worker_cleanup), _get_sz(m_worker_finalizers),
                                (int) m_pointer_fini.size());
        }
    };

    auto _cleanup = [](finalizer_pmap_t& _pdata) {
        for(auto& _functors : _pdata)
        {
            // reverse to delete the most recent additions first
            std::reverse(_functors.second.begin(), _functors.second.end());
            // invoke all the functions
            for(auto& itr : _functors.second)
                itr.second();
            // remove all these finalizers
            _functors.second.clear();
        }
    };

    auto _finalize = [](finalizer_smap_t& _pdata) {
        for(auto& _functors : _pdata)
        {
            // reverse to delete the most recent additions first
            std::reverse(_functors.second.begin(), _functors.second.end());
            // invoke all the functions
            for(auto& itr : _functors.second)
            {
                itr.second->destroy();
                delete itr.second;
            }
            // remove all these finalizers
            _functors.second.clear();
        }
    };

    m_is_finalizing = true;
    m_rank          = std::max<int32_t>(m_rank, dmp::rank());

    _print_size("finalizing");

    cleanup();

    _print_size("finalizing (pre-storage-cleanup)");

    //
    //  ideally, only one of these will be populated
    //  these clear the stack before outputting
    //
    // finalize workers first
    _cleanup(m_worker_cleanup);
    // finalize masters second
    _cleanup(m_master_cleanup);

    _print_size("finalizing (pre-storage-finalization)");

    //
    //  ideally, only one of these will be populated
    //
    // finalize workers first
    _finalize(m_worker_finalizers);
    // finalize masters second
    _finalize(m_master_finalizers);

    _print_size("finalizing (pre-pointer-finalization)");

    for(auto& itr : m_pointer_fini)
        itr.second();

    m_pointer_fini.clear();

    m_is_finalizing = false;

    _print_size("finalized");

    if(m_instance_count == 0 && m_rank == 0)
    {
        operation::finalize::ctest_notes<manager>::get_notes().reset();
        internal_write_metadata();
    }

    m_is_finalized = true;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::exit_hook()
{
    if(f_debug())
        TIMEMORY_PRINT_HERE("%s", "finalizing...");

    if(!manager::f_use_exit_hook())
        return;

    try
    {
        auto master_count = f_manager_instance_count().load();
        if(master_count > 0)
        {
            auto master_manager =
                get_shared_ptr_pair_main_instance<manager, TIMEMORY_API>();
            if(master_manager)
            {
                master_manager->internal_write_metadata("manager::exit_hook");
                master_manager.reset();
            }
        }
    } catch(...)
    {}

    if(f_debug())
        TIMEMORY_PRINT_HERE("%s", "finalizing...");
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::update_metadata_prefix()
{
    m_rank = std::max<int32_t>(m_rank, dmp::rank());
    if(m_write_metadata < 0)
        return;
    auto _settings = f_settings();
    if(!_settings)
        return;
    auto _outp_prefix = _settings->get_global_output_prefix();
    m_metadata_prefix = _outp_prefix;
    if(f_debug())
        TIMEMORY_PRINT_HERE("[rank=%i][id=%i] metadata prefix: '%s'", m_rank,
                            m_instance_count, m_metadata_prefix.c_str());
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE std::ostream&
                        manager::write_metadata(std::ostream& ofs)
{
    auto_lock_t _lk(type_mutex<manager>(), std::defer_lock);

    if(!_lk.owns_lock())
        _lk.lock();

    auto _settings = f_settings();

    // ensure json write final block during destruction before the file is closed
    using policy_type = policy::output_archive_t<manager>;
    auto oa           = policy_type::get(ofs);
    oa->setNextName(TIMEMORY_PROJECT_NAME);
    oa->startNode();
    {
        oa->setNextName("metadata");
        oa->startNode();
        // user
        {
            auto& _info_metadata = f_manager_persistent_data().info_metadata;
            auto& _func_metadata = f_manager_persistent_data().func_metadata;
            oa->setNextName("info");
            oa->startNode();
            for(const auto& itr : _info_metadata)
                (*oa)(cereal::make_nvp(itr.first.c_str(), itr.second));
            for(const auto& itr : _func_metadata)
                itr(static_cast<void*>(oa.get()));
            oa->finishNode();
        }
        // settings
        if(_settings)
        {
            settings::serialize_settings(*oa, *(_settings.get()));
        }
        // output
        {
            oa->setNextName("output");
            oa->startNode();
            for(const auto& itr : m_output_files)
                (*oa)(cereal::make_nvp(itr.first.c_str(), itr.second));
            oa->finishNode();
        }
        // environment
        {
            env_settings::serialize_environment(*oa);
        }
        //
        oa->finishNode();
    }
    oa->finishNode();
    return ofs;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::write_metadata(const std::string& _output_dir, const char* context,
                        settings::compose_filename_config _cfg)
{
    if(m_rank != 0)
    {
        if(f_debug())
            TIMEMORY_PRINT_HERE("[%s]> metadata disabled for rank %i", context,
                                (int) m_rank);
        return;
    }

    if(tim::get_env<bool>("TIMEMORY_CXX_PLOT_MODE", false))
    {
        if(f_debug())
            TIMEMORY_PRINT_HERE("[%s]> plot mode enabled. Skipping metadata", context);
        return;
    }

    _cfg.explicit_path = _output_dir;

    auto fname = settings::compose_output_filename("metadata", "json", _cfg);
    auto hname = settings::compose_output_filename("functions", "json", _cfg);

    auto _settings = f_settings();
    auto _banner   = (_settings) ? _settings->get_banner() : false;

    auto_lock_t _lk{ type_mutex<manager>(), std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();

    std::map<std::string, std::set<size_t>> _hashes{};
    if(m_hash_ids && m_hash_aliases)
    {
        for(const auto& itr : (*m_hash_aliases))
        {
            auto hitr = m_hash_ids->find(itr.second);
            if(hitr != m_hash_ids->end())
            {
                _hashes[operation::decode<TIMEMORY_API>{}(hitr->second)].insert(
                    itr.first);
                _hashes[operation::decode<TIMEMORY_API>{}(hitr->second)].insert(
                    hitr->first);
            }
        }
        for(const auto& itr : (*m_hash_ids))
            _hashes[operation::decode<TIMEMORY_API>{}(itr.second)].insert(itr.first);
    }

    // if there were no hashes generated, writing metadata is optional and the user
    // did not add to metadata -> return
    if(_hashes.empty() && m_write_metadata < 1 &&
       (f_manager_persistent_data().metadata_count == 0 || m_output_files.empty()))
        return;

    // if file output is disabled, return
    if(!f_settings()->file_output())
        return;

    {
        auto _fom = operation::file_output_message<manager>{};
        if((f_verbose() >= 0 || _banner || f_debug()) && !_hashes.empty())
            _fom(std::vector<std::string>{ fname, hname },
                 std::vector<std::string>{ "metadata" });
        else if((f_verbose() >= 0 || _banner || f_debug()) && _hashes.empty())
            _fom(std::vector<std::string>{ fname },
                 std::vector<std::string>{ "metadata" });

        std::ofstream ofs{};
        if(filepath::open(ofs, fname))
            write_metadata(ofs);
        if(ofs)
            ofs << std::endl;
        else
            _fom.append("Warning! Error opening '%s'...", fname.c_str());
        ofs.close();
    }

    if(_hashes.empty())
        return;

    std::ofstream hfs{};
    if(filepath::open(hfs, hname))
    {
        // ensure json write final block during destruction before the file is closed
        using policy_type = policy::output_archive_t<manager>;
        auto oa           = policy_type::get(hfs);
        oa->setNextName(TIMEMORY_PROJECT_NAME);
        oa->startNode();
        {
            oa->setNextName("functions");
            oa->startNode();
            // hash-keys
            {
                for(const auto& itr : _hashes)
                    (*oa)(cereal::make_nvp(itr.first.c_str(), itr.second));
            }
            //
            oa->finishNode();
        }
        oa->finishNode();
    }
    if(hfs)
        hfs << std::endl;
    else
        printf("[manager]> Warning! Error opening '%s'...\n", hname.c_str());

    hfs.close();
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::internal_write_metadata(const char* context)
{
    if(m_rank != 0)
    {
        if(f_debug())
            TIMEMORY_PRINT_HERE("[%s]> metadata disabled for rank %i", context,
                                (int) m_rank);
        return;
    }

    if(m_num_entries < 1 && m_write_metadata < 1)
    {
        if(f_debug())
            TIMEMORY_PRINT_HERE("[%s]> No components generated output. Skipping metadata",
                                context);
        return;
    }

    if(tim::get_env<bool>("TIMEMORY_CXX_PLOT_MODE", false))
    {
        if(f_debug())
            TIMEMORY_PRINT_HERE("[%s]> plot mode enabled. Skipping metadata", context);
        return;
    }
    auto _settings = f_settings();
    if(!_settings)
    {
        if(f_debug())
            TIMEMORY_PRINT_HERE("[%s]> Null pointer to settings", context);
        return;
    }

    bool _auto_output = _settings->get_auto_output();
    bool _file_output = _settings->get_file_output();
    auto _outp_prefix = _settings->get_global_output_prefix();

    static bool written = false;
    if(written || m_write_metadata < 1)
    {
        if(f_debug() && written)
            TIMEMORY_PRINT_HERE("[%s]> metadata already written", context);
        if(f_debug() && m_write_metadata < 1)
            TIMEMORY_PRINT_HERE("[%s]> metadata disabled: %i", context,
                                (int) m_write_metadata);
        return;
    }

    if((!_auto_output || !_file_output) && m_write_metadata < 1)
    {
        if(f_debug() && !_auto_output)
            TIMEMORY_PRINT_HERE("[%s]> metadata disabled because auto output disabled",
                                context);
        if(f_debug() && !_file_output)
            TIMEMORY_PRINT_HERE("[%s]> metadata disabled because file output disabled",
                                context);
        return;
    }

    written          = true;
    m_write_metadata = -1;

    if(f_debug())
        TIMEMORY_PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    // get the output prefix if not already set
    if(m_metadata_prefix.empty())
        m_metadata_prefix = _outp_prefix;

    if(f_debug())
        TIMEMORY_PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    // remove any non-ascii characters
    auto only_ascii = [](char c) { return isascii(c) == 0; };
    m_metadata_prefix.erase(
        std::remove_if(m_metadata_prefix.begin(), m_metadata_prefix.end(), only_ascii),
        m_metadata_prefix.end());

    if(f_debug())
        TIMEMORY_PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    // if empty, set to default
    if(m_metadata_prefix.empty())
        m_metadata_prefix = "timemory-output/";

    if(f_debug())
        TIMEMORY_PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    // if first char is a control character, the statics probably got deleted
    std::locale _lc{};
    if(std::iscntrl(m_metadata_prefix[0], _lc))
        m_metadata_prefix = "timemory-output/";

    if(f_debug())
        TIMEMORY_PRINT_HERE("metadata prefix: '%s'", m_metadata_prefix.c_str());

    write_metadata(m_metadata_prefix, context);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::add_file_output(const string_t& _category, const string_t& _label,
                         const string_t& _file)
{
    auto_lock_t _lk{ m_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    m_output_files[_category][_label].insert(_file);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::add_text_output(const string_t& _label, const string_t& _file)
{
    add_file_output("text", _label, _file);
    auto _settings = f_settings();
    if(_settings && _settings->get_ctest_notes())
        operation::finalize::ctest_notes<manager>::get_notes()->insert(_file);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::add_json_output(const string_t& _label, const string_t& _file)
{
    add_file_output("json", _label, _file);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::remove_cleanup(void* _key)
{
    if(m_pointer_fini.find(_key) != m_pointer_fini.end())
        m_pointer_fini.erase(_key);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::remove_cleanup(const std::string& _key)
{
    auto _remove_functor = [&](finalizer_plist_t& _functors) {
        for(auto itr = _functors.begin(); itr != _functors.end(); ++itr)
        {
            if(itr->first == _key)
            {
                _functors.erase(itr);
                return;
            }
        }
    };

    _remove_functor(m_finalizer_cleanups);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE std::vector<manager::finalizer_func_t>
                        manager::remove_finalizer(const std::string& _key)
{
    auto _finalizers = std::vector<manager::finalizer_func_t>{};

    auto _remove_cleanup = [&](finalizer_pmap_t& _pdata) {
        for(auto& _functors : _pdata)
        {
            for(auto itr = _functors.second.begin(); itr != _functors.second.end(); ++itr)
            {
                if(itr->first == _key)
                {
                    _finalizers.emplace_back(itr->second);
                    _functors.second.erase(itr);
                }
            }
        }
    };

    auto _remove_finalizer = [&](finalizer_smap_t& _pdata) {
        for(auto& _functors : _pdata)
        {
            for(auto itr = _functors.second.begin(); itr != _functors.second.end(); ++itr)
            {
                if(itr->first == _key)
                {
                    auto* _obj = itr->second;
                    _finalizers.emplace_back([_obj]() { _obj->destroy(); });
                    _functors.second.erase(itr);
                }
            }
        }
    };

    _remove_cleanup(m_master_cleanup);
    _remove_cleanup(m_worker_cleanup);
    _remove_finalizer(m_master_finalizers);
    _remove_finalizer(m_worker_finalizers);

    return _finalizers;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::add_metadata(const std::string& _key, const char* _value)
{
    auto_lock_t _lk(type_mutex<manager>());
    ++f_manager_persistent_data().metadata_count;
    f_manager_persistent_data().info_metadata.insert({ _key, std::string{ _value } });
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::add_metadata(const std::string& _key, const std::string& _value)
{
    auto_lock_t _lk(type_mutex<manager>());
    ++f_manager_persistent_data().metadata_count;
    f_manager_persistent_data().info_metadata.insert({ _key, _value });
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::add_synchronization(const std::string& _key, int64_t _id,
                             std::function<void()> _func)
{
    auto_lock_t _lk{ m_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    m_synchronize[_key].emplace(_id, std::move(_func));
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::remove_synchronization(const std::string& _key, int64_t _id)
{
    auto_lock_t _lk{ m_mutex, std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    if(m_synchronize[_key].find(_id) != m_synchronize[_key].end())
        m_synchronize[_key].erase(_id);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
manager::synchronize()
{
    for(auto& itr : m_synchronize)
    {
        for(auto& fitr : itr.second)
        {
            fitr.second();
        }
    }
}
//
//----------------------------------------------------------------------------------//
//
// persistent data for instance counting, threading counting, and exit-hook control
//
TIMEMORY_MANAGER_INLINE manager::persistent_data&
                        manager::f_manager_persistent_data()
{
    static persistent_data _instance{};
    return _instance;
}
//
//----------------------------------------------------------------------------------//
//
// get either master or thread-local instance
//
TIMEMORY_MANAGER_INLINE manager::pointer_t
                        manager::instance()
{
    static thread_local auto _v =
        get_shared_ptr_pair<manager, TIMEMORY_API>()
            ? get_shared_ptr_pair<manager, TIMEMORY_API>()->second
            : pointer_t{};

    static thread_local auto _dtor = scope::destructor{ []() {
        auto*                _settings= settings::instance();
        if(_v && _settings && !_settings->get_finalized())
            _v->finalize();
    } };

    return _v;
}
//
//----------------------------------------------------------------------------------//
//
// get master instance
//
TIMEMORY_MANAGER_INLINE manager::pointer_t
                        manager::master_instance()
{
    return get_shared_ptr_pair<manager, TIMEMORY_API>()
               ? get_shared_ptr_pair<manager, TIMEMORY_API>()->first
               : pointer_t{};
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE bool
manager::get_is_main_thread()
{
    if(!master_instance())
        return true;
    return (std::this_thread::get_id() == master_instance()->m_thread_id);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE manager*
                        timemory_manager_master_instance()
{
    auto& _pinst = tim::get_shared_ptr_pair<manager, TIMEMORY_API>();
    manager::set_persistent_master(_pinst->first);
    return (_pinst) ? _pinst->first.get() : nullptr;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_MANAGER_INLINE void
timemory_library_constructor()
{
    static auto _preloaded = []() {
        auto library_ctor = tim::get_env<bool>("TIMEMORY_LIBRARY_CTOR", true, false);
        if(!library_ctor)
            return true;

        auto ld_preload   = tim::get_env<std::string>("LD_PRELOAD", "", false);
        auto dyld_preload = tim::get_env<std::string>("DYLD_INSERT_LIBRARIES", "", false);
        std::regex lib_regex("libtimemory");
        if(std::regex_search(ld_preload, lib_regex) ||
           std::regex_search(dyld_preload, lib_regex))
        {
            tim::set_env("TIMEMORY_LIBRARY_CTOR", "OFF", 1);
            return true;
        }
        return false;
    }();

    if(_preloaded)
        return;

    static std::atomic<int32_t> _once{ 0 };
    if(_once++ == 0)
    {
        auto _settings                 = tim::settings::shared_instance();
        auto _strict_v                 = _settings->get_strict_config();
        _settings->get_strict_config() = false;
        _settings->init_config();
        _settings->get_strict_config() = _strict_v;
    }
    (void) timemory_manager_master_instance();
    (void) manager::master_instance();
    (void) manager::instance();

    if(tim::get_shared_ptr_pair<manager, TIMEMORY_API>())
        manager::set_persistent_master(
            tim::get_shared_ptr_pair<manager, TIMEMORY_API>()->first);
}
//
//--------------------------------------------------------------------------------------//
//
namespace
{
auto timemory_library_is_constructed = (timemory_library_constructor(), true);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#endif

// formerly in manager/extern.cpp
#if defined(TIMEMORY_MANAGER_SOURCE)
#    include "timemory/manager/extern.hpp"
#    include "timemory/manager/manager.hpp"
#    include "timemory/manager/types.hpp"
//
#    include "timemory/environment.hpp"
#    include "timemory/hash.hpp"
#    include "timemory/plotting.hpp"
#    include "timemory/settings.hpp"
#endif

#endif  // TIMEMORY_MANAGER_MANAGER_CPP_
