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
 * \file timemory/components/gperftools/components.hpp
 * \brief Implementation of the gperftools component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/gperftools/backends.hpp"
#include "timemory/components/gperftools/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
// start/stop gperftools cpu profiler
//
struct gperftools_cpu_profiler : public base<gperftools_cpu_profiler, void>
{
    using value_type = void;
    using this_type  = gperftools_cpu_profiler;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "gperftools_cpu_profiler"; }
    static std::string description()
    {
        return "Control switch for gperftools CPU profiler";
    }
    static value_type record() {}

    static void thread_init() { gperftools::cpu::register_thread(); }

    static void global_finalize()
    {
        if(gperftools::cpu::is_running())
        {
            gperftools::cpu::profiler_flush();
            gperftools::cpu::profiler_stop();
        }
    }

    void start()
    {
        if(!gperftools::cpu::is_running())
        {
            index                 = this_type::get_index()++;
            const auto& _dmp_info = get_dmp_info();
            bool        _dmp_init = std::get<0>(_dmp_info);
            int32_t     _dmp_rank = std::get<1>(_dmp_info);
            auto        _cfg      = settings::compose_filename_config{};
            _cfg.use_suffix       = _dmp_init;
            _cfg.suffix           = _dmp_rank;
            auto fname            = settings::compose_output_filename(
                label() + "_" + std::to_string(index), ".dat", _cfg);
            auto ret = gperftools::cpu::profiler_start(fname);
            if(ret == 0)
            {
                TIMEMORY_PRINTF(stderr, "[gperftools_cpu_profiler] Error starting %s...",
                                fname.c_str());
            }
        }
    }

    void stop()
    {
        if(index >= 0)
        {
            gperftools::cpu::profiler_flush();
            gperftools::cpu::profiler_stop();
        }
    }

private:
    int32_t index = -1;  // if this is >= zero, then we flush and stop

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance(0);
        return _instance;
    }

    using dmp_info_t = std::tuple<bool, int32_t, int32_t>;

    static const dmp_info_t& get_dmp_info()
    {
        static dmp_info_t _info{ dmp::is_initialized(), dmp::rank(), dmp::size() };
        return _info;
    }
};
//
//--------------------------------------------------------------------------------------//
// start/stop gperftools cpu profiler
//
struct gperftools_heap_profiler : public base<gperftools_heap_profiler, void>
{
    using value_type = void;
    using this_type  = gperftools_heap_profiler;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "gperftools_heap_profiler"; }
    static std::string description()
    {
        return "Control switch for the gperftools heap profiler";
    }
    static value_type record() {}

    static void global_finalize()
    {
        if(gperftools::heap::is_running())
        {
            gperftools::heap::profiler_flush("global_finalize");
            gperftools::heap::profiler_stop();
        }
    }

    void start()
    {
        if(!gperftools::heap::is_running())
        {
            index      = this_type::get_index()++;
            auto fname = settings::compose_output_filename(label(), ".dat");
            auto ret   = gperftools::heap::profiler_start(fname);
            if(ret > 0)
            {
                TIMEMORY_PRINTF(stderr, "[gperftools_heap_profiler] Error starting %s...",
                                prefix.c_str());
            }
        }
    }

    void stop()
    {
        if(index >= 0)
        {
            gperftools::heap::profiler_flush(prefix);
            gperftools::heap::profiler_stop();
        }
    }

    void set_prefix(const std::string& _prefix) { prefix = _prefix; }

protected:
    std::string prefix;
    int32_t     index = -1;  // if this is >= zero, then we flush and stop

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance(0);
        return _instance;
    }
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
