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

#include "timemory/library.h"
#include "timemory/timemory.hpp"
//
#include "timemory/components/ompt.hpp"

#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;

using api_t          = TIMEMORY_API;
using ompt_handle_t  = ompt_handle<api_t>;
using ompt_toolset_t = typename ompt_handle_t::toolset_type;
using ompt_bundle_t  = tim::component_tuple<ompt_handle_t>;
using handle_map_t   = std::unordered_map<uint64_t, ompt_bundle_t>;

static bool
setup_ompt()
{
    tim::trait::runtime_enabled<ompt_toolset_t>::set(false);
    return tim::trait::runtime_enabled<ompt_toolset_t>::get();
}

static handle_map_t          f_handle_map;
static std::atomic<uint64_t> f_handle_count;
static uint64_t              global_id      = 0;
static const bool            initial_config = setup_ompt();
static auto                  settings       = tim::settings::shared_instance();

extern "C"
{
    ompt_start_tool_result_t* ompt_start_tool(unsigned int, const char*);

    void timemory_ompt_library_ctor()
    {
        auto ompt   = tim::get_env<std::string>("TIMEMORY_OMPT_COMPONENTS", "");
        auto over   = (ompt.empty()) ? 1 : 0;
        auto trace  = tim::get_env<std::string>("TIMEMORY_TRACE_COMPONENTS", "");
        auto region = tim::get_env<std::string>("TIMEMORY_COMPONENTS", "");
        auto glob   = tim::get_env<std::string>("TIMEMORY_GLOBAL_COMPONENTS", "");
        if(!trace.empty())
            tim::set_env("TIMEMORY_OMPT_COMPONENTS", trace, over);
        else if(!region.empty())
            tim::set_env("TIMEMORY_OMPT_COMPONENTS", region, over);
        else if(!glob.empty())
            tim::set_env("TIMEMORY_OMPT_COMPONENTS", glob, over);
    }

    uint64_t timemory_start_ompt()
    {
        timemory_ompt_library_ctor();
        // provide environment variable for enabling/disabling
        if(tim::get_env<bool>("TIMEMORY_ENABLE_OMPT", true))
        {
            tim::auto_lock_t lk(tim::type_mutex<ompt_handle_t>());
            auto             idx = ++f_handle_count;
            f_handle_map.insert(
                { idx, ompt_bundle_t("openmp", true, tim::scope::get_default()) });
            f_handle_map[idx].start();
            return idx;
        }
        else
        {
            return 0;
        }
    }

    uint64_t timemory_stop_ompt(uint64_t id)
    {
        if(id == global_id)
        {
            for(auto& itr : f_handle_map)
                itr.second.stop();
            f_handle_map.clear();
        }
        else
        {
            auto itr = f_handle_map.find(id);
            if(itr != f_handle_map.end())
            {
                itr->second.stop();
                f_handle_map.erase(itr);
            }
        }
        return f_handle_map.size();
    }

    void timemory_register_ompt()
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s", "");
        global_id = timemory_start_ompt();
    }

    void timemory_deregister_ompt()
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s", "");
        global_id = timemory_stop_ompt(global_id);
    }

    // Below are for FORTRAN codes
    void     timemory_ompt_library_ctor_() {}
    uint64_t timemory_start_ompt_() { return timemory_start_ompt(); }
    uint64_t timemory_stop_ompt_(uint64_t id) { return timemory_stop_ompt(id); }
    void     timemory_register_ompt_() { timemory_register_ompt(); }
    void     timemory_deregister_ompt_() { timemory_deregister_ompt(); }

}  // extern "C"
