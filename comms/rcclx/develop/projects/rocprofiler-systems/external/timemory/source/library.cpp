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

#if !defined(TIMEMORY_LIBRARY_SOURCE)
#    define TIMEMORY_LIBRARY_SOURCE 1
#endif

#include "timemory/compat/library.h"

#include "timemory/config.hpp"
#include "timemory/library.h"
#include "timemory/runtime/configure.hpp"
#include "timemory/timemory.hpp"
#include "timemory/trace.hpp"

#include <cstdarg>
#include <deque>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace tim::component;

//======================================================================================//

extern "C"
{
    timemory_create_func_t timemory_create_function = nullptr;
    timemory_delete_func_t timemory_delete_function = nullptr;
}

//======================================================================================//

using string_t           = std::string;
using library_toolset_t  = TIMEMORY_LIBRARY_TYPE;
using toolset_t          = typename library_toolset_t::component_type;
using region_map_t       = std::unordered_map<std::string, std::stack<uint64_t>>;
using record_map_t       = std::unordered_map<uint64_t, toolset_t>;
using component_enum_t   = std::vector<TIMEMORY_COMPONENT>;
using components_stack_t = std::deque<component_enum_t>;

static std::string spacer =
    "#-------------------------------------------------------------------------#";

//--------------------------------------------------------------------------------------//

static record_map_t&
get_record_map()
{
    static thread_local record_map_t _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

static region_map_t&
get_region_map()
{
    static thread_local region_map_t _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

static components_stack_t&
get_components_stack()
{
    static auto                            _leader_tid = std::this_thread::get_id();
    static components_stack_t              _leader_instance;
    static thread_local components_stack_t _worker_instance = _leader_instance;
    return (std::this_thread::get_id() == _leader_tid) ? _leader_instance
                                                       : _worker_instance;
}

//--------------------------------------------------------------------------------------//
// default components to record
//
inline std::string&
get_default_components()
{
    static std::string _instance =
        tim::get_env<std::string>("TIMEMORY_GLOBAL_COMPONENTS", "wall_clock");
    return _instance;
}

//--------------------------------------------------------------------------------------//
// current set of components being recorded
//
inline component_enum_t&
get_current_components()
{
    auto& _stack = get_components_stack();
    if(_stack.empty())
    {
        _stack.push_back(
            tim::enumerate_components(get_default_components(), "TIMEMORY_COMPONENTS"));
    }
    return _stack.back();
}

//--------------------------------------------------------------------------------------//

std::array<bool, 2>&
get_library_state()
{
    static auto _instance = std::array<bool, 2>({ { false, false } });
    return _instance;
}

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    //----------------------------------------------------------------------------------//
    //  get a unique id
    //
    uint64_t timemory_get_unique_id(void)
    {
        // the maps are thread-local so no concerns for data-race here since
        // two threads updating at once and subsequently losing once of the updates
        // still results in a unique id for that thread
        static uint64_t uniqID = 0;
        return uniqID++;
    }

    //----------------------------------------------------------------------------------//
    //  create a toolset of measurements
    //
    void timemory_create_record(const char* name, uint64_t* id, int n, int* ctypes)
    {
        if(timemory_create_function)
        {
            (*timemory_create_function)(name, id, n, ctypes);
            return;
        }
        // else: provide default behavior

        static thread_local auto& _record_map = get_record_map();
        *id                                   = timemory_get_unique_id();
        _record_map.insert({ *id, toolset_t(name, true) });
        tim::initialize(_record_map[*id], n, ctypes);
        _record_map[*id].start();
        if(_record_map.bucket_count() > _record_map.size())
            _record_map.rehash(_record_map.size() + 10);
    }

    //----------------------------------------------------------------------------------//
    //  destroy a toolset of measurements
    //
    void timemory_delete_record(uint64_t id)
    {
        if(timemory_delete_function)
        {
            (*timemory_delete_function)(id);
        }
        else if(get_record_map().find(id) != get_record_map().end())
        {
            static thread_local auto& _record_map = get_record_map();
            // stop recording, destroy objects, and erase key from map
            _record_map[id].stop();
            _record_map.erase(id);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //
    bool timemory_library_is_initialized(void) { return get_library_state()[0]; }

    //----------------------------------------------------------------------------------//
    //
    //
    void timemory_named_init_library(char* name)
    {
        if(name)
        {
            char* _name[1] = { name };
            timemory_init_library(1, _name);
        }
    }

    //----------------------------------------------------------------------------------//
    //  initialize the library
    //
    void timemory_init_library(int argc, char** argv)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(get_library_state()[0])
            return;
        get_library_state()[0] = true;

        if(argc < 1 && argv)
            argc = 1;
        if(argv == nullptr)
        {
            argc = 0;
        }
        else
        {
            int i = 0;
            for(; i < argc; ++i)
            {
                if(argv[i] == nullptr)
                {
                    argc = i;
                    break;
                }
            }
        }

        if(tim::settings::verbose() > 0)
        {
            printf("%s\n", spacer.c_str());
            printf("\tInitialization of timemory library...\n");
            printf("%s\n\n", spacer.c_str());
        }

        if(argc > 0 && argv)
            tim::timemory_init(argc, argv);
        tim::manager::instance()->update_metadata_prefix();
        // tim::settings::parse();
    }

    //----------------------------------------------------------------------------------//
    //  finalize the library
    void timemory_finalize_library(void)
    {
        tim::trace::lock<tim::trace::library> lk{};
        get_library_state()[1] = true;

        auto  _manager  = tim::manager::master_instance();
        auto* _settings = tim::settings::instance();
        if(!_manager || !_settings)
            return;

        if(_settings->get_enabled() == false && get_record_map().empty())
            return;

        auto& _record_map = get_record_map();

        if(_settings->get_verbose() > 0)
        {
            printf("\n%s\n", spacer.c_str());
            printf("\tFinalization of timemory library...\n");
            printf("%s\n\n", spacer.c_str());
        }

        // put keys into a set so that a potential LD_PRELOAD for timemory_delete_record
        // is called and there is not a concern for the map iterator
        std::unordered_set<uint64_t> keys;
        for(auto& itr : _record_map)
            keys.insert(itr.first);

        // delete all the records
        for(const auto& itr : keys)
            timemory_delete_record(itr);

        // clear the map
        _record_map.clear();

        // have the manager finalize
        if(tim::manager::instance())
            tim::manager::instance()->finalize();

        // do the finalization
        tim::timemory_finalize();

        // just in case
        _settings->get_enabled() = false;

        // set the finalization state to true
        tim::dmp::set_finalized(true);

        // reset manager
        tim::manager::instance().reset();

        // PGI and Intel compilers don't respect destruction order
#if defined(__PGI) || defined(__INTEL_COMPILER)
        _settings->get_output() = false;
#endif
    }

    //----------------------------------------------------------------------------------//
    //  pause the collection
    //
    void timemory_pause(void) { tim::settings::enabled() = false; }

    //----------------------------------------------------------------------------------//
    //  resume the collection
    //
    void timemory_resume(void) { tim::settings::enabled() = true; }

    //----------------------------------------------------------------------------------//

    void timemory_set_default(const char* _component_string)
    {
        tim::trace::lock<tim::trace::library> lk{};
        get_default_components() =
            tim::get_env<std::string>("TIMEMORY_GLOBAL_COMPONENTS", _component_string);
        // tim::set_env("TIMEMORY_COMPONENTS", _component_string, 0);
        static thread_local auto& _stack = get_components_stack();
        if(_stack.empty())
            _stack.push_back(tim::enumerate_components(get_default_components()));
    }

    //----------------------------------------------------------------------------------//

    void timemory_set_environ(const char* evar, const char* eval, int over, int parse)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(evar && eval)
        {
            tim::set_env(evar, eval, over);
            if(parse > 0)
                tim::settings::parse();
        }
    }

    //----------------------------------------------------------------------------------//

    void timemory_add_components(const char* _component_string)
    {
        tim::trace::lock<tim::trace::library> lk{};
        auto&                                 _stack = get_current_components();
        for(auto itr : tim::enumerate_components(_component_string))
            _stack.push_back(itr);
    }

    //----------------------------------------------------------------------------------//

    void timemory_remove_components(const char* _component_string)
    {
        tim::trace::lock<tim::trace::library> lk{};
        auto&                                 _stack = get_current_components();
        for(auto itr : tim::enumerate_components(_component_string))
            _stack.erase(std::remove(_stack.begin(), _stack.end(), itr), _stack.end());
    }

    //----------------------------------------------------------------------------------//

    void timemory_push_components(const char* _component_string)
    {
        tim::trace::lock<tim::trace::library> lk{};
        static thread_local auto&             _stack = get_components_stack();
        _stack.push_back(tim::enumerate_components(_component_string));
    }

    //----------------------------------------------------------------------------------//

    void timemory_push_components_enum(int types, ...)
    {
        tim::trace::lock<tim::trace::library> lk{};
        static thread_local auto&             _stack = get_components_stack();

        component_enum_t comp({ types });
        va_list          args;
        va_start(args, types);
        for(int i = 0; i < TIMEMORY_COMPONENTS_END; ++i)
        {
            auto enum_arg = va_arg(args, int);
            if(enum_arg >= TIMEMORY_COMPONENTS_END)
                break;
            comp.push_back(enum_arg);
        }
        va_end(args);

        _stack.push_back(comp);
    }

    //----------------------------------------------------------------------------------//

    void timemory_pop_components(void)
    {
        tim::trace::lock<tim::trace::library> lk{};
        static thread_local auto&             _stack = get_components_stack();
        if(_stack.size() > 1)
            _stack.pop_back();
    }

    //----------------------------------------------------------------------------------//

    void timemory_begin_record(const char* name, uint64_t* id)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk || tim::settings::enabled() == false)
        {
            *id = std::numeric_limits<uint64_t>::max();
            return;
        }
        auto& comp = get_current_components();
        timemory_create_record(name, id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) *id);
#endif
    }

    //----------------------------------------------------------------------------------//

    void timemory_begin_record_types(const char* name, uint64_t* id, const char* ctypes)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk || tim::settings::enabled() == false)
        {
            *id = std::numeric_limits<uint64_t>::max();
            return;
        }

        auto comp = tim::enumerate_components(std::string(ctypes));
        timemory_create_record(name, id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) *id);
#endif
    }

    //----------------------------------------------------------------------------------//

    void timemory_begin_record_enum(const char* name, uint64_t* id, ...)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk || tim::settings::enabled() == false)
        {
            *id = std::numeric_limits<uint64_t>::max();
            return;
        }

        component_enum_t comp;
        va_list          args;
        va_start(args, id);
        for(int i = 0; i < TIMEMORY_COMPONENTS_END; ++i)
        {
            auto enum_arg = va_arg(args, int);
            if(enum_arg >= TIMEMORY_COMPONENTS_END)
                break;
            comp.push_back(enum_arg);
        }
        va_end(args);

        timemory_create_record(name, id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) *id);
#endif
    }

    //----------------------------------------------------------------------------------//

    uint64_t timemory_get_begin_record(const char* name)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk || tim::settings::enabled() == false)
            return std::numeric_limits<uint64_t>::max();

        uint64_t id   = 0;
        auto&    comp = get_current_components();
        timemory_create_record(name, &id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) id);
#endif

        return id;
    }

    //----------------------------------------------------------------------------------//

    uint64_t timemory_get_begin_record_types(const char* name, const char* ctypes)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk || tim::settings::enabled() == false)
            return std::numeric_limits<uint64_t>::max();

        uint64_t id   = 0;
        auto     comp = tim::enumerate_components(std::string(ctypes));
        timemory_create_record(name, &id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) id);
#endif

        return id;
    }

    //----------------------------------------------------------------------------------//

    uint64_t timemory_get_begin_record_enum(const char* name, ...)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk || tim::settings::enabled() == false)
            return std::numeric_limits<uint64_t>::max();

        uint64_t id = 0;

        component_enum_t comp;
        va_list          args;
        va_start(args, name);
        for(int i = 0; i < TIMEMORY_COMPONENTS_END; ++i)
        {
            auto enum_arg = va_arg(args, int);
            if(enum_arg >= TIMEMORY_COMPONENTS_END)
                break;
            comp.push_back(enum_arg);
        }
        va_end(args);

        timemory_create_record(name, &id, comp.size(), (int*) (comp.data()));

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("beginning record for '%s' (id = %lli)...\n", name,
                   (long long int) id);
#endif

        return id;
    }

    //----------------------------------------------------------------------------------//

    void timemory_end_record(uint64_t id)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk || id == std::numeric_limits<uint64_t>::max())
            return;

        timemory_delete_record(id);

#if defined(DEBUG)
        if(tim::settings::verbose() > 2)
            printf("ending record for %lli...\n", (long long int) id);
#endif
    }

    //----------------------------------------------------------------------------------//

    void timemory_push_region(const char* name)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk)
            return;
        auto& region_map = get_region_map();
        lk.release();
        auto idx = timemory_get_begin_record(name);
        lk       = tim::trace::lock<tim::trace::library>{};
        region_map[name].push(idx);
    }

    //----------------------------------------------------------------------------------//

    void timemory_pop_region(const char* name)
    {
        tim::trace::lock<tim::trace::library> lk{};
        if(!lk)
            return;
        auto& region_map = get_region_map();
        auto  itr        = region_map.find(name);
        if(itr == region_map.end() || (itr != region_map.end() && itr->second.empty()))
            fprintf(stderr, "Warning! region '%s' does not exist!\n", name);
        else
        {
            uint64_t idx = itr->second.top();
            lk.release();
            timemory_end_record(idx);
            lk = tim::trace::lock<tim::trace::library>{};
            itr->second.pop();
        }
    }

    //==================================================================================//
    //
    //      Symbols for Fortran
    //
    //==================================================================================//

    void timemory_create_record_(const char* name, uint64_t* id, int n, int* ct)
    {
        timemory_create_record(name, id, n, ct);
    }

    void timemory_delete_record_(uint64_t id) { timemory_delete_record(id); }

    void timemory_init_library_(int argc, char** argv)
    {
        timemory_init_library(argc, argv);
    }

    void timemory_finalize_library_(void) { timemory_finalize_library(); }

    void timemory_set_default_(const char* components)
    {
        timemory_set_default(components);
    }

    void timemory_push_components_(const char* components)
    {
        timemory_push_components(components);
    }

    void timemory_pop_components_(void) { timemory_pop_components(); }

    void timemory_begin_record_(const char* name, uint64_t* id)
    {
        timemory_begin_record(name, id);
    }

    void timemory_begin_record_types_(const char* name, uint64_t* id, const char* ctypes)
    {
        timemory_begin_record_types(name, id, ctypes);
    }

    uint64_t timemory_get_begin_record_(const char* name)
    {
        return timemory_get_begin_record(name);
    }

    uint64_t timemory_get_begin_record_types_(const char* name, const char* ctypes)
    {
        return timemory_get_begin_record_types(name, ctypes);
    }

    void timemory_end_record_(uint64_t id) { return timemory_end_record(id); }

    void timemory_push_region_(const char* name) { return timemory_push_region(name); }

    void timemory_pop_region_(const char* name) { return timemory_pop_region(name); }

    //======================================================================================//

}  // extern "C"
