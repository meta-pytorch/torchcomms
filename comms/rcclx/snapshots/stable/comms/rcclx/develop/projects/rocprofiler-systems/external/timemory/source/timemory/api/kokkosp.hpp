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

#include "timemory/api.hpp"
#include "timemory/compat/macros.h"
#include "timemory/components/data_tracker/components.hpp"
#include "timemory/components/user_bundle/types.hpp"
#include "timemory/variadic/component_bundle.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#if !defined(TIMEMORY_KOKKOSP_PREFIX)
#    if defined(TIMEMORY_LIBRARY_SOURCE)
#        define TIMEMORY_KOKKOSP_PREFIX TIMEMORY_WEAK_PREFIX
#    else
#        define TIMEMORY_KOKKOSP_PREFIX
#    endif
#endif

#if !defined(TIMEMORY_KOKKOSP_POSTFIX)
#    if defined(TIMEMORY_LIBRARY_SOURCE)
#        define TIMEMORY_KOKKOSP_POSTFIX                                                 \
            TIMEMORY_WEAK_POSTFIX TIMEMORY_VISIBILITY("default")
#    else
#        define TIMEMORY_KOKKOSP_POSTFIX TIMEMORY_VISIBILITY("default")
#    endif
#endif

struct SpaceHandle
{
    char name[64];
};

struct KokkosPDeviceInfo
{
    uint32_t deviceID;
};

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace kokkosp
{
//--------------------------------------------------------------------------------------//

enum Space
{
    SPACE_HOST,
    SPACE_CUDA
};

//--------------------------------------------------------------------------------------//

enum
{
    NSPACES = 2
};

//--------------------------------------------------------------------------------------//

inline Space
get_space(const SpaceHandle& handle)
{
    switch(handle.name[0])
    {
        case 'H': return SPACE_HOST;
        case 'C': return SPACE_CUDA;
    }
    abort();
    return SPACE_HOST;
}

//--------------------------------------------------------------------------------------//

inline const char*
get_space_name(int space)
{
    switch(space)
    {
        case SPACE_HOST: return "HOST";
        case SPACE_CUDA: return "CUDA";
    }
    abort();
    return nullptr;
}

//--------------------------------------------------------------------------------------//

inline uint64_t
get_unique_id()
{
    static thread_local uint64_t _instance = 0;
    return _instance++;
}

//--------------------------------------------------------------------------------------//

inline std::mutex&
get_cleanup_mutex()
{
    static std::mutex _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline auto&
get_cleanup()
{
    static std::vector<std::function<void()>> _instance{};
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline Tp&
get_tl_static()
{
    // create a thread-local instance
    static thread_local Tp _instance{};
    // on first pass, add to cleanup
    static thread_local bool _init = [&]() {
        get_cleanup_mutex().lock();
        get_cleanup().push_back([&]() { _instance.clear(); });
        get_cleanup_mutex().unlock();
        return true;
    }();
    consume_parameters(_init);

    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline Tp&
get_static()
{
    // create a thread-local instance
    static Tp _instance{};
    // on first pass, add to cleanup
    static bool _init = [&]() {
        get_cleanup_mutex().lock();
        get_cleanup().push_back([&]() { _instance.clear(); });
        get_cleanup_mutex().unlock();
    }();
    consume_parameters(_init);

    return _instance;
}

//--------------------------------------------------------------------------------------//

inline void
cleanup()
{
    get_cleanup_mutex().lock();
    for(auto& itr : get_cleanup())
        itr();
    get_cleanup_mutex().unlock();
}

//--------------------------------------------------------------------------------------//

struct kernel_logger : component::base<kernel_logger, void>
{
public:
    TIMEMORY_DEFAULT_OBJECT(kernel_logger)

    template <typename... Args>
    void mark(int64_t _inc_depth, Args&&... _args)
    {
        if(_inc_depth < 0)
            get_depth() += _inc_depth;
        {
            auto        _msg = TIMEMORY_JOIN('/', std::forward<Args>(_args)...);
            auto_lock_t _lk{ type_mutex<decltype(std::cerr)>() };
            std::cerr << get_indent() << get_message(_msg) << std::endl;
        }
        if(_inc_depth > 0)
            get_depth() += _inc_depth;
    }

public:
    static std::string get_message(string_view_cref_t _msg)
    {
        return std::string{ "[kokkos_kernel_logger]> " } + std::string{ _msg };
    }

    static int64_t& get_depth()
    {
        static int64_t _value = 0;
        return _value;
    }

    static std::string get_indent()
    {
        auto _depth = get_depth();
        if(_depth < 1)
            return "";
        std::stringstream ss;
        ss << std::right << std::setw(_depth * 2) << "";
        return ss.str();
    }
};

//--------------------------------------------------------------------------------------//

using memory_tracker = tim::component::data_tracker<int64_t, tim::project::kokkosp>;
using kokkos_bundle  = tim::component::user_kokkosp_bundle;

using logger_t = tim::component_bundle_t<project::kokkosp, kokkosp::kernel_logger*>;

template <typename... Tail>
using profiler_t =
    tim::component_bundle_t<project::kokkosp, kokkosp::memory_tracker, Tail...>;

template <typename... Tail>
using profiler_section_t = std::tuple<std::string, profiler_t<Tail...>>;

template <typename... Tail>
using profiler_alloc_t = tim::auto_tuple<kokkosp::memory_tracker, Tail...>;

// various data structurs used
template <typename... Tail>
using profiler_stack_t = std::vector<profiler_t<Tail...>>;

template <typename... Tail>
using profiler_memory_map_t =
    std::unordered_map<string_view_t,
                       std::unordered_map<string_view_t, profiler_t<Tail...>>>;

template <typename... Tail>
using profiler_index_map_t = std::unordered_map<uint64_t, profiler_t<Tail...>>;

template <typename... Tail>
using profiler_section_map_t = std::unordered_map<uint64_t, profiler_section_t<Tail...>>;

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline profiler_index_map_t<Tail...>&
get_profiler_index_map()
{
    return get_tl_static<profiler_index_map_t<Tail...>>();
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline profiler_section_map_t<Tail...>&
get_profiler_section_map()
{
    return get_tl_static<profiler_section_map_t<Tail...>>();
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline profiler_memory_map_t<Tail...>&
get_profiler_memory_map()
{
    return get_tl_static<profiler_memory_map_t<Tail...>>();
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline auto&
get_profiler_memory_map(SpaceHandle _space)
{
    return get_profiler_memory_map<Tail...>()[tim::string_view_t{ _space.name }];
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline profiler_stack_t<Tail...>&
get_profiler_stack()
{
    return get_tl_static<profiler_stack_t<Tail...>>();
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline void
create_profiler(const std::string& pname, uint64_t kernid)
{
    get_profiler_index_map<Tail...>().insert({ kernid, profiler_t<Tail...>(pname) });
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline void
destroy_profiler(uint64_t kernid)
{
    if(get_profiler_index_map<Tail...>().find(kernid) !=
       get_profiler_index_map<Tail...>().end())
        get_profiler_index_map<Tail...>().erase(kernid);
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline profiler_t<Tail...>*
get_profiler(uint64_t kernid)
{
    if(get_profiler_index_map<Tail...>().find(kernid) !=
       get_profiler_index_map<Tail...>().end())
        return &get_profiler_index_map<Tail...>().at(kernid);
    return nullptr;
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline void
start_profiler(uint64_t kernid)
{
    if(get_profiler_index_map<Tail...>().find(kernid) !=
       get_profiler_index_map<Tail...>().end())
        get_profiler_index_map<Tail...>().at(kernid).start();
}

//--------------------------------------------------------------------------------------//

template <typename... Tail>
inline void
stop_profiler(uint64_t kernid)
{
    if(get_profiler_index_map<Tail...>().find(kernid) !=
       get_profiler_index_map<Tail...>().end())
        get_profiler_index_map<Tail...>().at(kernid).stop();
}

//--------------------------------------------------------------------------------------//

}  // namespace kokkosp
}  // namespace tim

//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, kokkosp::memory_tracker, std::true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, kokkosp::memory_tracker,
                               std::true_type)

//--------------------------------------------------------------------------------------//

extern "C"
{
    TIMEMORY_KOKKOSP_PREFIX void kokkosp_print_help(char* argv0) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_parse_args(int    argc,
                                                    char** argv) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_declare_metadata(
        const char* key, const char* value) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_init_library(
        const int loadSeq, const uint64_t interfaceVer, const uint32_t devInfoCount,
        void* deviceInfo) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_finalize_library() TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_begin_parallel_for(
        const char* name, uint32_t devid, uint64_t* kernid) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_end_parallel_for(uint64_t kernid)
        TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_begin_parallel_reduce(
        const char* name, uint32_t devid, uint64_t* kernid) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_end_parallel_reduce(uint64_t kernid)
        TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_begin_parallel_scan(
        const char* name, uint32_t devid, uint64_t* kernid) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_end_parallel_scan(uint64_t kernid)
        TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_begin_fence(
        const char* name, uint32_t devid, uint64_t* kernid) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_end_fence(uint64_t kernid)
        TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_push_profile_region(const char* name)
        TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_pop_profile_region() TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_create_profile_section(
        const char* name, uint32_t* secid) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_destroy_profile_section(uint32_t secid)
        TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_start_profile_section(uint32_t secid)
        TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_stop_profile_section(uint32_t secid)
        TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_allocate_data(
        const SpaceHandle space, const char* label, const void* const ptr,
        const uint64_t size) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_deallocate_data(
        const SpaceHandle space, const char* label, const void* const ptr,
        const uint64_t size) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_begin_deep_copy(
        SpaceHandle dst_handle, const char* dst_name, const void* dst_ptr,
        SpaceHandle src_handle, const char* src_name, const void* src_ptr,
        uint64_t size) TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_end_deep_copy() TIMEMORY_KOKKOSP_POSTFIX;

    TIMEMORY_KOKKOSP_PREFIX void kokkosp_profile_event(const char* name)
        TIMEMORY_KOKKOSP_POSTFIX;
}
