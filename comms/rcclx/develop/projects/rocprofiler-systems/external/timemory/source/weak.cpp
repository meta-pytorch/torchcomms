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

#include "timemory/backends/process.hpp"
#include "timemory/compat/library.h"
#include "timemory/components/ompt/types.hpp"
#include "timemory/environment.hpp"
#include "timemory/library.h"
#include "timemory/settings.hpp"

#include <cstdint>
#include <limits>
#include <numeric>

#if !defined(TIMEMORY_WINDOWS)
#    include <dlfcn.h>
#endif

// Macro for obtaining jump pointer function association
#if defined(TIMEMORY_WINDOWS)
#    define DLSYM_FUNCTION(VARNAME, HANDLE, FUNCNAME)
#else
#    define DLSYM_FUNCTION(VARNAME, HANDLE, FUNCNAME)                                    \
        if(HANDLE)                                                                       \
        {                                                                                \
            *(void**) (&VARNAME) = dlsym(HANDLE, FUNCNAME.c_str());                      \
            if(VARNAME == nullptr)                                                       \
            {                                                                            \
                fprintf(stderr, "[%s][pid=%i]> %s\n", FUNCNAME.c_str(),                  \
                        tim::process::get_id(), dlerror());                              \
            }                                                                            \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            VARNAME = nullptr;                                                           \
        }
#endif

#if !defined(OS_DYNAMIC_LIBRARY_EXT)
#    if defined(TIMEMORY_MACOS)
#        define OS_DYNAMIC_LIBRARY_EXT "dylib"
#    elif defined(TIMEMORY_WINDOWS)
#        define OS_DYNAMIC_LIBRARY_EXT "dll"
#    else
#        define OS_DYNAMIC_LIBRARY_EXT "so"
#    endif
#endif

struct tools_stubs_dlsym
{
    using ctor_function_t       = void (*)();
    using register_function_t   = void (*)();
    using deregister_function_t = void (*)();
    using start_function_t      = uint64_t (*)();
    using stop_function_t       = uint64_t (*)(uint64_t);

    TIMEMORY_DEFAULT_OBJECT(tools_stubs_dlsym)

    tools_stubs_dlsym(std::string id, std::string libname = "")
    : m_id{ std::move(id) }
    , m_libname{ std::move(libname) }
    {
        TIMEMORY_DEBUG_PRINT_HERE("id: %s, libname: %s", m_id.c_str(), m_libname.c_str());
        load();
    }

    void load()
    {
#if !defined(TIMEMORY_WINDOWS)
        if(m_libname.empty())
            m_libname =
                TIMEMORY_JOIN("", "libtimemory-", m_id, '.', OS_DYNAMIC_LIBRARY_EXT);

        auto* libhandle = dlopen(m_libname.c_str(), RTLD_LAZY | RTLD_GLOBAL);

        if(!libhandle)
        {
            if(tim::settings::debug())
                fprintf(stderr, "%s\n", dlerror());
        }

        dlerror(); /* Clear any existing error */

        auto ctor_name       = TIMEMORY_JOIN("", "timemory_", m_id, "_library_ctor");
        auto register_name   = TIMEMORY_JOIN("", "timemory_register_", m_id);
        auto deregister_name = TIMEMORY_JOIN("", "timemory_deregister_", m_id);
        auto start_name      = TIMEMORY_JOIN("", "timemory_start_", m_id);
        auto stop_name       = TIMEMORY_JOIN("", "timemory_stop_", m_id);

        // Initialize all pointers
        DLSYM_FUNCTION(m_ctor, libhandle, ctor_name);
        DLSYM_FUNCTION(m_register, libhandle, register_name);
        DLSYM_FUNCTION(m_deregister, libhandle, deregister_name);
        DLSYM_FUNCTION(m_start, libhandle, start_name);
        DLSYM_FUNCTION(m_stop, libhandle, stop_name);
#endif
    }

    void invoke_ctor()
    {
        if(m_ctor)
            (*m_ctor)();
    }

    void invoke_register()
    {
        if(m_register)
            (*m_register)();
    }

    void invoke_deregister()
    {
        if(m_deregister)
            (*m_deregister)();
    }

    uint64_t invoke_start()
    {
        if(m_start)
            return (*m_start)();
        return 0;
    }

    uint64_t invoke_stop(uint64_t val)
    {
        if(m_stop)
            return (*m_stop)(val);
        return 0;
    }

private:
    std::string           m_id         = {};
    std::string           m_libname    = {};
    ctor_function_t       m_ctor       = nullptr;
    register_function_t   m_register   = nullptr;
    deregister_function_t m_deregister = nullptr;
    start_function_t      m_start      = nullptr;
    stop_function_t       m_stop       = nullptr;
};

enum TOOL_STUB_IDS
{
    mpip_idx = 0,
    ompt_idx,
    ncclp_idx,
    mallocp_idx,
};

template <int Idx>
tools_stubs_dlsym*
get_tool_stubs();

#define TOOL_DLSYM_SPECIALIZAITON(INDEX, ID, ENV_VAR)                                    \
    template <>                                                                          \
    tools_stubs_dlsym* get_tool_stubs<INDEX>()                                           \
    {                                                                                    \
        static auto _instance = std::unique_ptr<tools_stubs_dlsym>{};                    \
        if(!_instance)                                                                   \
        {                                                                                \
            auto _env = tim::get_env<std::string>(ENV_VAR, "");                          \
            _instance = std::make_unique<tools_stubs_dlsym>(ID, _env);                   \
        }                                                                                \
        return _instance.get();                                                          \
    }

TOOL_DLSYM_SPECIALIZAITON(mpip_idx, "mpip", "TIMEMORY_MPIP_LIBRARY")
TOOL_DLSYM_SPECIALIZAITON(ompt_idx, "ompt", "TIMEMORY_OMPT_LIBRARY")
TOOL_DLSYM_SPECIALIZAITON(ncclp_idx, "ncclp", "TIMEMORY_NCCLP_LIBRARY")
TOOL_DLSYM_SPECIALIZAITON(mallocp_idx, "mallocp", "TIMEMORY_MALLOCP_LIBRARY")

#define TOOL_INDEX(NAME) NAME##_idx
#define TOOL_DLSYM(NAME) get_tool_stubs<TOOL_INDEX(NAME)>()
#define TOOL_PREFIX TIMEMORY_WEAK_PREFIX
#define TOOL_SUFFIX TIMEMORY_WEAK_POSTFIX TIMEMORY_VISIBILITY("default")
#define TOOL_STUBS(NAME)                                                                 \
    TOOL_PREFIX                                                                          \
    void timemory_##NAME##_library_ctor() TOOL_SUFFIX;                                   \
    TOOL_PREFIX                                                                          \
    void timemory_register_##NAME() TOOL_SUFFIX;                                         \
    TOOL_PREFIX                                                                          \
    void timemory_deregister_##NAME() TOOL_SUFFIX;                                       \
    TOOL_PREFIX                                                                          \
    uint64_t timemory_start_##NAME() TOOL_SUFFIX;                                        \
    TOOL_PREFIX                                                                          \
    uint64_t timemory_stop_##NAME(uint64_t) TOOL_SUFFIX;                                 \
                                                                                         \
    void     timemory_##NAME##_library_ctor() { TOOL_DLSYM(NAME)->invoke_ctor(); }       \
    void     timemory_register_##NAME() { TOOL_DLSYM(NAME)->invoke_register(); }         \
    void     timemory_deregister_##NAME() { TOOL_DLSYM(NAME)->invoke_deregister(); }     \
    uint64_t timemory_start_##NAME() { return TOOL_DLSYM(NAME)->invoke_start(); }        \
    uint64_t timemory_stop_##NAME(uint64_t v) { return TOOL_DLSYM(NAME)->invoke_stop(v); }

extern "C"
{
    //
    //----------------------------------------------------------------------------------//
    //
    TOOL_STUBS(mpip)
    TOOL_STUBS(ompt)
    TOOL_STUBS(ncclp)
    TOOL_STUBS(mallocp)
    //
    //----------------------------------------------------------------------------------//
    //
}
