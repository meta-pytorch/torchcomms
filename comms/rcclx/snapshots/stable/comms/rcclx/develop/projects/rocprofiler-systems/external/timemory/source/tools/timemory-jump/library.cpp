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

#include "timemory/backends/process.hpp"
#include "timemory/environment.hpp"
#include "timemory/macros/os.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <string>

// Macro for obtaining jump pointer function association
#define DLSYM_JUMP_FUNCTION(VARNAME, HANDLE, FUNCNAME)                                   \
    if(HANDLE)                                                                           \
    {                                                                                    \
        *(void**) (&VARNAME) = dlsym(HANDLE, FUNCNAME);                                  \
        if(VARNAME == nullptr)                                                           \
        {                                                                                \
            fprintf(stderr, "[timemory-jump@%s][pid=%i]> %s\n", FUNCNAME,                \
                    tim::process::get_id(), dlerror());                                  \
        }                                                                                \
    }                                                                                    \
    else                                                                                 \
    {                                                                                    \
        VARNAME = nullptr;                                                               \
    }

//--------------------------------------------------------------------------------------//

// This class contains jump pointers for timemory's dyninst functions
class jump
{
public:
    void (*timemory_push_components_jump)(const char*);  // NOLINT
    void (*timemory_pop_components_jump)(void);          // NOLINT

    void (*timemory_push_region_jump)(const char*);  // NOLINT
    void (*timemory_pop_region_jump)(const char*);   // NOLINT

    void (*timemory_add_hash_id_jump)(uint64_t, const char*);          // NOLINT
    void (*timemory_push_trace_jump)(const char*);                     // NOLINT
    void (*timemory_pop_trace_jump)(const char*);                      // NOLINT
    void (*timemory_push_trace_hash_jump)(uint64_t);                   // NOLINT
    void (*timemory_pop_trace_hash_jump)(uint64_t);                    // NOLINT
    void (*timemory_trace_init_jump)(const char*, bool, const char*);  // NOLINT
    void (*timemory_trace_finalize_jump)(void);                        // NOLINT
    void (*timemory_trace_set_env_jump)(const char*, const char*);     // NOLINT
    void (*timemory_trace_set_mpi_jump)(bool, bool);                   // NOLINT

    explicit jump(std::string&& libpath)
    {
        auto* libhandle = dlopen(libpath.c_str(), RTLD_LAZY);

        if(!libhandle)
            fprintf(stderr, "%s\n", dlerror());

        dlerror(); /* Clear any existing error */

        /* Initialize all pointers */
        DLSYM_JUMP_FUNCTION(timemory_push_components_jump, libhandle,
                            "timemory_push_components");

        DLSYM_JUMP_FUNCTION(timemory_pop_components_jump, libhandle,
                            "timemory_pop_components");

        DLSYM_JUMP_FUNCTION(timemory_push_region_jump, libhandle, "timemory_push_region");

        DLSYM_JUMP_FUNCTION(timemory_pop_region_jump, libhandle, "timemory_pop_region");

        DLSYM_JUMP_FUNCTION(timemory_add_hash_id_jump, libhandle, "timemory_add_hash_id");

        DLSYM_JUMP_FUNCTION(timemory_push_trace_jump, libhandle, "timemory_push_trace");

        DLSYM_JUMP_FUNCTION(timemory_pop_trace_jump, libhandle, "timemory_pop_trace");

        DLSYM_JUMP_FUNCTION(timemory_push_trace_hash_jump, libhandle,
                            "timemory_push_trace_hash");

        DLSYM_JUMP_FUNCTION(timemory_pop_trace_hash_jump, libhandle,
                            "timemory_pop_trace_hash");

        DLSYM_JUMP_FUNCTION(timemory_trace_init_jump, libhandle, "timemory_trace_init");

        DLSYM_JUMP_FUNCTION(timemory_trace_finalize_jump, libhandle,
                            "timemory_trace_finalize");

        DLSYM_JUMP_FUNCTION(timemory_trace_set_env_jump, libhandle,
                            "timemory_trace_set_env");

        DLSYM_JUMP_FUNCTION(timemory_trace_set_mpi_jump, libhandle,
                            "timemory_trace_set_mpi");

        dlclose(libhandle);
    }
};

//--------------------------------------------------------------------------------------//

std::unique_ptr<jump>&
get_jump()
{
#if defined(TIMEMORY_MACOS)
    static std::unique_ptr<jump> obj = std::make_unique<jump>(
        tim::get_env<std::string>("TIMEMORY_JUMP_LIBRARY", "libtimemory.dylib"));
#else
    static std::unique_ptr<jump> obj = std::make_unique<jump>(
        tim::get_env<std::string>("TIMEMORY_JUMP_LIBRARY", "libtimemory.so"));
#endif
    return obj;
}

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//
extern "C"
{
    void timemory_push_components(const char* name)
    {
        (*get_jump()->timemory_push_components_jump)(name);
    }

    void timemory_pop_components(void) { (*get_jump()->timemory_pop_components_jump)(); }

    void timemory_push_region(const char* name)
    {
        (*get_jump()->timemory_push_region_jump)(name);
    }

    void timemory_pop_region(const char* name)
    {
        (*get_jump()->timemory_pop_region_jump)(name);
    }

    void timemory_add_hash_id(uint64_t hash, const char* name)
    {
        (*get_jump()->timemory_add_hash_id_jump)(hash, name);
    }

    void timemory_push_trace(const char* name)
    {
        (*get_jump()->timemory_push_trace_jump)(name);
    }

    void timemory_pop_trace(const char* name)
    {
        (*get_jump()->timemory_pop_trace_jump)(name);
    }

    void timemory_push_trace_hash(uint64_t hash)
    {
        (*get_jump()->timemory_push_trace_hash_jump)(hash);
    }

    void timemory_pop_trace_hash(uint64_t hash)
    {
        (*get_jump()->timemory_pop_trace_hash_jump)(hash);
    }

    void timemory_trace_init(const char* a, bool b, const char* c)
    {
        (*get_jump()->timemory_trace_init_jump)(a, b, c);
    }

    void timemory_trace_finalize(void) { (*get_jump()->timemory_trace_finalize_jump)(); }

    void timemory_trace_set_env(const char* a, const char* b)
    {
        (*get_jump()->timemory_trace_set_env_jump)(a, b);
    }

    void timemory_trace_set_mpi(bool a, bool b)
    {
        (*get_jump()->timemory_trace_set_mpi_jump)(a, b);
    }
}
