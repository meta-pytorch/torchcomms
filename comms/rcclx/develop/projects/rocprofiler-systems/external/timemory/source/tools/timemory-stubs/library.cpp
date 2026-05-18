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

#include <cstdint>
#include <cstring>
#include <limits>

struct timemory_settings
{};

#define RETURN_MAX(TYPE) return std::numeric_limits<TYPE>::max()

namespace
{
const char* empty_str = "";
}

//======================================================================================//

extern "C"
{
    typedef void (*timemory_create_func_t)(const char*, uint64_t*, int, int*);
    typedef void (*timemory_delete_func_t)(uint64_t);

    timemory_create_func_t timemory_create_function = nullptr;
    timemory_delete_func_t timemory_delete_function = nullptr;
}

//--------------------------------------------------------------------------------------//
//
//      timemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C"
{
    void        c_timemory_init(int, char**, timemory_settings) {}
    void        c_timemory_finalize(void) {}
    int         c_timemory_enabled(void) { return 0; }
    void*       c_timemory_create_auto_timer(const char*) { return nullptr; }
    void        c_timemory_delete_auto_timer(void*) {}
    void*       c_timemory_create_auto_tuple(const char*, ...) { return nullptr; }
    void        c_timemory_delete_auto_tuple(void*) {}
    const char* c_timemory_blank_label(const char*) { return empty_str; }
    const char* c_timemory_basic_label(const char*, const char*) { return empty_str; }
    const char* c_timemory_label(const char*, const char*, int, const char*)
    {
        return empty_str;
    }

    uint64_t timemory_get_unique_id(void) { RETURN_MAX(uint64_t); }
    void     timemory_create_record(const char*, uint64_t*, int, int*) {}
    void     timemory_delete_record(uint64_t) {}
    void     timemory_init_library(int, char**) {}
    void     timemory_finalize_library(void) {}
    void     timemory_pause(void) {}
    void     timemory_resume(void) {}
    void     timemory_set_default(const char*) {}
    void     timemory_add_components(const char*) {}
    void     timemory_remove_components(const char*) {}
    void     timemory_push_components(const char*) {}
    void     timemory_push_components_enum(int, ...) {}
    void     timemory_pop_components(void) {}
    void     timemory_begin_record(const char*, uint64_t*) {}
    void     timemory_begin_record_types(const char*, uint64_t*, const char*) {}
    void     timemory_begin_record_enum(const char*, uint64_t*, ...) {}
    uint64_t timemory_get_begin_record(const char*) { RETURN_MAX(uint64_t); }
    uint64_t timemory_get_begin_record_types(const char*, const char*)
    {
        RETURN_MAX(uint64_t);
    }
    uint64_t timemory_get_begin_record_enum(const char*, ...) { RETURN_MAX(uint64_t); }
    void     timemory_end_record(uint64_t) {}
    void     timemory_push_region(const char*) {}
    void     timemory_pop_region(const char*) {}

    bool timemory_is_throttled(const char*) { return true; }
    void timemory_add_hash_id(uint64_t, const char*) {}
    void timemory_add_hash_ids(uint64_t, uint64_t*, const char**) {}

    // tracing API
    void timemory_push_trace(const char*) {}
    void timemory_pop_trace(const char*) {}
    void timemory_push_trace_hash(uint64_t) {}
    void timemory_pop_trace_hash(uint64_t) {}
    void timemory_trace_init(const char*, bool, const char*) {}
    void timemory_trace_finalize(void) {}
    void timemory_trace_set_env(const char*, const char*) {}
    void timemory_trace_set_mpi(bool, bool) {}

    void     timemory_create_record_(const char*, uint64_t*, int, int*) {}
    void     timemory_delete_record_(uint64_t) {}
    void     timemory_init_library_(int, char**) {}
    void     timemory_finalize_library_(void) {}
    void     timemory_set_default_(const char*) {}
    void     timemory_push_components_(const char*) {}
    void     timemory_pop_components_(void) {}
    void     timemory_begin_record_(const char*, uint64_t*) {}
    void     timemory_begin_record_types_(const char*, uint64_t*, const char*) {}
    uint64_t timemory_get_begin_record_(const char*) { RETURN_MAX(uint64_t); }
    uint64_t timemory_get_begin_record_types_(const char*, const char*)
    {
        RETURN_MAX(uint64_t);
    }
    void timemory_end_record_(uint64_t) {}
    void timemory_push_region_(const char*) {}
    void timemory_pop_region_(const char*) {}

}  // extern "C"
