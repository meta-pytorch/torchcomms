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

#if defined(__cplusplus)
#    include <cstdint>
#    include <cstdio>
#    include <cstdlib>
#else
#    include <stdbool.h>
#    include <stddef.h>
#    include <stdint.h>
#    include <stdio.h>
#    include <stdlib.h>
#    include <string.h>
#endif

#include "timemory/compat/macros.h"
#include "timemory/enum.h"

//======================================================================================//

#if defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_GOTCHA)
#    define TIMEMORY_MPI_GOTCHA
#endif

// this is used bc it is easier to remove from generated XML when documentation is
// rendered
#if !defined(TIMEMORY_VISIBLE)
#    define TIMEMORY_VISIBLE TIMEMORY_VISIBILITY("default")
#endif

//======================================================================================//
//
//      C struct for settings
//
//======================================================================================//

typedef struct
{
    int enabled;
    int auto_output;
    int file_output;
    int text_output;
    int json_output;
    int cout_output;
    int precision;
    int width;
    int scientific;
    // skipping remainder
} timemory_settings;

//======================================================================================//

#if defined(__cplusplus)
extern "C"
{
#endif  // if defined(__cplusplus)

    /// \fn uint64_t timemory_get_unique_id(void)
    /// Returns a unique integer for a thread.
    extern uint64_t timemory_get_unique_id(void) TIMEMORY_VISIBLE;

    /// \fn void timemory_create_record(const char* name, uint64_t* id, int n, int* ct)
    /// \param [in] name label for the record
    /// \param [in,out] id assigned a unique identifier for the record
    /// \param [in] n number of components
    /// \param [in] ct array of enumeration identifiers of size n
    ///
    /// Function called by \ref timemory_begin_record, \ref
    /// timemory_begin_record_enum, \ref timemory_begin_record_types, \ref
    /// timemory_get_begin_record, \ref timemory_get_begin_record_enum, \ref
    /// timemory_get_begin_record_types, \ref timemory_push_region for creating and
    /// starting the current collection of components.
    extern void timemory_create_record(const char* name, uint64_t* id, int n,
                                       int* ct) TIMEMORY_VISIBLE;

    /// \fn void timemory_delete_record(uint64_t nid)
    /// Deletes the record created by \ref timemory_create_record.
    extern void timemory_delete_record(uint64_t nid) TIMEMORY_VISIBLE;

    /// \fn bool timemory_library_is_initialized(void)
    /// Returns whether the library is initialized or not.
    extern bool timemory_library_is_initialized(void) TIMEMORY_VISIBLE;

    /// \fn void timemory_named_init_library(const char* name)
    /// Initializes timemory with the exe name
    extern void timemory_named_init_library(char* name) TIMEMORY_VISIBLE;

    /// \fn void timemory_init_library(int argc, char** argv)
    /// Initializes timemory. Not strictly necessary but highly recommended.
    extern void timemory_init_library(int argc, char** argv) TIMEMORY_VISIBLE;

    /// \fn void timemory_finalize_library(void)
    /// Finalizes timemory. Output will be generated. Any attempt to store
    /// data within timemory storage is undefined after this point and will likely
    /// cause errors.
    extern void timemory_finalize_library(void) TIMEMORY_VISIBLE;

    /// \fn void timemory_pause(void)
    /// Turn off timemory collection
    extern void timemory_pause(void) TIMEMORY_VISIBLE;

    /// \fn void timemory_resume(void)
    /// Turn on timemory collection
    extern void timemory_resume(void) TIMEMORY_VISIBLE;

    /// \fn void timemory_set_default(const char* components)
    /// Pass in a default set of components to use. Will be overridden by
    /// TIMEMORY_COMPONENTS environment variable.
    ///
    /// \code{.cpp}
    /// timemory_set_default("wall_clock, cpu_clock, cpu_util");
    /// \endcode
    extern void timemory_set_default(const char* components) TIMEMORY_VISIBLE;

    /// \fn void timemory_set_environ(const char* evar, const char* eval, int ovr, int up)
    /// \param [in] evar Environment variable name
    /// \param [in] eval Environment variable value
    /// \param [in] ovr Overwrite existing environment variable when > 0
    /// \param [in] up Update settings when > 0
    ///
    /// Set an environment variable and (potentially) update settings with new value.
    ///
    /// \code{.cpp}
    /// // overwrites the TIMEMORY_GLOBAL_COMPONENTS environment variable and updates
    /// // settings
    /// timemory_set_environ("TIIMEMORY_GLOBAL_COMPONENTS,
    ///                      "wall_clock, cpu_clock, cpu_util", 1, 1);
    /// \endcode
    extern void timemory_set_environ(const char* evar, const char* eval, int ovr,
                                     int up) TIMEMORY_VISIBLE;

    /// \fn void timemory_add_components(const char* components)
    /// Add some components to the current set of components being collected
    /// Any components which are currently being collected are ignored.
    ///
    /// \code{.cpp}
    /// timemory_add_components("peak_rss, priority_context_switch");
    /// \endcode

    extern void timemory_add_components(const char* components) TIMEMORY_VISIBLE;

    /// \fn void timemory_remove_components(const char* components)
    /// Remove some components to the current set of components being collected.
    /// Any components which are not currently being collected are ignored.
    ///
    /// \code{.cpp}
    /// timemory_add_components("priority_context_switch, read_bytes");
    /// \endcode
    extern void timemory_remove_components(const char* components) TIMEMORY_VISIBLE;

    /// \fn void timemory_push_components(const char* components)
    /// Replace the current set of components with a new set of components.
    ///
    /// \code{.cpp}
    /// timemory_push_components("priority_context_switch, read_bytes");
    /// \endcode
    extern void timemory_push_components(const char* components) TIMEMORY_VISIBLE;

    /// \fn void timemory_push_components_enum(int args, ...)
    /// Replace the current set of components with a new set of components with
    /// the set of enumerations provided to the function. First argument should be the
    /// number of new components.
    ///
    /// \code{.cpp}
    /// timemory_push_components(2, WALL_CLOCK, CPU_CLOCK);
    /// \endcode
    extern void timemory_push_components_enum(int args, ...) TIMEMORY_VISIBLE;

    /// \fn void timemory_pop_components(void)
    /// Inverse of the last \ref timemory_push_components or \ref
    /// timemory_push_components_enum call. Popping all components will restore to set the
    /// configured as the default.
    extern void timemory_pop_components(void) TIMEMORY_VISIBLE;

    /// \fn void timemory_begin_record(const char* name, uint64_t* id)
    /// \param [in] name Label for the record
    /// \param [in,out] id identifier passed back to \ref timemory_end_record
    ///
    /// \code{.cpp}
    /// uint64_t idx = 0;
    /// timemory_begin_record("foo", &idx);
    /// // ...
    /// timemory_end_record(idx);
    /// \endcode
    extern void timemory_begin_record(const char* name, uint64_t* id) TIMEMORY_VISIBLE;

    /// \fn void timemory_begin_record_enum(const char* name, uint64_t*, ...)
    /// Similar to \ref timemory_begin_record but accepts a specific enumerated
    /// set of components, which is terminated by TIMEMORY_COMPONENTS_END.
    ///
    /// \code{.cpp}
    /// uint64_t idx = 0;
    /// timemory_begin_record("foo", &idx, WALL_CLOCK, CPU_UTIL, TIMEMORY_COMPONENTS_END);
    /// // ...
    /// timemory_end_record(idx);
    /// \endcode
    extern void timemory_begin_record_enum(const char* name, uint64_t*,
                                           ...) TIMEMORY_VISIBLE;

    /// \fn void timemory_begin_record_types(const char* name, uint64_t*, const char*)
    /// Similar to \ref timemory_begin_record but accepts a specific set of
    /// components as a string.
    ///
    /// \code{.cpp}
    /// uint64_t idx = 0;
    /// timemory_begin_record_types("foo", &idx, "wall_clock, cpu_util");
    /// // ...
    /// timemory_end_record(idx);
    /// \endcode
    extern void timemory_begin_record_types(const char* name, uint64_t*,
                                            const char*) TIMEMORY_VISIBLE;

    /// \fn uint64_t timemory_get_begin_record(const char* name)
    /// Variant to \ref timemory_begin_record which returns a unique integer
    extern uint64_t timemory_get_begin_record(const char* name) TIMEMORY_VISIBLE;

    /// \fn uint64_t timemory_get_begin_record_enum(const char* name, ...)
    /// Variant to \ref timemory_begin_record_enum which returns a unique integer
    extern uint64_t timemory_get_begin_record_enum(const char* name,
                                                   ...) TIMEMORY_VISIBLE;

    /// \fn uint64_t timemory_get_begin_record_types(const char* name, const char* types)
    /// Variant to \ref timemory_begin_record_types which returns a unique integer
    extern uint64_t timemory_get_begin_record_types(const char* name,
                                                    const char* ctypes) TIMEMORY_VISIBLE;

    /// \fn void timemory_end_record(uint64_t id)
    /// \param [in] id Identifier for the recording entry
    ///
    extern void timemory_end_record(uint64_t id) TIMEMORY_VISIBLE;

    /// \fn void timemory_push_region(const char* name)
    /// \param [in] name label for region
    ///
    /// Starts collection of components with label.
    ///
    /// \code{.cpp}
    /// void foo()
    /// {
    ///     timemory_push_region("foo");
    ///     // ...
    ///     timemory_pop_region("foo");
    /// }
    /// \endcode
    extern void timemory_push_region(const char* name) TIMEMORY_VISIBLE;

    /// \fn void timemory_pop_region(const char* name)
    /// \param [in] name label for region
    ///
    /// Stops collection of components with label.
    ///
    /// \code{.cpp}
    /// void foo()
    /// {
    ///     timemory_push_region("foo");
    ///     // ...
    ///     timemory_pop_region("foo");
    /// }
    /// \endcode
    extern void timemory_pop_region(const char* name) TIMEMORY_VISIBLE;

    extern void        c_timemory_init(int argc, char** argv,
                                       timemory_settings) TIMEMORY_VISIBLE;
    extern void        c_timemory_finalize(void) TIMEMORY_VISIBLE;
    extern int         c_timemory_enabled(void) TIMEMORY_VISIBLE;
    extern void*       c_timemory_create_auto_timer(const char*) TIMEMORY_VISIBLE;
    extern void        c_timemory_delete_auto_timer(void*) TIMEMORY_VISIBLE;
    extern void*       c_timemory_create_auto_tuple(const char*, ...) TIMEMORY_VISIBLE;
    extern void        c_timemory_delete_auto_tuple(void*) TIMEMORY_VISIBLE;
    extern const char* c_timemory_blank_label(const char*) TIMEMORY_VISIBLE;
    extern const char* c_timemory_basic_label(const char*, const char*) TIMEMORY_VISIBLE;
    extern const char* c_timemory_label(const char*, const char*, int,
                                        const char*) TIMEMORY_VISIBLE;
    extern int         cxx_timemory_enabled(void) TIMEMORY_VISIBLE;
    extern void        cxx_timemory_init(int, char**, timemory_settings) TIMEMORY_VISIBLE;
    extern void*       cxx_timemory_create_auto_timer(const char*) TIMEMORY_VISIBLE;
    extern void*       cxx_timemory_create_auto_tuple(const char*, int,
                                                      const int*) TIMEMORY_VISIBLE;
    extern void*       cxx_timemory_delete_auto_timer(void*) TIMEMORY_VISIBLE;
    extern void*       cxx_timemory_delete_auto_tuple(void*) TIMEMORY_VISIBLE;
    extern const char* cxx_timemory_label(int, int, const char*, const char*,
                                          const char*) TIMEMORY_VISIBLE;

    extern bool timemory_trace_is_initialized(void) TIMEMORY_VISIBLE;
    extern void timemory_reset_throttle(const char* name) TIMEMORY_VISIBLE;
    extern bool timemory_is_throttled(const char* name) TIMEMORY_VISIBLE;
    extern void timemory_add_hash_id(uint64_t id, const char* name) TIMEMORY_VISIBLE;
    extern void timemory_add_hash_ids(uint64_t nentries, uint64_t* ids,
                                      const char** names) TIMEMORY_VISIBLE;
    extern void timemory_push_trace_hash(uint64_t id) TIMEMORY_VISIBLE;
    extern void timemory_pop_trace_hash(uint64_t id) TIMEMORY_VISIBLE;
    extern void timemory_push_trace(const char* name) TIMEMORY_VISIBLE;
    extern void timemory_pop_trace(const char* name) TIMEMORY_VISIBLE;
    extern void timemory_trace_init(const char*, bool, const char*) TIMEMORY_VISIBLE;
    extern void timemory_trace_finalize(void) TIMEMORY_VISIBLE;
    extern void timemory_trace_set_env(const char*, const char*) TIMEMORY_VISIBLE;

#if defined(TIMEMORY_MPI_GOTCHA)
    /// \fn void timemory_trace_set_mpi(bool use, bool attached)
    /// \param[in] use Use MPI gotcha
    /// \param[in] attached Tracing application has attached to a running program
    ///
    /// This function is only declared and defined if timemory was built
    /// with support for MPI and GOTCHA.
    extern void timemory_trace_set_mpi(bool use, bool attached) TIMEMORY_VISIBLE;
#endif

    /// \typedef void (*timemory_create_func_t)(const char*, uint64_t*, int, int*)
    /// function pointer type for \ref timemory_create_function
    typedef void (*timemory_create_func_t)(const char*, uint64_t*, int, int*);

    /// \typedef void (*timemory_delete_func_t)(uint64_t)
    /// function pointer type for \ref timemory_delete_function
    typedef void (*timemory_delete_func_t)(uint64_t);

    /// \var timemory_create_func_t timemory_create_function
    /// The function pointer to set to customize which components are used by
    /// library interface.
    /// \code{.cpp}
    /// using namespace tim::component;
    /// using test_list_t =
    ///     tim::component_list<wall_clock, cpu_util, cpu_clock, peak_rss>;
    ///
    /// static std::map<uint64_t, std::shared_ptr<test_list_t>> test_map;
    ///
    /// void
    /// custom_create_record(const char* name, uint64_t* id, int n, int* ct)
    /// {
    ///     uint64_t idx = timemory_get_unique_id();
    ///     auto     tmp = std::make_shared<test_list_t>(name);
    ///     tim::initialize(*tmp, n, ct);
    ///     tmp->initialize<cpu_util, cpu_clock>();
    ///     test_map[idx] = tmp;
    ///     test_map[idx]->start();
    ///     *id = idx;
    /// }
    ///
    /// void
    /// main()
    /// {
    ///     // ... using default create/delete functions ...
    ///
    ///     timemory_create_function = &custom_create_record;
    ///     timemory_delete_function = ...;
    ///
    ///     // ... using custom create/delete functions ...
    ///
    ///     timemory_create_function = nullptr;
    ///     timemory_delete_function = ...;
    ///
    ///     // ... using default create/delete functions ...
    ///
    /// }
    /// \endcode
    extern TIMEMORY_DLL timemory_create_func_t timemory_create_function TIMEMORY_VISIBLE;

    /// \var timemory_delete_func_t timemory_delete_function
    /// The function pointer to set which deletes an entry created by \ref
    /// timemory_create_function.
    /// \code{.cpp}
    ///
    /// static std::map<uint64_t, std::shared_ptr<test_list_t>> test_map;
    ///
    /// void
    /// custom_delete_record(uint64_t id)
    /// {
    ///     auto itr = test_map.find(id);
    ///     if(itr != test_map.end())
    ///     {
    ///         itr->second->stop();
    ///         test_map.erase(itr);
    ///     }
    /// }
    ///
    /// void
    /// main()
    /// {
    ///     // ... using default create/delete functions ...
    ///
    ///     timemory_create_function = &custom_create_record;
    ///     timemory_delete_function = &custom_delete_record;
    ///
    ///     // ... using custom create/delete functions ...
    ///
    ///     timemory_create_function = nullptr;
    ///     timemory_delete_function = nullptr;
    ///
    ///     // ... using default create/delete functions ...
    ///
    /// }
    /// \endcode
    extern TIMEMORY_DLL timemory_delete_func_t timemory_delete_function TIMEMORY_VISIBLE;

#if defined(__cplusplus)
}
#endif  // if defined(__cplusplus)
