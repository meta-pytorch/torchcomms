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

#pragma once

// define TIMEMORY_EXTERNAL_PAPI_DEFINITIONS if these enumerations/defs in bits/papi.hpp
// cause problems

#include "timemory/backends/defines.hpp"
#include "timemory/backends/hardware_counters.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/backends/types/papi.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#if defined(TIMEMORY_UNIX)
#    include <pthread.h>
#    include <unistd.h>
#endif

#if defined(TIMEMORY_USE_PAPI)
#    include <papiStdEventDefs.h>
//
#    include <papi.h>
#    if defined(TIMEMORY_UNIX)
#        include <pthread.h>
#    endif
#endif

// int EventSet = PAPI_NULL;
// unsigned int native = 0x0;
//
// if(PAPI_create_eventset(&EventSet) != PAPI_OK)
//     handle_error(1);
//
// Add Total Instructions Executed to our EventSet
// if(PAPI_add_event(EventSet, PAPI_TOT_INS) != PAPI_OK)
//     handle_error(1);
//
// Add native event PM_CYC to EventSet
// if(PAPI_event_name_to_code("PM_CYC",&native) != PAPI_OK)
//     handle_error(1);
//
// if(PAPI_add_event(EventSet, native) != PAPI_OK)
//     handle_error(1);

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace papi
{
using string_t         = std::string;
using ulong_t          = unsigned long int;
using hwcounter_info_t = std::vector<hardware_counters::info>;

struct component_info : PAPI_component_info_t
{
    using base_type = PAPI_component_info_t;

    TIMEMORY_DEFAULT_OBJECT(component_info)

    component_info(int _idx, base_type _info)
    : base_type{ _info }
    , index{ _idx }
    {}

    explicit component_info(base_type _info)
    : base_type{ _info }
    , index{ _info.CmpIdx }
    {}

    friend bool operator==(const component_info& lhs, const component_info& rhs)
    {
        return (lhs.index == rhs.index && strcmp(lhs.name, rhs.name) == 0);
    }

    friend bool operator!=(const component_info& lhs, const component_info& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator<(const component_info& lhs, const component_info& rhs)
    {
        return (lhs.index < rhs.index);
    }

    int index = 0;
};

struct event_info : PAPI_event_info_t
{
    using base_type = PAPI_event_info_t;

    TIMEMORY_DEFAULT_OBJECT(event_info)

    event_info(PAPI_event_info_t _info)
    : base_type{ _info }
    {}

    event_info(PAPI_event_info_t _info, string_t _unqual_sym, string_t _unqual_long_descr,
               bool _qualified = true, bool _mod_short_descr = false)
    : base_type{ _info }
    , qualified{ _qualified }
    , modified_short_descr{ _mod_short_descr }
    , unqualified_symbol{ std::move(_unqual_sym) }
    , unqualified_long_descr{ std::move(_unqual_long_descr) }
    {}

    friend bool operator==(const event_info& lhs, const event_info& rhs)
    {
        return (lhs.component_index == rhs.component_index)
                   ? (lhs.event_code == rhs.event_code)
                   : false;
    }

    friend bool operator!=(const event_info& lhs, const event_info& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator<(const event_info& lhs, const event_info& rhs)
    {
        return (lhs.component_index == rhs.component_index)
                   ? (lhs.event_code < rhs.event_code)
                   : (lhs.component_index < rhs.component_index);
    }

    bool     qualified              = false;
    bool     modified_short_descr   = false;
    string_t unqualified_symbol     = {};
    string_t unqualified_long_descr = {};
};

using event_info_t         = event_info;
using component_info_t     = component_info;
using component_info_map_t = std::map<component_info, std::vector<event_info>>;

//--------------------------------------------------------------------------------------//

inline ulong_t
get_thread_index()
{
    return static_cast<ulong_t>(threading::get_id());
}

//--------------------------------------------------------------------------------------//

inline uint64_t
get_papi_thread_num()
{
#if defined(TIMEMORY_USE_PAPI)
    return PAPI_thread_id();
#else
    return static_cast<ulong_t>(threading::get_id());
#endif
}

//--------------------------------------------------------------------------------------//

inline void
check_papi_thread()
{
#if defined(TIMEMORY_USE_PAPI)
    auto                               tid         = PAPI_thread_id();
    static constexpr unsigned long int invalid_tid = static_cast<unsigned long int>(-1);
    if(tid == invalid_tid)
    {
        TIMEMORY_EXCEPTION("PAPI_thread_id() returned unknown thread");
    }
#endif
}

//--------------------------------------------------------------------------------------//

inline auto
get_tid()
{
    return threading::get_tid();
}

//--------------------------------------------------------------------------------------//

inline auto
get_main_tid()
{
    return threading::get_main_tid();
}

//--------------------------------------------------------------------------------------//

inline auto
is_main_thread()
{
    return threading::is_main_thread();
}

//--------------------------------------------------------------------------------------//

inline bool&
working()
{
    static thread_local bool _instance = true;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline auto&
get_component_info_map()
{
    static auto _v = std::make_unique<component_info_map_t>();
    return _v;
}

//--------------------------------------------------------------------------------------//

inline bool
check(int retval, string_view_cref_t mesg, bool quiet = false)
{
    bool success = (retval == PAPI_OK);
    if(!success && !quiet)
    {
#if defined(TIMEMORY_USE_PAPI)
        auto*  error_str = PAPI_strerror(retval);
        auto&& _msg      = TIMEMORY_JOIN(' ', "[timemory][papi]", mesg, ":: PAPI_error",
                                         retval, ":", error_str);
        if(settings::papi_fail_on_error())
        {
            TIMEMORY_EXCEPTION(_msg);
        }
        else
        {
            if(working() && !settings::papi_quiet())
                TIMEMORY_PRINTF(stderr, "%s\n", _msg.c_str());
        }
#else
        TIMEMORY_PRINTF(stderr, "[timemory][papi] %s (error code = %i)\n", mesg.data(),
                        retval);
#endif
    }
    return (success && working());
}

//--------------------------------------------------------------------------------------//

inline void
set_debug(int level)
{
    // set the current debug level for PAPI
#if defined(TIMEMORY_USE_PAPI)
    int retval = PAPI_set_debug(level);
    check(retval, "Warning!! Failure to set debug level");
#else
    consume_parameters(level);
#endif
}

//--------------------------------------------------------------------------------------//

inline string_t
as_string(int* events, long long* values, int num_events, const string_t& indent)
{
    string_t _str{};
#if defined(TIMEMORY_USE_PAPI)
    for(int i = 0; i < num_events; ++i)
    {
        PAPI_event_info_t evt_info;
        if(PAPI_get_event_info(events[i], &evt_info) == PAPI_OK)
            _str = TIMEMORY_JOIN("", _str, indent, evt_info.long_descr, " : ", values[i],
                                 '\n');
    }
#else
    consume_parameters(events, values, num_events, indent);
#endif
    return _str;
}

//--------------------------------------------------------------------------------------//

inline void
register_thread()
{
    // inform PAPI of the existence of a new thread
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_register_thread();
        working()  = check(retval, "Warning!! Failure registering thread");
        check_papi_thread();
    }
#endif
}

//--------------------------------------------------------------------------------------//

inline void
unregister_thread()
{
    // inform PAPI that a previously registered thread is disappearing
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_unregister_thread();
        working()  = check(retval, "Warning!! Failure unregistering thread");
    }
#endif
}

//--------------------------------------------------------------------------------------//

int
get_event_code(string_view_cref_t event_code_str);

//--------------------------------------------------------------------------------------//

std::string
get_event_code_name(int event_code);

//--------------------------------------------------------------------------------------//

inline event_info_t
get_event_info(int evt_type)
{
    PAPI_event_info_t evt_info;
#if defined(TIMEMORY_USE_PAPI)
    PAPI_get_event_info(evt_type, &evt_info);
#else
    consume_parameters(evt_type);
#endif
    return evt_info;
}

//--------------------------------------------------------------------------------------//

event_info_t
get_event_info(string_view_cref_t evt_type);

//--------------------------------------------------------------------------------------//

inline bool
is_initialized()
{
#if defined(TIMEMORY_USE_PAPI)
    return (PAPI_is_initialized() != 0);
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline bool
attach(int event_set, Tp pid_or_tid)
{
    // inform PAPI that a previously registered thread is disappearing
#if defined(TIMEMORY_USE_PAPI)
    if(settings::verbose() >= 1)
        TIMEMORY_PRINTF(stderr,
                        "[timemory][papi] attaching event set %i to pid/tid %li\n",
                        event_set, (long) pid_or_tid);
    int retval = PAPI_attach(event_set, pid_or_tid);
    return (working() =
                check(retval, TIMEMORY_JOIN(" ", "Warning!! Failure attaching event set",
                                            event_set, "to pid/tid", pid_or_tid)));
#else
    consume_parameters(event_set, pid_or_tid);
    return false;
#endif
}

//--------------------------------------------------------------------------------------//
namespace details
{
inline bool
init_threading()
{
#if defined(TIMEMORY_USE_PAPI)
    static bool _threading = settings::papi_threading();
    if(!_threading)
        return false;

    static bool threading_initialized = false;
    if(is_main_thread() && !threading_initialized && working())
    {
        int retval = PAPI_thread_init(get_thread_index);
        working()  = check(retval, "Warning!! Failure initializing PAPI thread support");
        threading_initialized = true;
    }
    else if(!threading_initialized)
    {
        if(is_main_thread() && !working())
        {
            static std::atomic<int> _once(0);
            if(_once++ == 0)
                TIMEMORY_PRINTF(
                    stderr, "Warning!! Thread support is not enabled because it is not "
                            "currently working\n");
        }
    }
    return working();
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
init_multiplexing()
{
#if defined(TIMEMORY_USE_PAPI)
    static bool _multiplexing = settings::papi_multiplexing();
    if(!_multiplexing)
        return false;

    static bool multiplexing_initialized = false;
    if(!multiplexing_initialized)
    {
        int retval = PAPI_multiplex_init();
        working()  = check(retval, "Warning!! Failure initializing PAPI multiplexing");
        multiplexing_initialized = true;
    }
    else if(multiplexing_initialized)
    {
        if(!working())
        {
            static std::atomic<int32_t> _once(0);
            if(_once++ == 0)
                TIMEMORY_PRINTF(
                    stderr,
                    "Warning!! Multiplexing is not enabled because of previous PAPI "
                    "errors\n");
        }
    }
    return working();
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

namespace
{
#if defined(TIMEMORY_USE_PAPI)
auto papi_main_tid_assigned = (get_main_tid(), true);
#endif
}  // namespace

//--------------------------------------------------------------------------------------//

inline bool
init_library()
{
#if defined(TIMEMORY_USE_PAPI)
    if(!is_initialized() && papi_main_tid_assigned)
    {
        int retval = PAPI_library_init(PAPI_VER_CURRENT);
        if(retval != PAPI_VER_CURRENT && retval > 0)
            TIMEMORY_PRINTF(stderr, "PAPI library version mismatch!\n");
        working() = (retval == PAPI_VER_CURRENT);
    }
    return working();
#else
    return false;
#endif
}

size_t
generate_component_info(bool _qualifiers = true, bool _force = false);
}  // namespace details

//--------------------------------------------------------------------------------------//

inline bool
init()
{
    // initialize the PAPI library
#if defined(TIMEMORY_USE_PAPI)
    if(!is_initialized())
    {
        details::init_library();
        details::init_multiplexing();
        if(!working())
            TIMEMORY_PRINTF(
                stderr,
                "[timemory][papi] Warning!! PAPI library not fully initialized!\n");
        else
        {
            if(details::generate_component_info() == 0)
            {
                TIMEMORY_PRINTF(
                    stderr,
                    "[timemory][papi] Warning!! No PAPI component info was found\n");
            }
        }
    }
#else
    working() = false;
#endif
    return working();
}

//--------------------------------------------------------------------------------------//

inline bool
shutdown()
{
    // finish using PAPI and free all related resources
#if defined(TIMEMORY_USE_PAPI)
    if(is_initialized())
    {
        unregister_thread();
        if(get_tid() == get_main_tid())
        {
            PAPI_shutdown();
            working() = false;
        }
        return true;
    }
#endif
    return false;
}

//--------------------------------------------------------------------------------------//

inline void
print_hw_info()
{
    init();
#if defined(TIMEMORY_USE_PAPI)
    const PAPI_hw_info_t* hwinfo = PAPI_get_hardware_info();
    const PAPI_mh_info_t* mh     = &hwinfo->mem_hierarchy;
    printf("\n");
    printf("                    Vendor :   %s\n", hwinfo->vendor_string);
    printf("                     Model :   %s\n", hwinfo->model_string);
    printf("                   CPU MHz :   %f\n", static_cast<double>(hwinfo->mhz));
    printf("               CPU Max MHz :   %i\n", hwinfo->cpu_max_mhz);
    printf("               CPU Min MHz :   %i\n", hwinfo->cpu_min_mhz);
    printf("          Total NUMA nodes :   %i\n", hwinfo->nnodes);
    printf("             Number of CPU :   %i\n", hwinfo->ncpu);
    printf("                 Total CPU :   %i\n", hwinfo->totalcpus);
    printf("                   Sockets :   %i\n", hwinfo->sockets);
    printf("                     Cores :   %i\n", hwinfo->cores);
    printf("                   Threads :   %i\n", hwinfo->threads);
    printf("    Memory Hierarch Levels :   %i\n", mh->levels);
    printf(" Max level of TLB or Cache :   %d\n", mh->levels);
    for(int i = 0; i < mh->levels; i++)
    {
        for(int j = 0; j < PAPI_MH_MAX_LEVELS; j++)
        {
            const PAPI_mh_cache_info_t* c = &mh->level[i].cache[j];
            const PAPI_mh_tlb_info_t*   t = &mh->level[i].tlb[j];
            printf("        Level %2d, TLB   %2d :  %2d, %8d, %8d\n", i, j, t->type,
                   t->num_entries, t->associativity);
            printf("        Level %2d, Cache %2d :  %2d, %8d, %8d, %8d, %8d\n", i, j,
                   c->type, c->size, c->line_size, c->num_lines, c->associativity);
        }
    }
    printf("\n");
#endif
}

//--------------------------------------------------------------------------------------//

inline void
enable_multiplexing(int event_set, int component = 0)
{
#if defined(TIMEMORY_USE_PAPI)
    bool working_assign = true;
    if(working())
    {
        auto              retval = PAPI_assign_eventset_component(event_set, component);
        std::stringstream ss;
        ss << "Warning!! Failure to assign event set component. event set: " << event_set
           << ", component: " << component;
        working_assign = check(retval, ss.str().c_str());
    }
    if(working_assign)
    {
        auto              retval = PAPI_set_multiplex(event_set);
        std::stringstream ss;
        ss << "Warning!! Failure to enable multiplex on EventSet " << event_set;
        check(retval, ss.str().c_str());
    }
#else
    consume_parameters(event_set, component);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
create_event_set(int* event_set, bool enable_multiplex)
{
    // create a new empty PAPI event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_create_eventset(event_set);
        working()  = check(retval, "Warning!! Failure to create event set");
        if(working() && enable_multiplex)
            enable_multiplexing(*event_set);
        if(working())
            details::init_threading();
    }
#else
    consume_parameters(event_set, enable_multiplex);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
destroy_event_set(int event_set)
{
    // remove all PAPI events from an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_cleanup_eventset(event_set);
        working()  = check(retval, "Warning!! Failure to cleanup event set");
    }
#endif

    // deallocates memory associated with an empty PAPI event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_destroy_eventset(&event_set);
        working()  = check(retval, "Warning!! Failure to destroy event set");
    }
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
start(int event_set)
{
    // start counting hardware events in an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_start(event_set);
        working()  = check(retval, "Warning!! Failure to start event set");
    }
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
stop(int event_set, long long* values)
{
    // stop counting hardware events in an event set and return current events
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_stop(event_set, values);
        working()  = check(retval, "Warning!! Failure to stop event set");
    }
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
read(int event_set, long long* values)
{
    // read hardware events from an event set with no reset
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_read(event_set, values);
        working()  = check(retval, "Warning!! Failure to read event set");
    }
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
write(int event_set, long long* values)
{
    // write counter values into counters
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_write(event_set, values);
        working()  = check(retval, "Warning!! Failure to write event set");
    }
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
accum(int event_set, long long* values)
{
    // accumulate and reset hardware events from an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_accum(event_set, values);
        working()  = check(retval, "Warning!! Failure to accum event set");
    }
#else
    consume_parameters(event_set, values);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
reset(int event_set)
{
    // reset the hardware event counts in an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_reset(event_set);
        working()  = check(retval, "Warning!! Failure to reset event set");
    }
#else
    consume_parameters(event_set);
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
add_event(int event_set, int event)
{
    // add single PAPI preset or native hardware event to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        int  retval   = PAPI_add_event(event_set, event);
        bool _working = check(retval, TIMEMORY_JOIN(" ", "Warning!! Failure to add event",
                                                    event, "to event set", event_set));
        working()     = _working;
        return _working;
    }
#else
    consume_parameters(event_set, event);
#endif
    return false;
}

//--------------------------------------------------------------------------------------//

bool
add_event(int event_set, string_view_cref_t event_name);

//--------------------------------------------------------------------------------------//

inline bool
remove_event(int event_set, int event)
{
    // add single PAPI preset or native hardware event to an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int  retval = PAPI_remove_event(event_set, event);
        bool _working =
            check(retval, TIMEMORY_JOIN(" ", "Warning!! Failure to remove event", event,
                                        "from event set", event_set));
        working() = _working;
        return _working;
    }
#else
    consume_parameters(event_set, event);
#endif
    return false;
}

//--------------------------------------------------------------------------------------//

inline bool
remove_event(int event_set, string_view_cref_t event_name)
{
    // add single PAPI preset or native hardware event to an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int  retval = PAPI_remove_named_event(event_set, event_name.data());
        bool _working =
            check(retval, TIMEMORY_JOIN(" ", "Warning!! Failure to remove named event",
                                        event_name, "from event set", event_set));
        working() = _working;
        return _working;
    }
#else
    consume_parameters(event_set, event_name);
#endif
    return false;
}

//--------------------------------------------------------------------------------------//

inline bool
add_events(int event_set, int* events, int number)
{
    // add array of PAPI preset or native hardware events to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        int retval = PAPI_add_events(event_set, events, number);
        working()  = check(retval, "Warning!! Failure to add events to event set");
    }
#else
    consume_parameters(event_set, events, number);
#endif
    return working();
}

//--------------------------------------------------------------------------------------//

std::vector<bool>
add_events(int event_set, string_t* events, int number);

//--------------------------------------------------------------------------------------//

inline bool
remove_events(int event_set, int* events, int number)
{
    // add array of PAPI preset or native hardware events to an event set
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_remove_events(event_set, events, number);
        working()  = check(retval, "Warning!! Failure to remove events from event set");
    }
#else
    consume_parameters(event_set, events, number);
#endif
    return working();
}

//--------------------------------------------------------------------------------------//

inline std::vector<bool>
remove_events(int event_set, string_t* events, int number)
{
    std::vector<bool> _success(number, false);
    // add array of PAPI preset or native hardware events to an event set
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        for(int i = 0; i < number; ++i)
        {
            auto retval = PAPI_remove_named_event(event_set, events[i].c_str());
            _success[i] = check(
                retval, TIMEMORY_JOIN(" ", "Warning!! Failure to remove named event",
                                      events[i], "from event set", event_set));
        }
    }
#else
    consume_parameters(event_set, events, number);
#endif
    return _success;
}

//--------------------------------------------------------------------------------------//

inline bool
assign_event_set_component(int event_set, int cidx)
{
    // assign a component index to an existing but empty eventset
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_assign_eventset_component(event_set, cidx);
        working()  = check(retval, "Warning!! Failure to assign event set component");
    }
#else
    consume_parameters(event_set, cidx);
#endif
    return working();
}

//--------------------------------------------------------------------------------------//

inline bool
attach(int event_set, unsigned long tid)
{
    // attach specified event set to a specific process or thread id
#if defined(TIMEMORY_USE_PAPI)
    init();
    if(working())
    {
        int retval = PAPI_attach(event_set, tid);
        working()  = check(retval, "Warning!! Failure to attach event set");
    }
#else
    consume_parameters(event_set, tid);
#endif
    return working();
}

//--------------------------------------------------------------------------------------//

inline bool
detach(int event_set)
{
    // detach specified event set from a previously specified process or thread id
#if defined(TIMEMORY_USE_PAPI)
    if(working())
    {
        int retval = PAPI_detach(event_set);
        working()  = check(retval, "Warning!! Failure to detach event set");
    }
#else
    consume_parameters(event_set);
#endif
    return working();
}

//--------------------------------------------------------------------------------------//

inline bool
query_event(int event)
{
#if defined(TIMEMORY_USE_PAPI)
    return (PAPI_query_event(event) == PAPI_OK);
#else
    consume_parameters(event);
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
query_event(string_view_cref_t event)
{
#if defined(TIMEMORY_USE_PAPI)
    return (PAPI_query_named_event(event.data()) == PAPI_OK);
#else
    consume_parameters(event);
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Func>
inline bool
overflow(int evt_set, int evt_code, int threshold, int flags, Func&& handler)
{
#if defined(TIMEMORY_USE_PAPI)
    return (PAPI_overflow(evt_set, evt_code, threshold, flags,
                          std::forward<Func>(handler)) == PAPI_OK);
#else
    consume_parameters(evt_set, evt_code, threshold, flags, handler);
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Func>
inline bool
overflow(int evt_set, string_view_cref_t evt_name, int threshold, int flags,
         Func&& handler)
{
#if defined(TIMEMORY_USE_PAPI)
    auto _info = get_event_info(evt_name);
    return (PAPI_overflow(evt_set, _info.event_code, threshold, flags,
                          std::forward<Func>(handler)) == PAPI_OK);
#else
    consume_parameters(evt_set, evt_name, threshold, flags, handler);
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

hwcounter_info_t
available_events_info(const std::unordered_set<std::string>& excluded_components = {});

//--------------------------------------------------------------------------------------//

tim::hardware_counters::info
get_hwcounter_info(const std::string& event_code_str);

//--------------------------------------------------------------------------------------//

}  // namespace papi
}  // namespace tim

#if defined(TIMEMORY_BACKENDS_HEADER_MODE) && TIMEMORY_BACKENDS_HEADER_MODE > 0
#    include "timemory/backends/papi.cpp"
#endif
