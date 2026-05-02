
#include "timemory/runtime/configure.hpp"
#include "timemory/timemory.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#if !_MSC_VER
#    include <execinfo.h>
#endif

using namespace tim::component;

static std::string spacer =
    "#---------------------------------------------------------------------------#";

using KokkosUserBundle = tim::component::user_kokkosp_bundle;

using external_profilers_t =
    tim::component_tuple<vtune_profiler, cuda_profiler, craypat_record, allinea_map>;
using profile_entry_t =
    tim::component_tuple_t<external_profilers_t, KokkosUserBundle, vtune_frame,
                           vtune_event, nvtx_marker, craypat_region>;

//
//  re-nice the start priority starts after profiler controllers
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, KokkosUserBundle, priority_constant<64>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, vtune_frame, priority_constant<64>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, vtune_event, priority_constant<64>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, nvtx_marker, priority_constant<64>)

//
//  re-nice the stop priority so stops before profilers
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, KokkosUserBundle, priority_constant<-64>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, vtune_frame, priority_constant<-64>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, vtune_event, priority_constant<-64>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, nvtx_marker, priority_constant<-64>)

//--------------------------------------------------------------------------------------//

// various data structures used
using section_entry_t = std::tuple<std::string, profile_entry_t>;
using profile_stack_t = std::vector<profile_entry_t>;
using profile_map_t   = std::unordered_map<uint64_t, profile_entry_t>;
using section_map_t   = std::unordered_map<uint64_t, section_entry_t>;

//--------------------------------------------------------------------------------------//

static std::string kernel_regex_expr = "^[A-Za-z]";
static auto        regex_constants =
    std::regex_constants::ECMAScript | std::regex_constants::optimize;
static auto kernel_regex = std::regex(kernel_regex_expr, regex_constants);

//--------------------------------------------------------------------------------------//

static uint64_t
get_unique_id()
{
    static thread_local uint64_t _instance = 0;
    return _instance++;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
Tp&
get_tl_static()
{
    static thread_local Tp _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

static external_profilers_t&
get_external_profilers()
{
    static external_profilers_t _instance("external_profilers");
    return _instance;
}

//--------------------------------------------------------------------------------------//

static profile_map_t&
get_profile_map()
{
    return get_tl_static<profile_map_t>();
}

//--------------------------------------------------------------------------------------//

static profile_stack_t&
get_profile_stack()
{
    return get_tl_static<profile_stack_t>();
}

//--------------------------------------------------------------------------------------//

static void
create_profiler(const std::string& pname, uint64_t kernid)
{
    get_profile_map().insert(std::make_pair(kernid, profile_entry_t(pname, true)));
}

//--------------------------------------------------------------------------------------//

static void
destroy_profiler(uint64_t kernid)
{
    if(get_profile_map().find(kernid) != get_profile_map().end())
        get_profile_map().erase(kernid);
}

//--------------------------------------------------------------------------------------//

static void
start_profiler(uint64_t kernid)
{
    if(get_profile_map().find(kernid) != get_profile_map().end())
        get_profile_map().at(kernid).start();
}

//--------------------------------------------------------------------------------------//

static void
stop_profiler(uint64_t kernid)
{
    if(get_profile_map().find(kernid) != get_profile_map().end())
        get_profile_map().at(kernid).stop();
}

//--------------------------------------------------------------------------------------//

static bool
check_regex(const std::string& _entry)
{
    try
    {
        return std::regex_search(_entry, kernel_regex);
    } catch(std::regex_error& e)
    {
        std::cerr << e.what() << std::endl;
    }
    return true;
}

//--------------------------------------------------------------------------------------//

bool
configure_environment()
{
    tim::set_env("TIMEMORY_TIME_OUTPUT", "ON", 0);
    tim::set_env("TIMEMORY_COUT_OUTPUT", "OFF", 0);
    tim::set_env("TIMEMORY_ADD_SECONDARY", "OFF", 0);
    return true;
}

static auto env_configured = (configure_environment(), true);

//======================================================================================//
//
//      Kokkos symbols
//
//======================================================================================//

extern "C" void
kokkosp_print_help(char* argv0)
{
    std::vector<std::string> _args = { argv0, "--help" };
    tim::timemory_argparse(_args);
}

extern "C" void
kokkosp_parse_args(int argc, char** argv)
{
    tim::timemory_init(argc, argv);

    std::vector<std::string> _args{};
    _args.reserve(argc);
    for(int i = 0; i < argc; ++i)
        _args.emplace_back(argv[i]);

    tim::timemory_argparse(_args);
}

extern "C" void
kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                     const uint32_t devInfoCount, void* deviceInfo)
{
    tim::consume_parameters(devInfoCount, deviceInfo);
    printf("%s\n", spacer.c_str());
    printf("# KokkosP: timemory Connector (sequence is %d, version: %llu)\n", loadSeq,
           (unsigned long long) interfaceVer);
    printf("%s\n\n", spacer.c_str());

    // this will ensure all external profilers are put into a stop state
    {
        get_external_profilers().start();
        get_external_profilers().stop();
    }

    // default metrics
    tim::set_env("TIMEMORY_CUPTI_METRICS", "gld_efficiency", 0);
    tim::set_env("TIMEMORY_PAPI_EVENTS", "PAPI_LST_INS", 0);

    // timemory_init is expecting some args so generate some
    std::array<char*, 1> cstr = { { strdup("kp_timemory_filter") } };
    tim::timemory_init(1, cstr.data());
    free(cstr.at(0));
    assert(env_configured);

    std::string default_components = "wall_clock";
    if(tim::trait::is_available<cupti_counters>::value)
        default_components = TIMEMORY_JOIN(",", default_components, "cupti_counters");
    else if(tim::trait::is_available<papi_vector>::value)
        default_components = TIMEMORY_JOIN(",", default_components, "papi_vector");

    // search unique and fallback environment variables
    tim::operation::init<KokkosUserBundle>(
        tim::operation::mode_constant<tim::operation::init_mode::global>{});

    // add defaults
    if(KokkosUserBundle::bundle_size() == 0)
        tim::configure<KokkosUserBundle>(
            tim::enumerate_components(tim::delimit(default_components)));

    std::cout << "USING: " << tim::demangle<profile_entry_t>() << "\n" << std::endl;
    kernel_regex_expr =
        tim::get_env<std::string>("KOKKOS_PROFILE_REGEX", kernel_regex_expr);
    std::cout << "KOKKOS_PROFILE_REGEX : \"" << kernel_regex_expr << "\"\n" << std::endl;
    kernel_regex = std::regex(kernel_regex_expr, regex_constants);
}

extern "C" void
kokkosp_finalize_library()
{
    printf("\n%s\n", spacer.c_str());
    printf("KokkosP: Finalization of timemory Connector. Complete.\n");
    printf("%s\n\n", spacer.c_str());

    for(auto& itr : get_profile_map())
        itr.second.stop();
    get_profile_map().clear();

    tim::timemory_finalize();
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
{
    if(!check_regex(name))
    {
        *kernid = std::numeric_limits<uint64_t>::max();
        return;
    }

    auto pname =
        (devid > std::numeric_limits<uint16_t>::max())  // junk device number
            ? TIMEMORY_JOIN('/', "kokkos", name)
            : TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
    *kernid = get_unique_id();
    create_profiler(pname, *kernid);
    start_profiler(*kernid);
}

extern "C" void
kokkosp_end_parallel_for(uint64_t kernid)
{
    if(kernid == std::numeric_limits<uint64_t>::max())
        return;

    stop_profiler(kernid);
    destroy_profiler(kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
{
    if(!check_regex(name))
    {
        *kernid = std::numeric_limits<uint64_t>::max();
        return;
    }

    auto pname =
        (devid > std::numeric_limits<uint16_t>::max())  // junk device number
            ? TIMEMORY_JOIN('/', "kokkos", name)
            : TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
    *kernid = get_unique_id();
    create_profiler(pname, *kernid);
    start_profiler(*kernid);
}

extern "C" void
kokkosp_end_parallel_reduce(uint64_t kernid)
{
    if(kernid == std::numeric_limits<uint64_t>::max())
        return;

    stop_profiler(kernid);
    destroy_profiler(kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
{
    if(!check_regex(name))
    {
        *kernid = std::numeric_limits<uint64_t>::max();
        return;
    }

    auto pname =
        (devid > std::numeric_limits<uint16_t>::max())  // junk device number
            ? TIMEMORY_JOIN('/', "kokkos", name)
            : TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
    *kernid = get_unique_id();
    create_profiler(pname, *kernid);
    start_profiler(*kernid);
}

extern "C" void
kokkosp_end_parallel_scan(uint64_t kernid)
{
    if(kernid == std::numeric_limits<uint64_t>::max())
        return;

    stop_profiler(kernid);
    destroy_profiler(kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_fence(const char* name, uint32_t devid, uint64_t* kernid)
{
    if(!check_regex(name))
    {
        *kernid = std::numeric_limits<uint64_t>::max();
        return;
    }

    auto pname =
        (devid > std::numeric_limits<uint16_t>::max())  // junk device number
            ? TIMEMORY_JOIN('/', "kokkos", name)
            : TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
    *kernid = get_unique_id();
    create_profiler(pname, *kernid);
    start_profiler(*kernid);
}

extern "C" void
kokkosp_end_fence(uint64_t kernid)
{
    if(kernid == std::numeric_limits<uint64_t>::max())
        return;

    stop_profiler(kernid);
    destroy_profiler(kernid);
}

//--------------------------------------------------------------------------------------//

static thread_local int64_t region_skip = 0;

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_push_profile_region(const char* name)
{
    if(!check_regex(name))
    {
        ++region_skip;
        return;
    }

    get_profile_stack().push_back(profile_entry_t(name, true));
    get_profile_stack().back().start();
}

extern "C" void
kokkosp_pop_profile_region()
{
    if(region_skip > 0)
    {
        --region_skip;
        return;
    }

    if(get_profile_stack().empty())
        return;
    get_profile_stack().back().stop();
    get_profile_stack().pop_back();
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_create_profile_section(const char* name, uint32_t* secid)
{
    if(!check_regex(name))
    {
        *secid = std::numeric_limits<uint32_t>::max();
        return;
    }

    *secid     = get_unique_id();
    auto pname = TIMEMORY_JOIN('/', "kokkos", name);
    create_profiler(pname, *secid);
}

extern "C" void
kokkosp_destroy_profile_section(uint32_t secid)
{
    if(secid == std::numeric_limits<uint32_t>::max())
        return;

    destroy_profiler(secid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_start_profile_section(uint32_t secid)
{
    if(secid == std::numeric_limits<uint32_t>::max())
        return;

    start_profiler(secid);
}

extern "C" void
kokkosp_stop_profile_section(uint32_t secid)
{
    if(secid == std::numeric_limits<uint32_t>::max())
        return;

    start_profiler(secid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_profile_event(const char* name)
{
    if(!check_regex(name))
        return;

    profile_entry_t{}.mark(name);
}

//--------------------------------------------------------------------------------------//
