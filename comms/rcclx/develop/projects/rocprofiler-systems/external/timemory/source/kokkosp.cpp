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

// used by Kokkos decls
#if !defined(TIMEMORY_LIBRARY_SOURCE)
#    define TIMEMORY_LIBRARY_SOURCE 1
#endif

#include "timemory/api/kokkosp.hpp"

#include "timemory/timemory.hpp"

#include <string>

namespace kokkosp = tim::kokkosp;

//--------------------------------------------------------------------------------------//

namespace tim
{
template <>
inline auto
invoke_preinit<kokkosp::memory_tracker>(long)
{
    kokkosp::memory_tracker::label()       = "kokkos_memory";
    kokkosp::memory_tracker::description() = "Kokkos Memory tracker";
}
}  // namespace tim

//--------------------------------------------------------------------------------------//

namespace
{
std::string kokkos_banner =
    "#---------------------------------------------------------------------------#";

//--------------------------------------------------------------------------------------//

bool
configure_environment()
{
    tim::set_env("KOKKOS_PROFILE_LIBRARY", "libtimemory.so", 0);
    return true;
}

auto env_configured       = (configure_environment(), true);
bool enable_kernel_logger = false;

inline void
add_kernel_logger()
{
    static bool _first = true;
    if(!_first)
        return;
    _first         = false;
    using strvec_t = std::vector<std::string>;
    using strset_t = std::set<std::string>;

    tim::settings::instance()->insert<bool, bool&>(
        TIMEMORY_SETTINGS_KEY("KOKKOS_KERNEL_LOGGER"),
        std::string{ "kokkos_kernel_logger" }, std::string{ "Enables kernel logging" },
        enable_kernel_logger, strset_t{ "native", "component", "kokkos" },
        strvec_t({ "--" TIMEMORY_PROJECT_NAME "-kokkos-kernel-logger" }));
}

inline void
setup_kernel_logger()
{
    if(tim::settings::debug() || tim::settings::verbose() > 3 || enable_kernel_logger)
    {
        kokkosp::logger_t::get_initializer() = [](kokkosp::logger_t& _obj) {
            _obj.initialize<kokkosp::kernel_logger>();
        };
    }
}

}  // namespace

//--------------------------------------------------------------------------------------//

extern "C"
{
    void kokkosp_print_help(char* argv)
    {
        add_kernel_logger();
        std::vector<std::string> _args = { argv, "--help" };
        tim::timemory_argparse(_args);
    }

    void kokkosp_parse_args(int argc, char** argv)
    {
        add_kernel_logger();

        tim::timemory_init(argc, argv);

        std::vector<std::string> _args{};
        _args.reserve(argc);
        for(int i = 0; i < argc; ++i)
            _args.emplace_back(argv[i]);

        tim::timemory_argparse(_args);

        setup_kernel_logger();
    }

    void kokkosp_declare_metadata(const char* key, const char* value)
    {
        tim::manager::add_metadata(key, value);
    }

    void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                              const uint32_t devInfoCount, void* deviceInfo)
    {
        add_kernel_logger();

        tim::consume_parameters(devInfoCount, deviceInfo);

        tim::set_env("TIMEMORY_TIME_OUTPUT", "ON", 0);
        tim::set_env("TIMEMORY_COUT_OUTPUT", "OFF", 0);
        tim::set_env("TIMEMORY_ADD_SECONDARY", "OFF", 0);

        printf("%s\n", kokkos_banner.c_str());
        printf("# KokkosP: timemory Connector (sequence is %d, version: %llu)\n", loadSeq,
               (unsigned long long) interfaceVer);
        printf("%s\n\n", kokkos_banner.c_str());

        if(!tim::settings::instance()->get_initialized())
        {
            // timemory_init is expecting some args so generate some
            std::array<char*, 1> cstr = { { strdup("kokkosp") } };
            tim::timemory_init(1, cstr.data());
            free(cstr[0]);
        }
        assert(env_configured);

        // search unique and fallback environment variables
        namespace operation = tim::operation;
        operation::init<kokkosp::kokkos_bundle>(
            operation::mode_constant<operation::init_mode::global>{});

        // add at least one
        if(kokkosp::kokkos_bundle::bundle_size() == 0)
            kokkosp::kokkos_bundle::configure<tim::component::wall_clock>();

        setup_kernel_logger();
    }

    void kokkosp_finalize_library()
    {
        printf("\n%s\n", kokkos_banner.c_str());
        printf("KokkosP: Finalization of timemory Connector. Complete.\n");
        printf("%s\n\n", kokkos_banner.c_str());

        kokkosp::cleanup();

        tim::timemory_finalize();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN('/', "kokkos", name)
                : TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *kernid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(*kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp::kokkos_bundle>(kernid);
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN('/', "kokkos", name)
                : TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *kernid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(*kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp::kokkos_bundle>(kernid);
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN('/', "kokkos", name)
                : TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *kernid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(*kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp::kokkos_bundle>(kernid);
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_fence(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN('/', "kokkos", name)
                : TIMEMORY_JOIN('/', "kokkos", TIMEMORY_JOIN("", "dev", devid), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *kernid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(*kernid);
    }

    void kokkosp_end_fence(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp::kokkos_bundle>(kernid);
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_push_profile_region(const char* name)
    {
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name);
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().push_back(
            kokkosp::profiler_t<kokkosp::kokkos_bundle>(name));
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().start();
    }

    void kokkosp_pop_profile_region()
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        if(kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().empty())
            return;
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().back().stop();
        kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>().pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        *secid     = kokkosp::get_unique_id();
        auto pname = TIMEMORY_JOIN('/', "kokkos", name);
        kokkosp::create_profiler<kokkosp::kokkos_bundle>(pname, *secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        kokkosp::destroy_profiler<kokkosp::kokkos_bundle>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_start_profile_section(uint32_t secid)
    {
        kokkosp::logger_t{}.mark(1, __FUNCTION__, secid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, secid);
        kokkosp::start_profiler<kokkosp::kokkos_bundle>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_allocate_data(const SpaceHandle space, const char* label,
                               const void* const ptr, const uint64_t size)
    {
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 TIMEMORY_JOIN("", '[', ptr, ']'), size);
        kokkosp::profiler_alloc_t<>{ TIMEMORY_JOIN('/', "kokkos/allocate", space.name,
                                                   label) }
            .store(std::plus<int64_t>{}, size);
    }

    void kokkosp_deallocate_data(const SpaceHandle space, const char* label,
                                 const void* const ptr, const uint64_t size)
    {
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 TIMEMORY_JOIN("", '[', ptr, ']'), size);
        kokkosp::profiler_alloc_t<>{ TIMEMORY_JOIN('/', "kokkos/deallocate", space.name,
                                                   label) }
            .store(std::plus<int64_t>{}, size);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name,
                                 const void* dst_ptr, SpaceHandle src_handle,
                                 const char* src_name, const void* src_ptr, uint64_t size)
    {
        kokkosp::logger_t{}.mark(1, __FUNCTION__, dst_handle.name, dst_name,
                                 TIMEMORY_JOIN("", '[', dst_ptr, ']'), src_handle.name,
                                 src_name, TIMEMORY_JOIN("", '[', src_ptr, ']'), size);

        auto name = TIMEMORY_JOIN('/', "kokkos/deep_copy",
                                  TIMEMORY_JOIN('=', dst_handle.name, dst_name),
                                  TIMEMORY_JOIN('=', src_handle.name, src_name));

        auto& _data = kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>();
        _data.emplace_back(name);
        _data.back().audit(dst_handle, dst_name, dst_ptr, src_handle, src_name, src_ptr,
                           size);
        _data.back().start();
        _data.back().store(std::plus<int64_t>{}, size);
    }

    void kokkosp_end_deep_copy()
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        auto& _data = kokkosp::get_profiler_stack<kokkosp::kokkos_bundle>();
        if(_data.empty())
            return;
        _data.back().store(std::minus<int64_t>{}, 0);
        _data.back().stop();
        _data.pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_profile_event(const char* name)
    {
        kokkosp::profiler_t<kokkosp::kokkos_bundle>{}.mark(name);
    }

    //----------------------------------------------------------------------------------//
}

//--------------------------------------------------------------------------------------//

TIMEMORY_INITIALIZE_STORAGE(kokkosp::memory_tracker)

//--------------------------------------------------------------------------------------//
