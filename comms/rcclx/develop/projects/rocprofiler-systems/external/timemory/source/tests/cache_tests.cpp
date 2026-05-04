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

#include "test_macros.hpp"
#include "timemory/units.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"

using namespace tim::component;

using floating_t  = double;
using cache_ptr_t = std::shared_ptr<tim::rusage_cache>;
using bundle_t =
    tim::component_bundle<TIMEMORY_API, wall_clock, user_clock, system_clock, peak_rss,
                          num_io_in, num_io_out, num_major_page_faults,
                          num_minor_page_faults, priority_context_switch,
                          voluntary_context_switch, current_peak_rss>;

// static const auto    memory_unit  = std::pair<int64_t, string_t>(tim::units::KiB,
// "KiB");
static const int64_t niter          = 20;
static const int64_t nelements      = 0.95 * (tim::units::get_page_size() * 500);
static auto          cache          = std::array<cache_ptr_t, 2>{};
static auto          bundle         = std::shared_ptr<bundle_t>{};
static auto          cache_bundle   = std::shared_ptr<bundle_t>{};
static auto          tot_size       = nelements * sizeof(int64_t);
static const double  peak_tolerance = 3 * tim::units::MB;

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

// get a random entry from vector
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

void
allocate_basic(bool _write = false)
{
    auto v    = std::vector<int64_t>(nelements, 15);
    auto ret  = fibonacci(0);
    long nfib = details::random_entry(v);
    auto ofs  = std::shared_ptr<std::ofstream>{};
    auto n    = niter;

    if(_write)
    {
        n *= 10;
        nfib -= 10;
        static bool       _first = true;
        std::stringstream _existing;
        if(!_first)
        {
            std::ifstream ifs(".cache_tests.dat");
            if(ifs)
            {
                while(!ifs.eof())
                {
                    std::string _inp;
                    ifs >> _inp;
                    _existing << _inp << '\n';
                }
            }
        }

        ofs = std::make_shared<std::ofstream>(".cache_tests.dat");
        if(ofs && !(*ofs))
            ofs.reset();
        if(ofs)
            *ofs << _existing.str();
        _first = false;
    }

    for(int64_t i = 0; i < n; ++i)
    {
        nfib = details::random_entry(v);
        ret += details::fibonacci(nfib);
        if(ofs && *ofs)
            *ofs << "i=" << i << ",nfib=" << nfib << ",ret=" << ret << '\n';
    }

    if(ret < 0)
        printf("fibonacci(%li) * %li = %li\n", (long) nfib, (long) niter, ret);
}

void
allocate()
{
    if(!bundle)
        bundle = std::make_shared<bundle_t>("allocate",
                                            tim::quirk::config<tim::quirk::no_store>{});

    if(!cache_bundle)
        cache_bundle = std::make_shared<bundle_t>(
            "allocate", tim::quirk::config<tim::quirk::no_store>{});

    bundle->reset();
    cache_bundle->reset();

    if(!cache.at(0))
        cache.at(0) = std::make_shared<tim::rusage_cache>();

    bundle->start();
    cache_bundle->start(*cache.at(0));

    allocate_basic();

    bundle->stop();
    cache_bundle->stop(*cache.at(0));

    if(!cache.at(1))
        cache.at(1) = std::make_shared<tim::rusage_cache>();
}

}  // namespace details

//--------------------------------------------------------------------------------------//

namespace
{
bool _memory_units_init =
    (tim::set_env(TIMEMORY_SETTINGS_PREFIX "MEMORY_UNITS", "B", 1), true);
}  // namespace

namespace trait = ::tim::trait;

class cache_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN

    // preform allocation only once here
    static void extra_setup()
    {
        EXPECT_TRUE(_memory_units_init);
        details::allocate();

        // disable roofline components and make sure ERT doesn't run excessively long
        tim::settings::ert_num_threads()      = 1;
        tim::settings::ert_num_streams()      = 1;
        tim::settings::ert_min_working_size() = 1000;
        tim::settings::ert_max_data_size()    = 10000;

        trait::runtime_enabled<cpu_roofline_flops>::set(false);
        trait::runtime_enabled<cpu_roofline_sp_flops>::set(false);
        trait::runtime_enabled<cpu_roofline_dp_flops>::set(false);
        trait::runtime_enabled<gpu_roofline_flops>::set(false);
        trait::runtime_enabled<gpu_roofline_sp_flops>::set(false);
        trait::runtime_enabled<gpu_roofline_dp_flops>::set(false);
        trait::runtime_enabled<cuda_event>::set(false);
        trait::runtime_enabled<cupti_activity>::set(false);
        trait::runtime_enabled<cupti_counters>::set(false);
        trait::runtime_enabled<cupti_pcsampling>::set(false);
    }

    static void extra_teardown()
    {
        for(auto& itr : cache)
            itr.reset();
        bundle.reset();
        cache_bundle.reset();
    }

    TIMEMORY_TEST_SUITE_SETUP(extra_setup();)
    TIMEMORY_TEST_SUITE_TEARDOWN(extra_teardown();)
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
auto
run1()
{
    using trait_type                = typename tim::trait::cache<Tp>::type;
    static constexpr bool is_null_v = tim::concepts::is_null_type<trait_type>::value;
    using type =
        tim::conditional_t<is_null_v, tim::type_list<>, tim::type_list<trait_type>>;
    using ttype     = tim::convert_t<tim::type_concat_t<type>, std::tuple<>>;
    using uniq_type = tim::mpl::unique_t<type, std::tuple<>>;
    std::cout << "\ttype           : " << tim::demangle<Tp>() << std::endl;
    std::cout << "\ttrait type     : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "\ttrait is null  : " << std::boolalpha << is_null_v << std::endl;
    std::cout << "\ttype list type : " << tim::demangle<type>() << std::endl;
    std::cout << "\ttuple type     : " << tim::demangle<ttype>() << std::endl;
    std::cout << "\tunique type    : " << tim::demangle<uniq_type>() << std::endl;
    return uniq_type{};
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
auto
run2()
{
    using trait_type =
        typename tim::mpl::impl::get_trait_type_tuple<tim::trait::cache, Tp>::trait_type;
    static constexpr bool is_null_v =
        tim::mpl::impl::get_trait_type_tuple<tim::trait::cache, Tp>::is_null_v;
    using type      = tim::mpl::impl::get_trait_type_tuple_t<tim::trait::cache, Tp>;
    using ttype     = tim::convert_t<tim::type_concat_t<type>, std::tuple<>>;
    using uniq_type = tim::mpl::unique_t<type, std::tuple<>>;
    std::cout << "\ttype           : " << tim::demangle<Tp>() << std::endl;
    std::cout << "\ttrait type     : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "\ttrait is null  : " << std::boolalpha << is_null_v << std::endl;
    std::cout << "\ttype list type : " << tim::demangle<type>() << std::endl;
    std::cout << "\ttuple type     : " << tim::demangle<ttype>() << std::endl;
    std::cout << "\tunique type    : " << tim::demangle<uniq_type>() << std::endl;
    return uniq_type{};
}

//--------------------------------------------------------------------------------------//

template <typename... Tp>
auto
run3()
{
    using namespace tim::stl;
    using namespace tim::stl::ostream;

    using trait_type =
        typename tim::mpl::impl::get_trait_type<tim::trait::cache, Tp...>::trait_type;
    using type      = tim::mpl::impl::get_trait_type_t<tim::trait::cache, Tp...>;
    using ttype     = tim::convert_t<type, std::tuple<>>;
    using uniq_type = tim::mpl::unique_t<type, std::tuple<>>;
    std::cout << "\ttype           : " << tim::demangle<std::tuple<Tp...>>() << std::endl;
    std::cout << "\ttrait type     : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "\ttype list type : " << tim::demangle<type>() << std::endl;
    std::cout << "\ttuple type     : " << tim::demangle<ttype>() << std::endl;
    std::cout << "\tunique type    : " << tim::demangle<uniq_type>() << std::endl;
    return uniq_type{};
}

//--------------------------------------------------------------------------------------//

std::string
print_rusage_cache(const tim::rusage_cache& _rusage)
{
    std::stringstream _msg;
    _msg << "\tpeak_rss          : " << _rusage.get_peak_rss() << '\n';
    _msg << "\tkernel mode time  : " << _rusage.get_kernel_mode_time() << '\n';
    _msg << "\tuser mode time    : " << _rusage.get_user_mode_time() << '\n';
    _msg << "\tio in             : " << _rusage.get_num_io_in() << '\n';
    _msg << "\tio out            : " << _rusage.get_num_io_out() << '\n';
    _msg << "\tmajor page faults : " << _rusage.get_num_major_page_faults() << '\n';
    _msg << "\tminor page faults : " << _rusage.get_num_minor_page_faults() << '\n';
    _msg << "\tprio ctx switch   : " << _rusage.get_num_priority_context_switch() << '\n';
    _msg << "\tvol ctx switch    : " << _rusage.get_num_voluntary_context_switch()
         << '\n';
    if(tim::settings::debug() || tim::settings::verbose() > 0)
        std::cerr << _msg.str() << std::endl;
    return _msg.str();
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, units)
{
    auto _settings = tim::settings::instance();
    EXPECT_TRUE(_settings->get_initialized());
    EXPECT_EQ(tim::get_env<std::string>("TIMEMORY_MEMORY_UNITS", "", false),
              std::string{ "B" });
    EXPECT_EQ(_settings->get_memory_units(), std::string{ "B" });
    EXPECT_EQ(std::get<0>(tim::units::get_memory_unit(_settings->get_memory_units())),
              std::string{ "byte" });
    EXPECT_EQ(peak_rss::get_unit(), tim::units::byte);
    EXPECT_EQ(page_rss::get_unit(), tim::units::byte);
    EXPECT_EQ(peak_rss::get_display_unit(), "byte");
    EXPECT_EQ(page_rss::get_display_unit(), "byte");
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, peak_rss)
{
    auto ret1 = run1<peak_rss>();
    puts("");
    auto ret2 = run2<peak_rss>();

    auto check1 = std::is_same<decltype(ret1), std::tuple<tim::rusage_cache>>::value;
    auto check2 = std::is_same<decltype(ret2), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check1);
    EXPECT_TRUE(check2);
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, wall_clock)
{
    auto ret1 = run1<wall_clock>();
    puts("");
    auto ret2   = run2<wall_clock>();
    auto check1 = std::is_same<decltype(ret1), std::tuple<>>::value;
    auto check2 = std::is_same<decltype(ret2), std::tuple<>>::value;
    EXPECT_TRUE(check1);
    EXPECT_TRUE(check2);
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, peak_rss_wall_clock)
{
    auto ret   = run3<peak_rss, wall_clock>();
    auto check = std::is_same<decltype(ret), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check);
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, rusage)
{
    using data_type =
        std::tuple<peak_rss, current_peak_rss, num_io_in, num_io_out, wall_clock>;
    using trait_type    = tim::mpl::get_trait_type_t<tim::trait::cache, data_type>;
    using cache_type    = typename tim::operation::construct_cache<data_type>::type;
    auto        _cache  = tim::invoke::get_cache<data_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    auto _msg  = print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check) << _msg;
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, component_bundle)
{
    using bundle_type   = tim::component_bundle<TIMEMORY_API, peak_rss, current_peak_rss,
                                              num_io_in*, num_io_out*, wall_clock>;
    using data_type     = typename bundle_type::data_type;
    using trait_type    = tim::mpl::get_trait_type_t<tim::trait::cache, bundle_type>;
    using cache_type    = tim::operation::construct_cache_t<bundle_type>;
    auto        _cache  = tim::invoke::get_cache<bundle_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "bundle type       : " << tim::demangle<bundle_type>() << std::endl;
    std::cout << "data type         : " << tim::demangle<data_type>() << std::endl;
    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    auto _msg  = print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check) << _msg;
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, auto_bundle)
{
    using bundle_type   = tim::auto_bundle<TIMEMORY_API, peak_rss, current_peak_rss,
                                         num_io_in*, num_io_out*, wall_clock*>;
    using data_type     = typename bundle_type::data_type;
    using trait_type    = tim::mpl::get_trait_type_t<tim::trait::cache, bundle_type>;
    using cache_type    = tim::operation::construct_cache_t<bundle_type>;
    auto        _cache  = tim::invoke::get_cache<bundle_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "bundle type       : " << tim::demangle<bundle_type>() << std::endl;
    std::cout << "data type         : " << tim::demangle<data_type>() << std::endl;
    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    auto _msg  = print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check) << _msg;
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, component_tuple)
{
    using bundle_type   = tim::component_tuple<peak_rss, current_peak_rss, num_io_in,
                                             num_io_out, wall_clock>;
    using data_type     = typename bundle_type::data_type;
    using trait_type    = tim::mpl::get_trait_type_t<tim::trait::cache, bundle_type>;
    using cache_type    = tim::operation::construct_cache_t<bundle_type>;
    auto        _cache  = tim::invoke::get_cache<bundle_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "bundle type       : " << tim::demangle<bundle_type>() << std::endl;
    std::cout << "data type         : " << tim::demangle<data_type>() << std::endl;
    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    auto _msg  = print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check) << _msg;
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, auto_tuple)
{
    using bundle_type =
        tim::auto_tuple<peak_rss, current_peak_rss, num_io_in, num_io_out, wall_clock>;
    using data_type     = typename bundle_type::data_type;
    using trait_type    = tim::mpl::get_trait_type_t<tim::trait::cache, bundle_type>;
    using cache_type    = tim::operation::construct_cache_t<bundle_type>;
    auto        _cache  = tim::invoke::get_cache<bundle_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "bundle type       : " << tim::demangle<bundle_type>() << std::endl;
    std::cout << "data type         : " << tim::demangle<data_type>() << std::endl;
    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    auto _msg  = print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check) << _msg;
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, component_list)
{
    using bundle_type   = tim::component_list<peak_rss, current_peak_rss, num_io_in,
                                            num_io_out, wall_clock>;
    using data_type     = typename bundle_type::data_type;
    using trait_type    = tim::mpl::get_trait_type_t<tim::trait::cache, bundle_type>;
    using cache_type    = tim::operation::construct_cache_t<bundle_type>;
    auto        _cache  = tim::invoke::get_cache<bundle_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "bundle type       : " << tim::demangle<bundle_type>() << std::endl;
    std::cout << "data type         : " << tim::demangle<data_type>() << std::endl;
    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    auto _msg  = print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check) << _msg;
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, auto_list)
{
    using bundle_type =
        tim::auto_list<peak_rss, current_peak_rss, num_io_in, num_io_out, wall_clock>;
    using data_type     = typename bundle_type::data_type;
    using trait_type    = tim::mpl::get_trait_type_t<tim::trait::cache, bundle_type>;
    using cache_type    = tim::operation::construct_cache_t<bundle_type>;
    auto        _cache  = tim::invoke::get_cache<bundle_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "bundle type       : " << tim::demangle<bundle_type>() << std::endl;
    std::cout << "data type         : " << tim::demangle<data_type>() << std::endl;
    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    auto _msg  = print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check) << _msg;
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, lightweight_tuple)
{
    using bundle_type   = tim::lightweight_tuple<peak_rss, current_peak_rss, num_io_in,
                                               num_io_out, wall_clock>;
    using data_type     = typename bundle_type::data_type;
    using trait_type    = tim::mpl::get_trait_type_t<tim::trait::cache, bundle_type>;
    using cache_type    = tim::operation::construct_cache_t<bundle_type>;
    auto        _cache  = tim::invoke::get_cache<bundle_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "bundle type       : " << tim::demangle<bundle_type>() << std::endl;
    std::cout << "data type         : " << tim::demangle<data_type>() << std::endl;
    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    auto _msg  = print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check) << _msg;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Arg,
          tim::enable_if_t<tim::trait::is_available<Tp>::value> = 0>
double
get_value(Arg& _arg)
{
    auto _obj = _arg.template get<Tp>();
    if(_obj)
        return std::get<0>(_obj->get());
    return 0;
}

template <typename Tp, typename Arg,
          tim::enable_if_t<!tim::trait::is_available<Tp>::value> = 0>
double
get_value(Arg&)
{
    return -1.0;
}

//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_LINUX)
TEST_F(cache_tests, io)
{
    if(!std::ifstream{ TIMEMORY_JOIN('/', "/proc", tim::process::get_target_id(), "io") })
        return;
    using io_bundle_t =
        tim::component_tuple<read_bytes, read_char, written_bytes, written_char>;
    using timing_t   = tim::auto_tuple<wall_clock>;
    using cache_type = tim::io_cache;

    int                   n = 50;
    std::array<double, 4> _tot;
    _tot.fill(0.0);

    for(int i = 0; i < n; ++i)
    {
        io_bundle_t _bundle{ details::get_test_name() };
        {
            TIMEMORY_BLANK_MARKER(timing_t, details::get_test_name(), '/', "non-cached");
            // traditional
            _bundle.start();
            details::allocate_basic(true);
            _bundle.stop();
        }
        {
            TIMEMORY_BLANK_MARKER(timing_t, details::get_test_name(), '/', "cached");
            // using cache
            _bundle.start(cache_type{});
            details::allocate_basic(true);
            _bundle.stop(cache_type{});
        }

        std::cout << _bundle << std::endl;
        if(i + 1 == n)
        {
            std::cout << "[io_cache]\n" << cache_type{} << std::flush;
            std::stringstream cmd;
            cmd << "cat /proc/" << tim::process::get_id() << "/io";
            std::cout << "[system]\n";
            auto ret = std::system(cmd.str().c_str());
            if(ret != 0)
                printf("[std::system]> cmd: '%s' failed\n", cmd.str().c_str());
        }

        // for checks
        _tot.at(0) += get_value<read_bytes>(_bundle);
        _tot.at(1) += get_value<written_bytes>(_bundle);
        _tot.at(2) += get_value<read_char>(_bundle);
        _tot.at(3) += get_value<written_char>(_bundle);
    }

    std::array<std::string, 4> _labels = { { "read_bytes", "written_bytes", "read_char",
                                             "written_char" } };

    for(int i = 0; i < 4; ++i)
    {
        if(_tot.at(i) == -1.0 * n)
            continue;
        else if(_labels.at(i).find("bytes") != std::string::npos)
            EXPECT_GE(_tot.at(i), 0.0) << " " << _labels.at(i) << " = " << _tot.at(i);
        else
            EXPECT_GT(_tot.at(i), 0.0) << " " << _labels.at(i) << " = " << _tot.at(i);
    }
}
#endif
//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, validation)
{
    // puts("\n>>> INITIAL CACHE <<<\n");
    // puts(print_rusage_cache(*cache.at(0)).c_str());
    // puts("\n>>> BUNDLE <<<\n");
    // std::cout << *bundle << std::endl;
    // puts("\n>>> CACHE BUNDLE <<<\n");
    // std::cout << *cache_bundle << std::endl;
    // puts("\n>>> FINAL CACHE <<<\n");
    // puts(print_rusage_cache(*cache.at(1)).c_str());

    EXPECT_NEAR(tot_size, bundle->get<peak_rss>()->get(), peak_tolerance);
    EXPECT_NEAR(std::get<0>(bundle->get<current_peak_rss>()->get()),
                cache.at(0)->get_peak_rss(), peak_tolerance);
    EXPECT_NEAR(std::get<1>(bundle->get<current_peak_rss>()->get()),
                cache.at(1)->get_peak_rss(), peak_tolerance);
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, complete_tuple)
{
    tim::settings::debug()   = false;
    tim::settings::verbose() = 0;

    namespace component = tim::component;
    namespace mpl       = tim::mpl;

    tim::component_tuple_t<TIMEMORY_COMPONENT_TYPES> _bundle{ details::get_test_name() };
    auto&                                            _bcache = *cache.at(0);
    auto&                                            _ecache = *cache.at(1);
    _bundle.start(_bcache);
    details::consume(1000);
    _bundle.store(mpl::piecewise_ignore<perfetto_trace>{}, 1000);
    _bundle.store(mpl::piecewise_ignore<perfetto_trace>{}, 1000.0);
    _bundle.stop(_ecache);

    // puts("\n>>> INITIAL CACHE <<<\n");
    // puts(print_rusage_cache(_bcache).c_str());
    // puts("\n>>> BUNDLE <<<\n");
    // std::cout << _bundle << std::endl;
    // puts("\n>>> FINAL CACHE <<<\n");
    // puts(print_rusage_cache(_ecache).c_str());

    _bundle.push(mpl::piecewise_select<data_tracker_floating, data_tracker_integer,
                                       data_tracker_unsigned>{});
    _bundle.store(mpl::piecewise_ignore<perfetto_trace>{}, 100);
    _bundle.store(mpl::piecewise_ignore<perfetto_trace>{}, 100.0);
    _bundle.pop(mpl::piecewise_select<data_tracker_floating, data_tracker_integer,
                                      data_tracker_unsigned>{});

    EXPECT_NEAR(tot_size, _bundle.get<peak_rss>()->get(), peak_tolerance);
    EXPECT_NEAR(std::get<0>(_bundle.get<current_peak_rss>()->get()),
                _bcache.get_peak_rss(), peak_tolerance);
    EXPECT_NEAR(std::get<1>(_bundle.get<current_peak_rss>()->get()),
                _ecache.get_peak_rss(), peak_tolerance);

    // these should use cache
    if(tim::trait::is_available<user_mode_time>::value)
    {
        // if the user mode time is less than the resolution of user_clock
        // then this test will fail
        auto _usrclk_val = bundle->get<user_clock>()->get();
        if(_usrclk_val >= 1.0e-3)
        {
            EXPECT_NEAR(_bundle.get<user_mode_time>()->get(), _usrclk_val,
                        1.5 * _usrclk_val);
        }
    }

    if(tim::trait::is_available<kernel_mode_time>::value)
    {
        // if the kernel mode time is less than the resolution of system_clock
        // then this test will fail
        auto _sysclk_val = bundle->get<system_clock>()->get();
        if(_sysclk_val >= 1.0e-3)
        {
            EXPECT_NEAR(_bundle.get<kernel_mode_time>()->get(), _sysclk_val,
                        1.5 * _sysclk_val);
        }
    }

    auto fp_storage = tim::storage<data_tracker_floating>::instance()->get();
    auto ui_storage = tim::storage<data_tracker_unsigned>::instance()->get();
    auto si_storage = tim::storage<data_tracker_integer>::instance()->get();

    EXPECT_EQ(fp_storage.back().data().get_laps(), 2);
    EXPECT_EQ(ui_storage.back().data().get_laps(), 2);
    EXPECT_EQ(si_storage.back().data().get_laps(), 2);

    EXPECT_NEAR(fp_storage.back().stats().get_min(), 100.0, 1.0e-6);
    EXPECT_EQ(ui_storage.back().stats().get_min(), 100);
    EXPECT_EQ(si_storage.back().stats().get_min(), 100);

    EXPECT_NEAR(fp_storage.back().stats().get_max(), 1000.0, 1.0e-6);
    EXPECT_EQ(ui_storage.back().stats().get_max(), 1000);
    EXPECT_EQ(si_storage.back().stats().get_max(), 1000);

    EXPECT_NEAR(fp_storage.back().stats().get_mean(), 550.0, 1.0e-6);
    EXPECT_EQ(ui_storage.back().stats().get_mean(), 550);
    EXPECT_EQ(si_storage.back().stats().get_mean(), 550);

    EXPECT_NEAR(fp_storage.back().stats().get_stddev(), 636.396, 1.0e-1);
    EXPECT_EQ(ui_storage.back().stats().get_stddev(), 636);
    EXPECT_EQ(si_storage.back().stats().get_stddev(), 636);
}

//--------------------------------------------------------------------------------------//
