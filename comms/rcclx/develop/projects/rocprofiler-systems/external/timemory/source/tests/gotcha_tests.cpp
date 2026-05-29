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
//

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "gotcha_tests_lib.hpp"
#include "timemory/components/gotcha/memory_allocations.hpp"
#include "timemory/mpl/macros.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/timemory.hpp"

#include "gtest/gtest.h"
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using tim::component_tuple_t;

// create a hybrid for inside the gotcha
using gotcha_hybrid_t =
    tim::auto_bundle_t<TIMEMORY_API, wall_clock, peak_rss, cpu_clock*>;

// create gotcha types for various bundles of functions
using mpi_gotcha_t    = tim::component::gotcha<1, gotcha_hybrid_t>;
using work_gotcha_t   = tim::component::gotcha<1, gotcha_hybrid_t, int>;
using memfun_gotcha_t = tim::component::gotcha<5, gotcha_hybrid_t>;
using malloc_gotcha_t = malloc_gotcha::gotcha_type<component_tuple_t<>>;

TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, mpi_gotcha_t, priority_constant<256>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, malloc_gotcha_t, priority_constant<512>)

TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, mpi_gotcha_t, priority_constant<-256>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, malloc_gotcha_t, priority_constant<-512>)

using auto_hybrid_t =
    tim::component_bundle_t<TIMEMORY_API, wall_clock, cpu_clock, peak_rss, mpi_gotcha_t,
                            work_gotcha_t, memfun_gotcha_t, malloc_gotcha_t, cpu_clock*>;

template <typename Tp>
using vector_t = std::vector<Tp>;

static constexpr int64_t nitr      = 100000;
static const double      tolerance = 1.0e-2;
static constexpr size_t  exp_nitr  = 10;

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

//--------------------------------------------------------------------------------------//

inline auto&
get_cleanup()
{
    static std::vector<std::function<void()>> _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline void
cleanup()
{
    for(auto& itr : get_cleanup())
        itr();
    get_cleanup().clear();
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
auto
allocate(int64_t nsize)
{
    Tp* ptr = (Tp*) malloc(nsize * sizeof(Tp));
    memset(ptr, 0, nsize * sizeof(Tp));
    get_cleanup().emplace_back([=]() { free(ptr); });
    std::vector<Tp> buf{};
    buf.assign(ptr, ptr + nsize);
    return buf;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline vector_t<Tp>
generate(const int64_t& nsize)
{
    auto sendbuf = allocate<Tp>(nsize);

    std::mt19937 rng;
    rng.seed(54561434UL);
    auto dist = [&]() { return std::generate_canonical<Tp, 10>(rng); };
    std::generate(sendbuf.begin(), sendbuf.end(), [&]() { return dist(); });
    return sendbuf;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline void
generate(const int64_t& nsize, std::vector<Tp>& sendbuf)
{
    sendbuf.clear();
    sendbuf = allocate<Tp>(nsize);

    for(auto& itr : sendbuf)
        itr = static_cast<Tp>(0.0);
    std::mt19937 rng;
    rng.seed(54561434UL);
    auto dist = [&]() { return std::generate_canonical<Tp, 10>(rng); };
    std::generate(sendbuf.begin(), sendbuf.end(), [&]() { return dist(); });
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline vector_t<Tp>
allreduce(const vector_t<Tp>& sendbuf)
{
    auto recvbuf = allocate<Tp>(sendbuf.size());
#if defined(TIMEMORY_USE_MPI)
    auto dtype = (std::is_same<Tp, float>::value) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), dtype, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    std::copy(sendbuf.begin(), sendbuf.end(), recvbuf.begin());
#endif
    return recvbuf;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline void
allreduce(const vector_t<Tp>& sendbuf, vector_t<Tp>& recvbuf)
{
    recvbuf = allocate<Tp>(sendbuf.size());
#if defined(TIMEMORY_USE_MPI)
    auto dtype = (std::is_same<Tp, float>::value) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), dtype, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    std::copy(sendbuf.begin(), sendbuf.end(), recvbuf.begin());
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace details

//======================================================================================//

class gotcha_tests : public ::testing::Test
{
protected:
    void TearDown() override { details::cleanup(); }

    static void SetUpTestSuite()
    {
        tim::settings::width()        = 16;
        tim::settings::precision()    = 6;
        tim::settings::timing_units() = "sec";
        tim::settings::memory_units() = "kB";
        tim::settings::verbose()      = 0;
        tim::settings::debug()        = false;
        tim::settings::json_output()  = true;
        tim::settings::mpi_thread()   = false;
        tim::dmp::initialize(test_argc, test_argv);
        tim::timemory_init(&test_argc, &test_argv);
        tim::settings::dart_output() = true;
        tim::settings::dart_count()  = 1;
        tim::settings::dart_type()   = "peak_rss";
#if defined(TIMEMORY_USE_PAPI)
        cpu_roofline_sp_flops::ert_config_type<float>::configure(1, 64);
        cpu_roofline_dp_flops::ert_config_type<double>::configure(1, 64);
#endif
    }

    static void TearDownTestSuite()
    {
        tim::timemory_finalize();
        tim::manager::master_instance().reset();
        tim::dmp::finalize();
    }
};

//======================================================================================//

TEST_F(gotcha_tests, mpi_explicit)
{
    mpi_gotcha_t::get_initializer() = [=]() {
#if defined(TIMEMORY_USE_MPI)
        TIMEMORY_PRINT_HERE("%s", details::get_test_name().c_str());
        mpi_gotcha_t::configure<0, int, const void*, void*, int, MPI_Datatype, MPI_Op,
                                MPI_Comm>("MPI_Allreduce");
#endif
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto fsendbuf = details::generate<float>(1000);
        auto frecvbuf = details::allreduce(fsendbuf);
        fsum +=
            static_cast<float>(std::accumulate(frecvbuf.begin(), frecvbuf.end(), 0.0));

        auto dsendbuf = details::generate<double>(1000);
        auto drecvbuf = details::allreduce(dsendbuf);
        dsum += std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);
        details::cleanup();
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank,
                   static_cast<double>(fsum));
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    EXPECT_NEAR(fsum, 49892284.00 * size, tolerance);
    EXPECT_NEAR(dsum, 49868704.48 * size, tolerance);
}

//======================================================================================//

TEST_F(gotcha_tests, mpi_macro)
{
    mpi_gotcha_t::get_initializer() = [=]() {
#if defined(TIMEMORY_USE_MPI)
        TIMEMORY_PRINT_HERE("%s", details::get_test_name().c_str());
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 0, MPI_Allreduce);
#endif
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto fsendbuf = details::generate<float>(1000);
        auto frecvbuf = details::allreduce(fsendbuf);
        fsum +=
            static_cast<float>(std::accumulate(frecvbuf.begin(), frecvbuf.end(), 0.0));

        auto dsendbuf = details::generate<double>(1000);
        auto drecvbuf = details::allreduce(dsendbuf);
        dsum += std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);
        details::cleanup();
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank,
                   static_cast<double>(fsum));
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    EXPECT_NEAR(fsum, 49892284.00 * size, tolerance);
    EXPECT_NEAR(dsum, 49868704.48 * size, tolerance);
}

//======================================================================================//

TEST_F(gotcha_tests, work_explicit)
{
    using tuple_type = std::tuple<float, double>;
    using pair_type  = std::pair<float, double>;

    work_gotcha_t::get_initializer() = [=]() {
        TIMEMORY_PRINT_HERE("%s", details::get_test_name().c_str());
        auto mangled_do_work = tim::mangle<decltype(ext::do_work)>("ext::do_work");
        work_gotcha_t::configure<0, tuple_type, int64_t, const pair_type&>(
            mangled_do_work);
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto ret = ext::do_work(1000, pair_type(0.25, 0.125));
        fsum += std::get<0>(ret);
        dsum += std::get<1>(ret);
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank,
                   static_cast<double>(fsum));
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    EXPECT_NEAR(fsum, -2416347.50, tolerance);
    EXPECT_NEAR(dsum, -1829370.79, tolerance);
}

//======================================================================================//

TEST_F(gotcha_tests, work_macro)
{
    using pair_type = std::pair<float, double>;

    work_gotcha_t::get_initializer() = [=]() {
        TIMEMORY_PRINT_HERE("%s", details::get_test_name().c_str());
        TIMEMORY_CXX_GOTCHA(work_gotcha_t, 0, ext::do_work);
    };

    auto _exec = [&]() {
        TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

        float  fsum = 0.0;
        double dsum = 0.0;
        for(int i = 0; i < nitr; ++i)
        {
            auto ret = ext::do_work(1000, pair_type(0.25, 0.125));
            fsum += std::get<0>(ret);
            dsum += std::get<1>(ret);
        }

        auto rank = tim::mpi::rank();
        auto size = tim::mpi::size();
        for(int i = 0; i < size; ++i)
        {
            tim::mpi::barrier();
            if(i == rank)
            {
                printf("\n");
                printf("[%i]> single-precision sum = %8.2f\n", rank,
                       static_cast<double>(fsum));
                printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
            }
            tim::mpi::barrier();
        }
        tim::mpi::barrier();
        if(rank == 0)
            printf("\n");

        EXPECT_NEAR(fsum, -2416347.50, tolerance);
        EXPECT_NEAR(dsum, -1829370.79, tolerance);
    };

    for(auto i = 0; i < 4; ++i)
        _exec();
}

//======================================================================================//

template <typename func_t>
void
print_func_info(const std::string& fname)
{
    using ret_type = typename tim::mpl::function_traits<func_t>::result_type;
    using arg_type = typename tim::mpl::function_traits<func_t>::args_type;
    std::cout << std::endl;
    std::cout << "  func name = " << fname << std::endl;
    std::cout << "memfun type = " << tim::demangle(typeid(func_t).name()) << std::endl;
    std::cout << "result type = " << tim::demangle(typeid(ret_type).name()) << std::endl;
    std::cout << "  args type = " << tim::demangle(typeid(arg_type).name()) << std::endl;
    std::cout << std::endl;
}

//======================================================================================//

TEST_F(gotcha_tests, malloc_gotcha)
{
    auto _settings           = tim::settings::push<TIMEMORY_API>();
    tim::settings::debug()   = false;
    tim::settings::verbose() = 0;

    using toolset_t =
        tim::auto_tuple_t<wall_clock, peak_rss, memory_allocations, mpi_gotcha_t>;

    mpi_gotcha_t::get_initializer() = [=]() {
#if defined(TIMEMORY_USE_MPI)
        TIMEMORY_PRINT_HERE("%s", details::get_test_name().c_str());
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 0, MPI_Allreduce);
#endif
    };

    TIMEMORY_PRINT_HERE("%s", "starting");
    toolset_t tool(details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    {
        for(int i = 0; i < nitr / 10; ++i)
        {
            std::vector<float>  fsendbuf, frecvbuf;
            std::vector<double> dsendbuf, drecvbuf;

            details::generate<float>(1000, fsendbuf);
            details::allreduce(fsendbuf, frecvbuf);
            fsum += static_cast<float>(
                std::accumulate(frecvbuf.begin(), frecvbuf.end(), 0.0));

            details::generate<double>(1000, dsendbuf);
            details::allreduce(dsendbuf, drecvbuf);
            dsum += std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);

            details::cleanup();
        }
    }

    tool.stop();
    TIMEMORY_PRINT_HERE("%s", "stopped");

    // malloc_gotcha& mc = *tool.get<malloc_gotcha>();
    // std::cout << mc << std::endl;

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();

    EXPECT_GE(rank, 0);
    EXPECT_GE(size, 1);
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank,
                   static_cast<double>(fsum));
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    EXPECT_NEAR(fsum, 4986708.50 * size, tolerance);
    EXPECT_NEAR(dsum, 4986870.45 * size, tolerance);

    malloc_gotcha::tear_down<component_tuple_t<>>();

    auto _data = tim::storage<malloc_gotcha>::instance()->get();
    EXPECT_GE(_data.size(), 2);
    for(auto& itr : _data)
    {
        if(itr.prefix() == "calloc")
            continue;
        EXPECT_GE(itr.data().get(), 240000.) << itr.prefix() << ": " << itr.data();
    }

    tim::settings::pop<TIMEMORY_API>();
    tim::settings::debug()   = _settings->get_debug();
    tim::settings::verbose() = _settings->get_verbose();
}

//======================================================================================//

TEST_F(gotcha_tests, void_function)
{
    auto _dbg              = tim::settings::debug();
    tim::settings::debug() = true;

    using puts_bundle_t = tim::component_tuple<trip_count>;
    using puts_gotcha_t = tim::component::gotcha<1, puts_bundle_t>;
    using void_bundle_t = tim::lightweight_tuple<puts_gotcha_t>;

    puts_gotcha_t::get_initializer() = [=]() {
        TIMEMORY_CXX_GOTCHA(puts_gotcha_t, 0, ext::do_puts);
    };

    auto _beg_ct = tim::storage<trip_count>::instance()->size();

    void_bundle_t _bundle{ details::get_test_name() };
    _bundle.start();
    for(int i = 0; i < 10; ++i)
        ext::do_puts(details::get_test_name().c_str());
    _bundle.stop();

    auto _end_ct = tim::storage<trip_count>::instance()->size();
    EXPECT_EQ(_end_ct - _beg_ct, 1) << " begin: " << _beg_ct << ", end: " << _end_ct;

    tim::settings::debug() = _dbg;
}

//======================================================================================//

namespace tim
{
namespace component
{
struct cmath_intercept : public base<cmath_intercept, void>
{
    static auto& get_intercepts()
    {
        static std::atomic<unsigned long> _instance{ 0 };
        return _instance;
    }

    static auto& puts_intercepts()
    {
        static std::atomic<unsigned long> _instance{ 0 };
        return _instance;
    }

    TIMEMORY_NOINLINE void operator()(const char* msg) const
    {
        auto _n = ++puts_intercepts();
        if(_n < 10)
            puts(msg);
    }

    TIMEMORY_NOINLINE double operator()(double val) const
    {
        ++get_intercepts();
        return exp(val);
    }
};
}  // namespace component
}  // namespace tim

using cmath_intercept_t = gotcha<4, std::tuple<>, cmath_intercept>;

TEST_F(gotcha_tests, replacement)
{
    auto _dbg = tim::settings::debug();
    // tim::settings::debug() = true;

    using exp_bundle_t = tim::component_bundle<TIMEMORY_API, cmath_intercept_t>;

    static_assert(cmath_intercept_t::components_size == 0,
                  "cmath_intercept_t should have no components");
    static_assert(cmath_intercept_t::replaces, "cmath_intercept_t does not replace exp!");

    //
    // configure the initializer for the gotcha component which replaces exp
    //
    cmath_intercept_t::get_default_ready() = true;
    cmath_intercept_t::get_initializer()   = []() {
        puts("Generating exp intercept...");
#if defined(TIMEMORY_HIPCC)
        cmath_intercept_t::template instrument<0, double, double>::generate("exp");
#else
        TIMEMORY_C_GOTCHA(cmath_intercept_t, 0, exp);
#endif
        // TIMEMORY_C_GOTCHA(cmath_intercept_t, 1, test_exp);
        TIMEMORY_CXX_GOTCHA(cmath_intercept_t, 2, ext::do_puts);
#if defined(TIMEMORY_HIPCC)
        cmath_intercept_t::template instrument<0, double, double>::generate(
            "__exp_finite");
#else
        TIMEMORY_DERIVED_GOTCHA(cmath_intercept_t, 3, exp, "__exp_finite");
#endif
    };

    auto _func = [](bool _first) {
        if(!_first)
            cmath_intercept_t::enable();

        auto inp = std::min<double>(10.0, exp(5.0));
        {
            auto _configured = cmath_intercept_t::is_configured();
            if(_first)
            {
                EXPECT_FALSE(_configured);
            }
            else
            {
                EXPECT_TRUE(_configured);
            }
        }
        auto _handle = exp_bundle_t(details::get_test_name());
        {
            auto _info = cmath_intercept_t::get_info();
            EXPECT_GE(_info[0], 0) << "# ready";
            if(_first)
            {
                EXPECT_GE(_info[1], 0) << "# filled";
            }
            else
            {
                EXPECT_GE(_info[1], 2) << "# filled";
            }
            EXPECT_GE(_info[2], 0) << "# is_active";
            EXPECT_EQ(_info[3], 0) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        _handle.start();
        {
            auto _configured = cmath_intercept_t::is_configured();
            EXPECT_TRUE(_configured);
            auto _info = cmath_intercept_t::get_info();
            EXPECT_GE(_info[0], 2) << "# ready";
            EXPECT_GE(_info[1], 2) << "# filled";
            EXPECT_GE(_info[2], 2) << "# is_active";
            EXPECT_EQ(_info[3], 0) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        volatile double ret = inp;
        for(int i = 0; i < 10; ++i)
        {
            auto val = ret / ::pow(ret, 2);
            ret += test_exp(val);
            ext::do_puts(TIMEMORY_JOIN("", "computing exp(", val, ")").c_str());
        }
        {
            auto _info = cmath_intercept_t::get_info();
            EXPECT_GE(_info[0], 2) << "# ready";
            EXPECT_GE(_info[1], 2) << "# filled";
            EXPECT_GE(_info[2], 2) << "# is_active";
            EXPECT_EQ(_info[3], 0) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        _handle.stop();
        {
            auto _info = cmath_intercept_t::get_info();
            EXPECT_GE(_info[0], 0) << "# ready";
            EXPECT_GE(_info[1], 2) << "# filled";
            EXPECT_GE(_info[2], 0) << "# is_active";
            EXPECT_EQ(_info[3], 0) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        cmath_intercept_t::disable();
        {
            auto _configured = cmath_intercept_t::is_configured();
            EXPECT_FALSE(_configured);
            auto _info = cmath_intercept_t::get_info();
            EXPECT_GE(_info[0], 0) << "# ready";
            EXPECT_GE(_info[1], 2) << "# filled";
            EXPECT_GE(_info[2], 0) << "# is_active";
            EXPECT_GE(_info[3], 2) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        if(_first)
            printf("result: %f\n", ret);
        EXPECT_GT(ret, 10.9) << "result: " << ret;
    };

    for(size_t j = 0; j < exp_nitr; ++j)
    {
        printf("Starting iteration %zu...\n", j);
        _func(j == 0);
    }

    printf("number of exp  intercepts: %lu\n", cmath_intercept::get_intercepts().load());
    printf("number of puts intercepts: %lu\n", cmath_intercept::puts_intercepts().load());

    EXPECT_EQ(cmath_intercept::get_intercepts(), 10 * exp_nitr);
    EXPECT_EQ(cmath_intercept::puts_intercepts(), 10 * exp_nitr);

    tim::settings::debug() = _dbg;
}

//======================================================================================//

namespace tim
{
namespace component
{
struct cmath_reentry : public base<cmath_reentry, void>
{
    static auto& get_intercepts()
    {
        static std::atomic<unsigned long> _instance{ 0 };
        return _instance;
    }

    static auto& puts_intercepts()
    {
        static std::atomic<unsigned long> _instance{ 0 };
        return _instance;
    }

    TIMEMORY_NOINLINE void operator()(const gotcha_data&, void (*_func)(const char*),
                                      const char* msg) const
    {
        auto _n = ++puts_intercepts();
        if(_n < 10)
            (*_func)(msg);
    }

    TIMEMORY_NOINLINE double operator()(double (*_func)(double), double val) const
    {
        ++get_intercepts();
        return (*_func)(val);
    }
};
}  // namespace component
}  // namespace tim

using cmath_reentry_t = gotcha<4, std::tuple<>, cmath_reentry>;

TIMEMORY_DEFINE_CONCRETE_TRAIT(static_data, cmath_reentry_t, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(prevent_reentry, cmath_reentry_t, false_type)

TEST_F(gotcha_tests, unprotected_replacement)
{
    auto _dbg = tim::settings::debug();
    // tim::settings::debug() = true;

    using exp_bundle_t = tim::component_bundle<TIMEMORY_API, cmath_reentry_t>;

    static_assert(cmath_reentry_t::components_size == 0,
                  "cmath_reentry_t should have no components");
    static_assert(cmath_reentry_t::replaces, "cmath_reentry_t does not replace exp!");

    //
    // configure the initializer for the gotcha component which replaces exp
    //
    cmath_reentry_t::get_default_ready() = true;
    cmath_reentry_t::get_initializer()   = []() {
        puts("Generating exp intercept...");
#if defined(TIMEMORY_HIPCC)
        cmath_reentry_t::template instrument<0, double, double>::generate("exp");
#else
        TIMEMORY_C_GOTCHA(cmath_reentry_t, 0, exp);
#endif
        // TIMEMORY_C_GOTCHA(cmath_reentry_t, 1, test_exp);
        TIMEMORY_CXX_GOTCHA(cmath_reentry_t, 2, ext::do_puts);
#if defined(TIMEMORY_HIPCC)
        cmath_reentry_t::template instrument<0, double, double>::generate("__exp_finite");
#else
        TIMEMORY_DERIVED_GOTCHA(cmath_reentry_t, 3, exp, "__exp_finite");
#endif
    };

    auto _func = [](bool _first) {
        if(!_first)
            cmath_reentry_t::enable();

        auto inp = std::min<double>(10.0, exp(5.0));
        {
            auto _configured = cmath_reentry_t::is_configured();
            if(_first)
            {
                EXPECT_FALSE(_configured);
            }
            else
            {
                EXPECT_TRUE(_configured);
            }
        }
        auto _handle = exp_bundle_t(details::get_test_name());
        {
            auto _info = cmath_reentry_t::get_info();
            EXPECT_GE(_info[0], 0) << "# ready";
            EXPECT_GE(_info[1], (_first) ? 0 : 2) << "# filled";
            EXPECT_GE(_info[2], 0) << "# is_active";
            EXPECT_EQ(_info[3], 0) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        _handle.start();
        {
            auto _configured = cmath_reentry_t::is_configured();
            EXPECT_TRUE(_configured);
            auto _info = cmath_reentry_t::get_info();
            EXPECT_GE(_info[0], 2) << "# ready";
            EXPECT_GE(_info[1], 2) << "# filled";
            EXPECT_GE(_info[2], 2) << "# is_active";
            EXPECT_EQ(_info[3], 0) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        volatile double ret = inp;
        for(int i = 0; i < 10; ++i)
        {
            auto val = ret / ::pow(ret, 2);
            ret += test_exp(val);
            ext::do_puts(TIMEMORY_JOIN("", "computing exp(", val, ")").c_str());
        }
        {
            auto _info = cmath_reentry_t::get_info();
            EXPECT_GE(_info[0], 2) << "# ready";
            EXPECT_GE(_info[1], 2) << "# filled";
            EXPECT_GE(_info[2], 2) << "# is_active";
            EXPECT_EQ(_info[3], 0) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        _handle.stop();
        {
            auto _info = cmath_reentry_t::get_info();
            EXPECT_GE(_info[0], 0) << "# ready";
            EXPECT_GE(_info[1], 2) << "# filled";
            EXPECT_GE(_info[2], 0) << "# is_active";
            EXPECT_EQ(_info[3], 0) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        cmath_reentry_t::disable();
        {
            auto _configured = cmath_reentry_t::is_configured();
            EXPECT_FALSE(_configured);
            auto _info = cmath_reentry_t::get_info();
            EXPECT_GE(_info[0], 0) << "# ready";
            EXPECT_GE(_info[1], 2) << "# filled";
            EXPECT_GE(_info[2], 0) << "# is_active";
            EXPECT_GE(_info[3], 2) << "# is_finalized";
            EXPECT_EQ(_info[4], 0) << "# suppression";
        }
        if(_first)
            printf("result: %f\n", ret);
        EXPECT_GT(ret, 10.9) << "result: " << ret;
    };

    auto _parallel_func = [_func](size_t _beg, size_t _end) {
        for(size_t j = _beg; j < _end; ++j)
        {
            printf("Starting iteration %zu...\n", j);
            _func(j == 0);
        }
    };

    std::thread{ _parallel_func, 0, exp_nitr / 2 }.join();
    _parallel_func(exp_nitr / 2, exp_nitr);

    printf("number of exp  intercepts: %lu\n", cmath_intercept::get_intercepts().load());
    printf("number of puts intercepts: %lu\n", cmath_intercept::puts_intercepts().load());

    EXPECT_EQ(cmath_reentry::get_intercepts(), 10 * exp_nitr)
        << "number of exp intercepts: " << cmath_reentry::get_intercepts();
    EXPECT_EQ(cmath_reentry::puts_intercepts(), 10 * exp_nitr)
        << "number of puts intercepts: " << cmath_reentry::puts_intercepts();

    tim::settings::debug() = _dbg;
}

//======================================================================================//

TEST_F(gotcha_tests, member_functions)
{
    using pair_type     = std::pair<float, double>;
    using bundle_type   = tim::auto_tuple<memfun_gotcha_t>;
    auto real_storage   = tim::storage<wall_clock>::instance();
    auto real_init_size = real_storage->size();
    printf("[initial]> wall-clock storage size: %li\n", (long int) real_init_size);

    memfun_gotcha_t::get_default_ready() = true;
    memfun_gotcha_t::get_initializer()   = [=]() {
        TIMEMORY_PRINT_HERE("%s", details::get_test_name().c_str());

        {
            using func_t = decltype(&DoWork::get);
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::get));

            TIMEMORY_CXX_GOTCHA(memfun_gotcha_t, 0, &DoWork::get);
        }
        {
            using func_t = decltype(&DoWork::execute_fp4);
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::execute_fp4));

            TIMEMORY_CXX_GOTCHA_MEMFUN(memfun_gotcha_t, 1, DoWork::execute_fp4);
        }
        {
            using func_t = decltype(&DoWork::execute_fp8);
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::execute_fp8));

            TIMEMORY_CXX_GOTCHA(memfun_gotcha_t, 2, &DoWork::execute_fp8);
        }
        {
            using func_t = decltype(&DoWork::execute_fp);
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::execute_fp));

            TIMEMORY_CXX_GOTCHA(memfun_gotcha_t, 3, &DoWork::execute_fp);
        }
    };

    float  fsum = 0.0;
    double dsum = 0.0;
    {
        TIMEMORY_BLANK_POINTER(gotcha_hybrid_t, details::get_test_name());

        DoWork dw(pair_type(0.25, 0.5));

        decltype(nitr) _ndiv = 1000;
        auto           _nitr = nitr / _ndiv;
        int64_t        ntot  = 0;
        for(int i = 0; i < _nitr; i += _ndiv)
        {
            TIMEMORY_BLANK_POINTER(bundle_type, details::get_test_name());
            ntot += _ndiv;
            if(i >= (_nitr - _ndiv))
            {
                for(int j = 0; j < _ndiv; ++j)
                {
                    dw.execute_fp4(10);
                    dw.execute_fp8(10);
                    auto ret = dw.get();
                    fsum += std::get<0>(ret);
                    dsum += std::get<1>(ret);
                }
            }
            else
            {
                auto _fp4 = [&]() {
                    for(int j = 0; j < _ndiv; ++j)
                    {
                        dw.execute_fp4(10);
                        auto ret = dw.get();
                        fsum += std::get<0>(ret);
                    }
                };

                auto _fp8 = [&]() {
                    for(int j = 0; j < _ndiv; ++j)
                    {
                        dw.execute_fp8(10);
                        auto ret = dw.get();
                        dsum += std::get<1>(ret);
                    }
                };

                std::thread t4(_fp4);
                std::thread t8(_fp8);

                t4.join();
                t8.join();
            }
        }

        int rank = tim::dmp::rank();
        if(rank == 0)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank,
                   static_cast<double>(fsum));
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }

        auto _mfg = memfun_gotcha_t{};
        _mfg.start();
        float  fsum2 = 0.0;
        double dsum2 = 0.0;
        for(int64_t i = 0; i < ntot; ++i)
        {
            dw.execute_fp(10, { 0.25 }, { 0.5 });
            auto ret = dw.get();
            fsum2 += std::get<0>(ret);
            dsum2 += std::get<1>(ret);
        }
        _mfg.stop();

        if(rank == 0)
        {
            printf("\n");
            printf("[%i]> single-precision sum2 = %8.2f\n", rank,
                   static_cast<double>(fsum2));
            printf("[%i]> double-precision sum2 = %8.2f\n", rank, dsum2);
        }

        EXPECT_NEAR(fsum2, fsum, tolerance);
        EXPECT_NEAR(dsum2, dsum, tolerance);
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank,
                   static_cast<double>(fsum));
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    auto real_data       = real_storage->get();
    auto real_final_size = real_storage->size();
    printf("[final]> wall-clock storage size: %li\n", (long int) real_final_size);

    EXPECT_NEAR(fsum, -1818.20, tolerance);
    EXPECT_NEAR(dsum, -858.72, tolerance);
    EXPECT_EQ(real_final_size, 5 + real_init_size);
    {
        auto itr = real_data.begin();
        std::advance(itr, real_init_size);
        EXPECT_EQ(itr->data().get_laps(), 1) << itr->prefix();
    }
    {
        auto itr = real_data.begin();
        std::advance(itr, real_init_size + 1);
        EXPECT_GE(itr->data().get_laps(), 1000) << itr->prefix();
        ++itr;
        EXPECT_GE(itr->data().get_laps(), 1000) << itr->prefix();
        ++itr;
        EXPECT_GE(itr->data().get_laps(), 1000) << itr->prefix();
        ++itr;
        EXPECT_GE(itr->data().get_laps(), 1000) << itr->prefix();
    }
    {
        auto itr = real_data.begin();
        std::advance(itr, real_init_size + 1);
        EXPECT_LE(itr->data().get_laps(), 2000) << itr->prefix();
        ++itr;
        EXPECT_LE(itr->data().get_laps(), 2000) << itr->prefix();
        ++itr;
        EXPECT_LE(itr->data().get_laps(), 2000) << itr->prefix();
        ++itr;
        EXPECT_LE(itr->data().get_laps(), 2000) << itr->prefix();
    }
    {
        int64_t _nlaps = 0;
        auto    bitr   = real_data.begin() + real_init_size;
        auto    eitr   = bitr + 5;
        for(auto itr = bitr; itr != real_data.end() && itr != eitr; ++itr)
            _nlaps += itr->data().get_laps();
        EXPECT_EQ(_nlaps, 5001);
    }
}

//======================================================================================//

#if defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_GOTCHA)

#    include "timemory/components/gotcha/mpip.hpp"

TEST_F(gotcha_tests, mpip)
{
    using api_t         = TIMEMORY_API;
    using mpi_toolset_t = tim::lightweight_tuple<trip_count>;
    using mpip_tuple_t =
        tim::component_tuple<trip_count, mpip_handle<mpi_toolset_t, api_t>>;

    configure_mpip<mpi_toolset_t, api_t>();
    auto ret = activate_mpip<mpi_toolset_t, api_t>();

    {
        TIMEMORY_BLANK_POINTER(mpip_tuple_t, details::get_test_name());

        float  fsum = 0.0;
        double dsum = 0.0;
        for(int i = 0; i < nitr; ++i)
        {
            auto fsendbuf = details::generate<float>(1000);
            auto frecvbuf = details::allreduce(fsendbuf);
            fsum += static_cast<float>(
                std::accumulate(frecvbuf.begin(), frecvbuf.end(), 0.0));

            auto dsendbuf = details::generate<double>(1000);
            auto drecvbuf = details::allreduce(dsendbuf);
            dsum += std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);

            details::cleanup();
        }

        auto rank = tim::mpi::rank();
        auto size = tim::mpi::size();
        for(int i = 0; i < size; ++i)
        {
            tim::mpi::barrier();
            if(i == rank)
            {
                printf("\n");
                printf("[%i]> single-precision sum = %8.2f\n", rank,
                       static_cast<double>(fsum));
                printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
            }
            tim::mpi::barrier();
        }
        tim::mpi::barrier();
        if(rank == 0)
            printf("\n");

        EXPECT_NEAR(fsum, 49892284.00 * size, tolerance);
        EXPECT_NEAR(dsum, 49868704.48 * size, tolerance);
    }

    deactivate_mpip<mpi_toolset_t, api_t>(ret);
}

#endif

//======================================================================================//

TIMEMORY_INITIALIZE_STORAGE(mpi_gotcha_t, work_gotcha_t, memfun_gotcha_t, malloc_gotcha_t,
                            cmath_intercept, cmath_intercept_t, cmath_reentry,
                            cmath_reentry_t, wall_clock, cpu_clock, peak_rss)
