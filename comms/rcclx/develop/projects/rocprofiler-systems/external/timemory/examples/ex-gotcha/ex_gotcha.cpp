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

#if defined(DEBUG) && !defined(VERBOSE)
#    define VERBOSE
#endif

#include "ex_gotcha_lib.hpp"
#include "timemory/timemory.hpp"

#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

//======================================================================================//

using namespace tim;
using namespace tim::component;

constexpr size_t N     = 10;
using general_bundle_t = user_bundle<0>;
using tool_tuple_t     = tim::component_tuple<general_bundle_t>;
using put_gotcha_t     = tim::component::gotcha<N, tool_tuple_t, char>;
using mpi_gotcha_t     = tim::component::gotcha<N, tool_tuple_t, double>;
using fake_gotcha_t    = tim::component::gotcha<N, tim::component_tuple<>, float>;
using gotcha_tuple_t   = tim::auto_tuple_t<tool_tuple_t, user_global_bundle>;

#if !defined(TIMEMORY_USE_MPI)
namespace tim
{
namespace trait
{
template <>
struct is_available<mpi_gotcha_t> : std::false_type
{
};
}  // namespace trait
}  // namespace tim
#endif

//======================================================================================//

void
init()
{
    put_gotcha_t::get_initializer() = []() {
        put_gotcha_t::configure<0, int, const char*>("puts");
    };

#if defined(TIMEMORY_USE_MPI)
    mpi_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 0, MPI_Barrier);
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 1, MPI_Bcast);
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 2, MPI_Scan);
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 3, MPI_Allreduce);
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 4, MPI_Reduce);
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 5, MPI_Alltoall);
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 6, MPI_Allgather);
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 7, MPI_Gather);
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 8, MPI_Scatter);
    };
#endif

    printf("put gotcha is available: %s\n",
           trait::as_string<trait::is_available<put_gotcha_t>>().c_str());
    printf("mpi gotcha is available: %s\n",
           trait::as_string<trait::is_available<mpi_gotcha_t>>().c_str());
    printf("\n");

    // configure the bundles
    general_bundle_t::configure<wall_clock, cpu_clock, peak_rss>();
    user_global_bundle::configure<fake_gotcha_t>();
    if(tim::get_env("MPI_INTERCEPT", true)) user_global_bundle::configure<mpi_gotcha_t>();
    if(tim::get_env("PUT_INTERCEPT", true)) user_global_bundle::configure<put_gotcha_t>();
}

//======================================================================================//

int
main(int argc, char** argv)
{
    settings::width()        = 12;
    settings::precision()    = 6;
    settings::timing_units() = "msec";
    settings::memory_units() = "kB";
    settings::verbose()      = 1;
    settings::mpi_init()     = true;
    settings::mpi_finalize() = true;
    settings::mpi_thread()   = false;

    tim::timemory_init(&argc, &argv);

    init();

    int rank = tim::dmp::rank();
    int size = tim::dmp::size();

    printf("size = %i\n", (int) size);
    printf("rank = %i\n", (int) rank);

    auto _exec = [&]() {
#if defined(VERBOSE)
        printf("[%i]> BEGIN SCOPED GOTCHA....\n", rank);
#endif

        TIMEMORY_BLANK_MARKER(gotcha_tuple_t, argv[0]);

        puts("Testing puts gotcha wraper...");

        tim::dmp::barrier();

        rank = tim::dmp::rank();
        size = tim::dmp::size();

        dmp::barrier();

        int nitr = 15;
        if(argc > 1) nitr = atoi(argv[1]);

        auto _exp = ext::do_exp_work(nitr);
        printf("\n");
        printf("[iterations=%i]>      single-precision exp = %f\n", nitr,
               std::get<0>(_exp));
        printf("[iterations=%i]>      double-precision exp = %f\n", nitr,
               std::get<1>(_exp));
        printf("\n");

#if defined(TIMEMORY_USE_MPI)
        int nsize = 1000;
        if(argc > 2) nsize = atoi(argv[2]);

        std::vector<double> recvbuf(nsize, 0.0);
        std::vector<double> sendbuf(nsize, 0.0);
        std::mt19937        rng;
        rng.seed((rank + 1) * std::random_device()());
        auto dist = [&]() { return std::generate_canonical<double, 10>(rng); };
        std::generate(sendbuf.begin(), sendbuf.end(), [&]() { return dist(); });

        MPI_Allreduce(sendbuf.data(), recvbuf.data(), nsize, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        double sum = std::accumulate(recvbuf.begin(), recvbuf.end(), 0.0);
        for(int i = 0; i < size; ++i) printf("[%i]> sum = %8.2f\n", rank, sum);
#endif

#if defined(VERBOSE)
        printf("[%i]> END SCOPED GOTCHA....\n", rank);
#endif
    };

    for(auto i = 0; i < 10; ++i) _exec();

    // MPI_Barrier needs to be disabled before finalization
    mpi_gotcha_t::disable();
    tim::timemory_finalize();

    return 0;
}

//======================================================================================//
