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

#include "ex_optional.hpp"  // file that includes optional usage

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>

//--------------------------------------------------------------------------------------//
//
//      Declarations
//
//--------------------------------------------------------------------------------------//

template <typename Tp> using vector_t = std::vector<Tp>;

static bool quiet = false;
namespace impl
{
long fibonacci(long n);
}
long                fibonacci(long n);
void                status();
void                write(long, long);
void                allreduce(const vector_t<long>& send, vector_t<long>& recv);
void                scatter_gather(int num_elements_per_proc);
std::vector<double> create_rand_nums(int num_elements);
double              compute_avg(const std::vector<double>& array);

//--------------------------------------------------------------------------------------//
//
//      Construct a main that uses macro tricks to avoid:
//
//      #ifdef USE_TIMEMORY
//          ....
//      #endif
//
//--------------------------------------------------------------------------------------//

int main(int argc, char** argv)
{
    // setenv when available
#if(_POSIX_C_SOURCE >= 200112L) || defined(_BSD_SOURCE) || defined(TIMEMORY_UNIX)
    setenv("TIMEMORY_TIMING_UNITS", "sec", 0);
    setenv("TIMEMORY_MEMORY_UNITS", "mb", 0);
    setenv("TIMEMORY_TIMING_WIDTH", "14", 0);
    setenv("TIMEMORY_MEMORY_WIDTH", "12", 0);
    setenv("TIMEMORY_TIMING_PRECISION", "6", 0);
    setenv("TIMEMORY_MEMORY_PRECISION", "3", 0);
    setenv("TIMEMORY_TIMING_SCIENTIFIC", "OFF", 0);
    setenv("TIMEMORY_MEMORY_SCIENTIFIC", "OFF", 0);
#endif

    //
    //  Dummy functions when USE_TIMEMORY not defined
    //
    tim::mpi::initialize(argc, argv);
    status();
    tim::timemory_init(argc, argv);
    //
    //  Provide some work
    //
    std::vector<long> fibvalues;
    for(int i = 1; i < argc; ++i) fibvalues.push_back(atol(argv[i]));
    if(fibvalues.empty())
    {
        fibvalues.resize(44);
        long n = 0;
        std::generate_n(fibvalues.data(), fibvalues.size(), [&]() { return ++n; });
    }

    //
    // create an auto tuple accessible via a caliper integer or expand to nothing
    //
    auto main = TIMEMORY_HANDLE(auto_hybrid_t, "");
    TIMEMORY_CALIPER(0, auto_hybrid_t, "");

    //
    // call <auto_tuple_t>.report_at_exit(true) or expand to nothing
    //
    main.report_at_exit(true);
    TIMEMORY_CALIPER_APPLY(0, report_at_exit, true);

    //
    //  Execute the work
    //
    long ret_sum = 0;
#pragma omp parallel for reduction(+ : ret_sum)
    for(int64_t i = 0; i < (int64_t) fibvalues.size(); ++i)
    {
        auto itr = fibvalues.at(i);
        auto ret = fibonacci(itr);
        ret_sum += ret * ret;
    }

    // avoid set but unused warning
    if(ret_sum < 0) printf("sum: %li\n", ret_sum);

    std::vector<long> ret_reduce;
    std::vector<long> ret_send;
    for(size_t i = 0; i < fibvalues.size(); ++i)
        ret_send.push_back(fibonacci(fibvalues.at(i) % 20 + 10));

    allreduce(ret_send, ret_reduce);

    quiet       = true;
    auto modulo = fibonacci(25);

    for(size_t i = 0; i < fibvalues.size(); ++i) write(fibvalues.at(i), ret_reduce.at(i));
    for(size_t i = 0; i < fibvalues.size(); ++i)
        scatter_gather(fibonacci(fibvalues.at(i)) % modulo + 200);

    //
    // call <auto_tuple_t>.stop() or expand to nothing
    //
    TIMEMORY_CALIPER_APPLY(0, stop);

    //
    // sleep for 1 second so difference between two calipers
    //
    std::this_thread::sleep_for(std::chrono::seconds(1));

    //
    // call <auto_tuple_t>.stop() or expand to nothing
    //
    main.stop();

    status();
    tim::timemory_finalize();
    tim::mpi::finalize();

    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//
//
//      Implementations
//
//--------------------------------------------------------------------------------------//

long impl::fibonacci(long n)
{
    return (n < 2) ? n : (impl::fibonacci(n - 1) + impl::fibonacci(n - 2));
}

long fibonacci(long n)
{
    TIMEMORY_BASIC_MARKER(auto_hybrid_t, "(", n, ")");
    auto ret = impl::fibonacci(n);
    write(n, ret);
    return ret;
}

void write(long nfib, long answer)
{
    if(quiet) return;
#pragma omp critical
    printf("fibonacci(%li) = %li\n", nfib, answer);
}

void status()
{
#if defined(USE_TIMEMORY)
    printf("\n#----------------- TIMEMORY is enabled  ----------------#\n\n");
#else
    printf("\n#----------------- TIMEMORY is disabled ----------------#\n\n");
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

//--------------------------------------------------------------------------------------//

void allreduce(const vector_t<long>& sendbuf, vector_t<long>& recvbuf)
{
    recvbuf.resize(sendbuf.size(), 0L);
#if defined(TIMEMORY_USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce((long*) sendbuf.data(), recvbuf.data(), sendbuf.size(), MPI_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#else
    std::copy(sendbuf.begin(), sendbuf.end(), recvbuf.begin());
#endif
    for(auto& itr : recvbuf) itr = static_cast<long>(log(itr));
}

//--------------------------------------------------------------------------------------//

void scatter_gather(int num_elements_per_proc)
{
    if(num_elements_per_proc == 0) return;

    TIMEMORY_BLANK_MARKER(auto_hybrid_t, "scatter_gatther_", num_elements_per_proc);
    (void) num_elements_per_proc;
    // printf("Number of elements per process: %i\n", num_elements_per_proc);

#if defined(TIMEMORY_USE_MPI)
    auto n = num_elements_per_proc;

    // Seed the random number generator to get different results each time
    srand(time(NULL));

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // decide who is the master rank
    int master_rank = rand() % world_size;
    if(num_elements_per_proc % 2 == 0)
    {
        int root = 0;
        if(world_rank == root)
        {
            // If we are the root process, send our data to everyone
            for(int i = 0; i < world_size; i++)
            {
                if(i != world_rank)
                    MPI_Send(&master_rank, 1, MPI_INT, i, n % world_size, MPI_COMM_WORLD);
            }
        } else
        {
            // If we are a receiver process, receive the data from the root
            MPI_Recv(&master_rank, 1, MPI_INT, root, n % world_size, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    } else
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&master_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if(world_rank == master_rank)
        printf("Master rank: %i, Number of elements per process: %i\n", master_rank,
               num_elements_per_proc);

    // Create a random array of elements on the root process. Its total
    // size will be the number of elements per process times the number
    // of processes
    std::vector<double> rand_nums;
    if(world_rank == master_rank)
        rand_nums = create_rand_nums(num_elements_per_proc * world_size);

    // For each process, create a buffer that will hold a subset of the entire array
    std::vector<double> sub_rand_nums(num_elements_per_proc, 0.0);

    // Scatter the random numbers from the root process to all processes in the MPI world
    MPI_Scatter(rand_nums.data(), num_elements_per_proc, MPI_DOUBLE, sub_rand_nums.data(),
                num_elements_per_proc, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

    // Compute the average of your subset
    double sub_avg = compute_avg(sub_rand_nums);

    // Gather all partial averages down to the root process
    std::vector<double> sub_avgs;
    if(world_rank == master_rank) sub_avgs.resize(world_size, 0.0);

    MPI_Gather(&sub_avg, 1, MPI_DOUBLE, sub_avgs.data(), 1, MPI_DOUBLE, master_rank,
               MPI_COMM_WORLD);

    // Now that we have all of the partial averages on the root, compute the
    // total average of all numbers. Since we are assuming each process computed
    // an average across an equal amount of elements, this computation will
    // produce the correct answer.
    if(world_rank == master_rank)
    {
        double avg               = compute_avg(sub_avgs);
        double original_data_avg = compute_avg(rand_nums);
        printf("Avg of all elements is %10.8f, Avg computed across original data is "
               "%10.8f\n",
               avg, original_data_avg);
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

//--------------------------------------------------------------------------------------//

// Creates an array of random numbers. Each number has a value from 0 - 1
std::vector<double> create_rand_nums(int num_elements)
{
    std::vector<double> rand_nums(num_elements, 0.0);
    for(auto& itr : rand_nums) itr = (rand() / (double) RAND_MAX);
    return rand_nums;
}

//--------------------------------------------------------------------------------------//

// Computes the average of an array of numbers
double compute_avg(const std::vector<double>& array)
{
    return std::accumulate(array.begin(), array.end(), 0.0) / array.size();
}

//--------------------------------------------------------------------------------------//
