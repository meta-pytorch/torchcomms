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

#include "pybind11/cast.h"
#include "pybind11/chrono.h"
#include "pybind11/eval.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(USE_MPI) || defined(TIMEMORY_USE_MPI)
#    include <mpi.h>
#endif

#if defined(_OPENMP)
#    include <omp.h>
#endif

namespace py = pybind11;

template <typename Tp>
using vector_t = std::vector<Tp>;

static std::mt19937 rng;

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline vector_t<Tp>
generate(const int64_t& nsize)
{
    std::vector<Tp> sendbuf(nsize, 0.0);
    std::mt19937    rng;
    rng.seed(54561434UL);
    auto dist = [&]() { return std::generate_canonical<Tp, 10>(rng); };
    std::generate(sendbuf.begin(), sendbuf.end(), [&]() { return dist(); });
    return sendbuf;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline vector_t<Tp>
allreduce(const vector_t<Tp>& sendbuf)
{
    vector_t<Tp> recvbuf(sendbuf.size(), 0.0);
#if defined(USE_MPI) || defined(TIMEMORY_USE_MPI)
    auto dtype = (std::is_same<Tp, float>::value) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), dtype, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    throw std::runtime_error("No MPI support!");
#endif
    return recvbuf;
}

//--------------------------------------------------------------------------------------//

double
run(int nitr, int nsize);

//--------------------------------------------------------------------------------------//

void
scatter_gather(int num_elements_per_proc);

//--------------------------------------------------------------------------------------//

std::vector<double>
create_rand_nums(int num_elements);

//--------------------------------------------------------------------------------------//

double
compute_avg(const std::vector<double>& array);

//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(libex_python_bindings, ex)
{
    auto _run = [](int nitr, int nsize) {
        auto value = 0.0;
        try
        {
            py::gil_scoped_release release;
#if defined(_OPENMP)
            int nrank = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &nrank);
            omp_set_num_threads(std::thread::hardware_concurrency() / nrank);
#endif
            value = run(nitr, nsize);
        } catch(std::exception& e)
        {
            fprintf(stderr, "Error! %s\n", e.what());
            throw;
        }

        scatter_gather(nitr * nsize);

        return value;
    };

    ex.def("run", _run, "Run a calculation", py::arg("nitr") = 10,
           py::arg("nsize") = 5000);
}

//--------------------------------------------------------------------------------------//

double
run(int nitr, int nsize)
{
    rng.seed(54561434UL);

    printf("[%s] Running MPI algorithm with %i iterations and %i entries...\n", __func__,
           nitr, nsize);

    double dsum = 0.0;
#pragma omp parallel for
    for(int i = 0; i < nitr; ++i)
    {
        auto dsendbuf = generate<double>(nsize);
        auto drecvbuf = allreduce(dsendbuf);
        auto dtmp     = std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);
#pragma omp atomic
        dsum += dtmp;
    }
    return dsum;
}

void
scatter_gather(int num_elements_per_proc)
{
    if(num_elements_per_proc == 0)
        return;

#if defined(USE_MPI) || defined(TIMEMORY_USE_MPI)
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
        }
        else
        {
            // If we are a receiver process, receive the data from the root
            MPI_Recv(&master_rank, 1, MPI_INT, root, n % world_size, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }
    else
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
    if(world_rank == master_rank)
        sub_avgs.resize(world_size, 0.0);

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
#else
    (void) compute_avg(create_rand_nums(num_elements_per_proc));
#endif
}

// Creates an array of random numbers. Each number has a value from 0 - 1
std::vector<double>
create_rand_nums(int num_elements)
{
    std::vector<double> rand_nums(num_elements, 0.0);
    for(auto& itr : rand_nums)
        itr = (rand() / (double) RAND_MAX);
    return rand_nums;
}

// Computes the average of an array of numbers
double
compute_avg(const std::vector<double>& array)
{
    return std::accumulate(array.begin(), array.end(), 0.0) / array.size();
}
