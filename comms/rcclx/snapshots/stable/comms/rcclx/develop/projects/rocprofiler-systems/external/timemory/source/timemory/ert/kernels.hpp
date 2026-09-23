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

/** \file timemory/ert/kernels.hpp
 * \headerfile timemory/ert/kernels.hpp "timemory/ert/kernels.hpp"
 * Provides kernels for executing kernels in ERT
 *
 */

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/backends/gpu.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/components/cuda/backends.hpp"
#include "timemory/ert/counter.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/macros.hpp"

#include <algorithm>  // std::max
#include <cstdint>
#include <functional>   // std::ref
#include <mutex>        // std::mutex, std::unique_lock
#include <thread>       // std::thread
#include <type_traits>  // std::is_same
#include <utility>      // std::forward
#include <vector>

namespace tim
{
namespace ert
{
//--------------------------------------------------------------------------------------//
//
//      CPU -- multiple trial
//
//--------------------------------------------------------------------------------------//

template <size_t Nrep, typename DeviceT, typename Intp, typename Tp, typename OpsFuncT,
          typename StoreFuncT, device::enable_if_cpu_t<DeviceT> = 0>
void
ops_kernel(Intp ntrials, Intp nsize, Tp* A, OpsFuncT&& ops_func, StoreFuncT&& store_func)
{
    // divide by two here because macros halve, e.g. ERT_FLOP == 4 means 2 calls
    constexpr size_t NUM_REP = Nrep / 2;
    constexpr size_t MOD_REP = Nrep % 2;
    auto             range   = device::grid_strided_range<DeviceT, 0, Intp>(nsize);

    Tp alpha = static_cast<Tp>(0.5);
    for(Intp j = 0; j < ntrials; ++j)
    {
        for(auto i = range.begin(); i < range.end(); i += range.stride())
        {
            Tp beta = static_cast<Tp>(0.8);
            mpl::apply<void>::unroll<NUM_REP + MOD_REP, DeviceT>(ops_func, beta, A[i],
                                                                 alpha);
            store_func(A[i], beta);
        }
        alpha *= static_cast<Tp>(1.0 - 1.0e-8);
    }
}

//--------------------------------------------------------------------------------------//
//
//      GPU -- multiple trial
//
//--------------------------------------------------------------------------------------//

template <size_t Nrep, typename DeviceT, typename Intp, typename Tp, typename OpsFuncT,
          typename StoreFuncT, device::enable_if_gpu_t<DeviceT> = 0,
          enable_if_t<!std::is_same<Tp, gpu::fp16_t>::value> = 0>
TIMEMORY_GLOBAL_FUNCTION void
ops_kernel(Intp ntrials, Intp nsize, Tp* A, OpsFuncT&& ops_func, StoreFuncT&& store_func)
{
    // divide by two here because macros halve, e.g. ERT_FLOP == 4 means 2 calls
    constexpr size_t NUM_REP = Nrep / 2;
    constexpr size_t MOD_REP = Nrep % 2;
    auto             range   = device::grid_strided_range<DeviceT, 0, Intp>(nsize);

    Tp alpha = static_cast<Tp>(0.5);
    for(Intp j = 0; j < ntrials; ++j)
    {
        for(auto i = range.begin(); i < range.end(); i += range.stride())
        {
            Tp beta = static_cast<Tp>(0.8);
            mpl::apply<void>::unroll<NUM_REP + MOD_REP, DeviceT>(ops_func, beta, A[i],
                                                                 alpha);
            store_func(A[i], beta);
        }
        alpha *= static_cast<Tp>(1.0 - 1.0e-8);
    }
}

//--------------------------------------------------------------------------------------//
//
//      GPU -- multiple trial -- packed (2) half-precision
//
//--------------------------------------------------------------------------------------//

template <size_t Nrep, typename DeviceT, typename Intp, typename Tp, typename OpsFuncT,
          typename StoreFuncT, device::enable_if_gpu_t<DeviceT> = 0,
          enable_if_t<std::is_same<Tp, gpu::fp16_t>::value> = 0>
TIMEMORY_GLOBAL_FUNCTION void
ops_kernel(Intp ntrials, Intp nsize, Tp* A, OpsFuncT&& ops_func, StoreFuncT&& store_func)
{
    // divide by four instead of two here because fp16_t is a packed operation
    constexpr size_t NUM_REP = Nrep / 4;
    constexpr size_t MOD_REP = Nrep % 4;
    auto             range   = device::grid_strided_range<DeviceT, 0, int32_t>(nsize);

    Tp alpha = { 0.5, 0.5 };
    for(int32_t j = 0; j < ntrials; ++j)
    {
        for(auto i = range.begin(); i < range.end(); i += range.stride())
        {
            Tp beta = { 0.8, 0.8 };
            mpl::apply<void>::unroll<NUM_REP + MOD_REP, DeviceT>(ops_func, beta, A[i],
                                                                 alpha);
            store_func(A[i], beta);
        }
        alpha *= { 1.0 - 1.0e-8, 1.0 - 1.0e-8 };
    }
}

//--------------------------------------------------------------------------------------//
///
///     This is the "main" function for ERT
///
template <size_t Nops, size_t... Nextra, typename DeviceT, typename Tp, typename CounterT,
          typename OpsFuncT, typename StoreFuncT,
          enable_if_t<sizeof...(Nextra) == 0, int> = 0>
bool
ops_main(counter<DeviceT, Tp, CounterT>& _counter, OpsFuncT&& ops_func,
         StoreFuncT&& store_func) TIMEMORY_VISIBILITY("default");

template <size_t Nops, size_t... Nextra, typename DeviceT, typename Tp, typename CounterT,
          typename OpsFuncT, typename StoreFuncT,
          enable_if_t<sizeof...(Nextra) == 0, int>>
bool
ops_main(counter<DeviceT, Tp, CounterT>& _counter, OpsFuncT&& ops_func,
         StoreFuncT&& store_func)
{
    if(_counter.skip(Nops))
        return false;

    using stream_list_t   = std::vector<gpu::stream_t>;
    using thread_list_t   = std::vector<std::thread>;
    using device_params_t = device::params<DeviceT>;
    using Intp            = int32_t;
    using ull             = long long unsigned;

    constexpr bool is_gpu = std::is_same<DeviceT, device::gpu>::value;

    if(settings::verbose() > 0 || settings::debug())
        printf("[%s] Executing %li ops...\n", __FUNCTION__, (long int) Nops);

    if(_counter.bytes_per_element == 0)
    {
        fprintf(stderr, "[%s:%i]> bytes-per-element is not set!\n", __FUNCTION__,
                __LINE__);
    }

    if(_counter.memory_accesses_per_element == 0)
    {
        fprintf(stderr, "[%s:%i]> memory-accesses-per-element is not set!\n",
                __FUNCTION__, __LINE__);
    }

    // list of streams
    stream_list_t streams;
    // generate async streams if multiple streams were requested
    if(_counter.params.nstreams > 1)
    {
        // fill with implicit stream
        streams.resize(_counter.params.nstreams, 0);
        for(auto& itr : streams)
            gpu::stream_create(itr);
    }

    std::mutex _opmutex{};

    auto _opfunc = [&](uint64_t tid, thread_barrier* fbarrier, thread_barrier* lbarrier) {
        threading::set_thread_name(
            TIMEMORY_JOIN('.', "ert", DeviceT::name(), tid).c_str());
        threading::affinity::set();
        {
            std::unique_lock<std::mutex> _lk{ _opmutex };
            // execute the callback
            _counter.configure(tid);
        }
        // allocate buffer
        auto     buf = _counter.get_buffer();
        uint64_t n   = _counter.params.working_set_min;
        // cache this
        const uint64_t nstreams = std::max<uint64_t>(_counter.params.nstreams, 1);
        // create the launch parameters (ignored on CPU)
        //
        // if grid_size is zero (default), the launch command will calculate a grid-size
        // as follows:
        //
        //      grid_size = ((data_size + block_size - 1) / block_size)
        //
        device_params_t dev_params(_counter.params.grid_size, _counter.params.block_size,
                                   _counter.params.shmem_size, DeviceT::default_stream);
        //
        if(n > _counter.nsize)
        {
            fprintf(stderr,
                    "[%s@'%s':%i]> Warning! ERT not running any trials because working "
                    "set min > nsize: %llu > %llu\n",
                    TIMEMORY_ERROR_FUNCTION_MACRO, __FILE__, __LINE__, (ull) n,
                    (ull) _counter.nsize);
        }

        while(n <= _counter.nsize)
        {
            // working set - nsize
            uint64_t ntrials = _counter.nsize / n;
            if(ntrials < 1)
                ntrials = 1;

            if(settings::debug() && tid == 0)
            {
                printf("[tim::ert::ops_main<%llu>]> number of trials: %llu, n = %llu, "
                       "nsize "
                       "= %llu\n",
                       (ull) Nops, (ull) ntrials, (ull) n, (ull) _counter.nsize);
            }

            auto _itr_params = _counter.params;

            if(is_gpu)
            {
                // make sure all streams are synced
                for(auto& itr : streams)
                    gpu::stream_sync(itr);

                // sync the streams
                if(nstreams < 2)
                    gpu::device_sync();
            }

            // wait for all threads
            if(fbarrier)
            {
                fbarrier->wait();
                if(fbarrier->get_count() > 0)
                    throw std::runtime_error("Error! first barrier failed! " +
                                             std::to_string(fbarrier->get_count()));
            }

            // get instance of object measuring something during the calculation
            CounterT ct = _counter.get_counter();
            // start the timer or anything else being recorded
            ct.start();

            // only do this more complicated mess if we need to
            if(nstreams > 1)
            {
                auto nchunk  = n / nstreams;
                auto nmodulo = n % nstreams;
                for(uint64_t i = 0; i < nstreams; ++i)
                {
                    // calculate the size of the subchunk
                    int32_t _n      = nchunk + ((i + 1 == nstreams) ? nmodulo : 0);
                    auto    _params = dev_params;  // copy of the parameters
                    device::launch(
                        _n, streams.at(i % streams.size()), _params,
                        ops_kernel<Nops, DeviceT, Intp, Tp, OpsFuncT, StoreFuncT>,
                        ntrials, _n, buf + (i * nchunk), std::forward<OpsFuncT>(ops_func),
                        std::forward<StoreFuncT>(store_func));
                    _itr_params.grid_size =
                        (i == 0) ? _params.grid
                                 : std::max<int64_t>(_itr_params.grid_size, _params.grid);
                }
            }
            else
            {
                device::launch(n, dev_params,
                               ops_kernel<Nops, DeviceT, Intp, Tp, OpsFuncT, StoreFuncT>,
                               ntrials, n, buf, std::forward<OpsFuncT>(ops_func),
                               std::forward<StoreFuncT>(store_func));

                _itr_params.grid_size = dev_params.grid;
            }

            if(is_gpu)
            {
                for(auto& itr : streams)
                    gpu::stream_sync(itr);

                // sync the streams
                if(nstreams < 2)
                    gpu::device_sync();
            }

            // wait for all threads
            if(lbarrier)
            {
                lbarrier->wait();
                if(lbarrier->get_count() > 0)
                    throw std::runtime_error("Error! last barrier failed! " +
                                             std::to_string(lbarrier->get_count()));
            }

            // stop the timer or anything else being recorded
            ct.stop();

            // store the result
            if(tid == 0)
            {
                // ensure there is not a data race if more than one thread somehow
                // has a tid of 0
                std::unique_lock<std::mutex> _lk{ _opmutex };
                _counter.record(ct, n, ntrials, Nops, _itr_params);
            }

            n = ((1.1 * n) == n) ? (n + 1) : (1.1 * n);
        }

        if(is_gpu)
            gpu::device_sync();

        _counter.destroy_buffer(buf);
    };

    // guard against multiple threads trying to call ERT for some reason
    static std::mutex            _mtx;
    std::unique_lock<std::mutex> _lk{ _mtx };

    dmp::barrier();  // synchronize MPI processes

    if(is_gpu)
        gpu::device_sync();

    if(_counter.params.nthreads > 1)
    {
        // create synchronization barriers for the threads
        thread_barrier fbarrier{ _counter.params.nthreads };
        thread_barrier lbarrier{ _counter.params.nthreads };

        // list of threads
        thread_list_t threads{};
        // create the threads
        for(uint64_t i = 0; i < _counter.params.nthreads; ++i)
            threads.emplace_back(_opfunc, i, &fbarrier, &lbarrier);

        // wait for threads to finish
        for(auto& itr : threads)
            itr.join();
    }
    else
    {
        _opfunc(0, nullptr, nullptr);
    }

    if(is_gpu)
        gpu::device_sync();

    dmp::barrier();  // synchronize MPI processes

    // code was executed
    return true;
}

//--------------------------------------------------------------------------------------//
///
///     This is invokes the "main" function for ERT for all the desired "FLOPs" that
///     are unrolled in the kernel
///
template <size_t Nops, size_t... Nextra, typename DeviceT, typename Tp, typename CounterT,
          typename OpsFuncT, typename StoreFuncT,
          enable_if_t<(sizeof...(Nextra) > 0), int> = 0>
bool
ops_main(counter<DeviceT, Tp, CounterT>& _counter, OpsFuncT&& ops_func,
         StoreFuncT&& store_func)
{
    bool ret = false;
    // execute a single parameter
    ret |= ops_main<Nops>(std::ref(_counter).get(), ops_func, store_func);
    // continue the recursive loop
    ret |= ops_main<Nextra...>(std::ref(_counter).get(), ops_func, store_func);
    return ret;
}

//--------------------------------------------------------------------------------------//
///
///     This is invoked when TIMEMORY_USER_ERT_FLOPS is empty
///
template <size_t... Nops, typename DeviceT, typename Tp, typename CounterT,
          typename OpsFuncT, typename StoreFuncT,
          enable_if_t<sizeof...(Nops) == 0, int> = 0>
bool
ops_main(counter<DeviceT, Tp, CounterT>&, OpsFuncT&&, StoreFuncT&&)
{
    return false;
}

//--------------------------------------------------------------------------------------//

}  // namespace ert
}  // namespace tim
