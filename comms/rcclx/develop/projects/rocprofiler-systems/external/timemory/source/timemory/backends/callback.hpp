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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file callback.hpp
 * \headerfile callback.hpp "timemory/details/callback.hpp"
 * Provides implementation of callback for cudaEvent_t
 *
 */

#pragma once

#if defined(TIMEMORY_USE_CUDA)

#    include <condition_variable>
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <memory>
#    include <mutex>
#    include <thread>

namespace tim
{
namespace cuda
{
//--------------------------------------------------------------------------------------//
// Create thread
//
using callback_thread_t = std::unique_ptr<std::thread>;
struct callback_barrier_t
{
    std::mutex              mutex;
    std::condition_variable condvar;
    std::atomic<int>        releaseCount;
    std::atomic<int>        count;

    callback_barrier_t(int _releaseCount)
    : releaseCount(_releaseCount)
    {}

    callback_barrier_t(const callback_barrier_t&) = delete;
    callback_barrier_t(callback_barrier_t&&)      = default;

    callback_barrier_t& operator=(const callback_barrier_t&) = delete;
    callback_barrier_t& operator=(callback_barrier_t&&) = default;

    void increment()
    {
        if(++count >= releaseCount)
            condvar.notify_one();
    }

    void wait()
    {
        mutex.lock();
        while(count.load() < releaseCount.load())
        {
            condvar.wait(mutex);
        }
        mutex.unlock();
    }
};

//--------------------------------------------------------------------------------------//
//
//
inline callback_barrier_t&
callback_barrier()
{
    static callback_barrier_t _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//
//
//
inline callback_thread_t&
callback_thread()
{
    static callback_thread_t _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//
//
//
template <typename FuncT, typename... ArgsT>
inline callback_thread_t&
cutStartThread(FuncT&& func, ArgsT&&... args)
{
    callback_thread_t _instance = callback_thread_t(
        new std::thread(std::forward<FuncT>(func), std::forward<ArgsT>(args)...));
    return std::move(_instance);
}

//--------------------------------------------------------------------------------------//
// Wait for thread to finish
//
inline void
cutEndThread(callback_thread_t& _thread)
{
    _thread->join();
}

//--------------------------------------------------------------------------------------//
// Wait for multiple threads
//
inline void
cutWaitForThreads(std::list<callback_thread_t>& threads)
{
    for(auto& itr : threads)
        cutEndThread(itr);
}

struct heterogeneous_workload
{
    int          id;
    int          cudaDeviceID;
    cudaStream_t stream;
    cudaEvent_t* event;
    bool         success;
};

//--------------------------------------------------------------------------------------//
//
//
static void
postprocess(void* data)
{
    auto* workload = static_cast<heterogeneous_workload*>(data);
    // ... GPU is done with processing, continue on new CPU thread...

    // Select GPU for this CPU thread
    cudaSetDevice(workload->cudaDeviceID);

    // CPU thread consumes results from GPU
    workload->success = true;

    /*for (int i=0; i< N_workloads; ++i)
    {
        workload->success &= workload->h_data[i] == i + workload->id + 1;
    }*/

    // Signal the end of the heterogeneous workload to main thread
    callback_barrier().increment();
}

//--------------------------------------------------------------------------------------//
//
//
static void CUDART_CB
cutStreamCallback(cudaStream_t event, cudaError_t status, void* data)
{
    cutStartThread(postprocess, data);
}

}  // namespace cuda
}  // namespace tim

#endif
