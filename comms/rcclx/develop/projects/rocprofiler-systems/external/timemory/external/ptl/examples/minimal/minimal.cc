//
// MIT License
// Copyright (c) 2019 Jonathan R. Madsen
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
/// \file tasking.cc
/// \brief Example showing the usage of tasking

#include "PTL/PTL.hh"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using namespace PTL;

static std::mt19937 rng;

//============================================================================//

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
Tp
random_entry(const std::vector<Tp>& v)
{
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    AutoLock lk{ TypeMutex<decltype(rng)>() };
    return v.at(dist(rng));
}

//============================================================================//

int
main(int argc, char** argv)
{
    ConsumeParameters(argc, argv);
    rng.seed(std::random_device()());
    Threading::SetThreadId(0);
    Backtrace::Enable();

    auto hwthreads = std::thread::hardware_concurrency();
    auto nthreads  = GetEnv<decltype(hwthreads)>("NUM_THREADS", hwthreads);
    if(argc > 1)
        nthreads = std::stoul(argv[1]) % (hwthreads + 1);

    if(nthreads == 0)
        nthreads = 1;

    std::cout << "[" << argv[0] << "]> "
              << "Number of threads: " << nthreads << std::endl;
    Timer total_timer;
    total_timer.Start();

    // Construct the default run manager
    bool use_tbb     = GetEnv("PTL_USE_TBB", false);
    auto run_manager = TaskRunManager(use_tbb);
    run_manager.Initialize(nthreads);

    // the TaskManager is a utility that wraps the function calls into tasks for the
    // ThreadPool
    TaskManager*              task_manager = run_manager.GetTaskManager();
    auto*                     tp           = task_manager->thread_pool();
    std::set<std::thread::id> tids{};

    //------------------------------------------------------------------------//
    //                                                                        //
    //                  Execute function on all threads                       //
    //                                                                        //
    //------------------------------------------------------------------------//

    std::atomic<size_t> _all_exec{ 0 };
    tp->execute_on_all_threads([&tids, &_all_exec]() {
        ++_all_exec;
        std::stringstream ss;
        ss << "thread " << std::setw(4) << PTL::Threading::GetThreadId() << " executed\n";
        AutoLock lk{ TypeMutex<decltype(std::cout)>() };
        std::cout << ss.str();
        tids.insert(std::this_thread::get_id());
    });

    auto _size = tp->size();
    if(_all_exec != _size)
        throw std::runtime_error(
            std::string{ "Error! Did not execute on every thread: " } +
            std::to_string(_all_exec.load()) + " vs. " + std::to_string(_size));
    else
    {
        printf("Successful execution on every thread: %i\n", (int) _all_exec);
    }

    //------------------------------------------------------------------------//
    //                                                                        //
    //                  Execute function on specific threads                  //
    //                                                                        //
    //------------------------------------------------------------------------//

    auto _target_sz = tids.size() / 2 + ((tids.size() % 2 == 0) ? 0 : 1);
    if(_target_sz == 0)
        _target_sz = 1;
    std::atomic<size_t> _specific_exec{ 0 };
    while(tids.size() > _target_sz)
        tids.erase(tids.begin());

    tp->execute_on_specific_threads(tids, [&_specific_exec]() {
        ++_specific_exec;
        std::stringstream ss;
        ss << "thread " << std::setw(4) << PTL::Threading::GetThreadId() << " executed [specific]\n";
        AutoLock lk{ TypeMutex<decltype(std::cout)>() };
        std::cout << ss.str();
    });

    if(_specific_exec != _target_sz)
        throw std::runtime_error(
            std::string{ "Error! Did not execute on specific thread: " } +
            std::to_string(_specific_exec.load()) + " vs. " + std::to_string(_target_sz));
    else
    {
        printf("Successful execution on subset of threads: %i\n", (int) _specific_exec);
    }

    //------------------------------------------------------------------------//
    //                                                                        //
    //                        TaskGroup examples/tests                        //
    //                                                                        //
    //------------------------------------------------------------------------//
    {
        long     nfib  = std::max<long>(GetEnv<long>("FIBONACCI", 30), 30);
        long     nloop     = 100;
        long     ndiv      = 4;
        long     npart     = nloop / ndiv;
        long expected = (fibonacci(nfib + 0) * npart) + (fibonacci(nfib + 1) * npart) +
                        (fibonacci(nfib + 2) * npart) + (fibonacci(nfib + 3) * npart);
        auto join = [](long& lhs, long rhs) {
            std::stringstream ss;
            ss << "thread " << std::setw(4) << PTL::Threading::GetThreadId() << " adding "
               << rhs << " to " << lhs << std::endl;
            {
                AutoLock lk{ TypeMutex<decltype(std::cout)>() };
                std::cout << ss.str();
            }
            return lhs += rhs;
        };

        auto     entry = [](uint64_t n) {
            std::vector<double> v(n * 100, 0);
            for(auto& itr : v)
            {
                AutoLock lk{ TypeMutex<decltype(rng)>() };
                itr = std::generate_canonical<double, 12>(rng);
            }
            auto e = random_entry(v);
            std::stringstream ss;
            ss << "[" << n << "]> random entry from thread " << std::setw(4)
               << PTL::Threading::GetThreadId() << " was : " << std::setw(8)
               << std::setprecision(6) << std::fixed << e << std::endl;
            AutoLock lk{ TypeMutex<decltype(std::cout)>() };
            std::cout << ss.str();
        };

        TaskGroup<long> tgf(join);
        TaskGroup<void> tgv;
        for(long i = 0; i < nloop; ++i)
        {
            tgf.exec(fibonacci, nfib + (i % ndiv));
            tgv.exec(consume, 100);
            tgv.exec(do_sleep, 50);
            tgv.exec(entry, i + 1);
        }

        auto ret = tgf.join();
        tgv.join();
        std::cout << "fibonacci(" << nfib << ") * " << nloop << " = " << ret << std::endl;
        std::cout << std::endl;
        if(expected != ret)
        {
            throw std::runtime_error(std::string{ "Error wrong answer! Expected: " } +
                                     std::to_string(expected) +
                                     ". Result: " + std::to_string(ret));
        }
    }

    //------------------------------------------------------------------------//
    //                                                                        //
    //                      Asynchronous examples/tests                       //
    //                                                                        //
    //------------------------------------------------------------------------//
    {
        long nfib = std::max<long>(GetEnv<long>("FIBONACCI", 35), 30);
        std::vector<std::shared_ptr<VTask>> _asyncs{};
        std::vector<std::future<int64_t>>   _futures{};
        _asyncs.reserve(nthreads);
        _futures.reserve(nthreads);
        Timer _at{};
        _at.Start();
        for(decltype(nthreads) i = 0; i < nthreads; ++i)
        {
            auto fib_async = task_manager->async<int64_t>(fibonacci, nfib);
            _futures.emplace_back(fib_async->get_future());
            _asyncs.emplace_back(fib_async);
        }
        std::vector<int64_t> _values{};
        _values.reserve(nthreads);
        for(decltype(nthreads) i = 0; i < nthreads; ++i)
        {
            _values.emplace_back(_futures.at(i).get());
        }
        _at.Stop();
        for(decltype(nthreads) i = 0; i < nthreads; ++i)
        {
            std::cout << "[async test][" << i << "] fibonacci(" << nfib
                      << ") = " << _values[i] << std::endl;
        }
        std::cout << "[async test] " << _at << std::endl;
        std::cout << std::endl;
    }

    //------------------------------------------------------------------------//
    //                                                                        //
    //                          Task-group example/test                       //
    //                                                                        //
    //------------------------------------------------------------------------//

    // print the time for the calculation
    total_timer.Stop();
    std::cout << "Total time: \t" << total_timer << std::endl;

    tp->destroy_threadpool();
    task_manager->finalize();
}
