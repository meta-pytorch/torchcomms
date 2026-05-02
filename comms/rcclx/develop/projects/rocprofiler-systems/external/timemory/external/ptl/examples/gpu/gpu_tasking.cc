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
// ---------------------------------------------------------------
//
//
/// \file gpu_tasking.cc
/// \brief Example showing the usage of tasking with GPU
//

#include "common/utils.hh"
#include "sum.hh"

#if defined(PTL_USE_TIMEMORY)
#    include "timemory/auto_timer.hpp"
#    include "timemory/manager.hpp"
#endif

//============================================================================//

ThreadPool*&
GetGpuPool()
{
    static thread_local ThreadPool* _instance = nullptr;
    return _instance;
}

//============================================================================//

TaskManager*&
GetGpuManager()
{
    static thread_local TaskManager* _instance = nullptr;
    return _instance;
}

//============================================================================//

void
execute_cpu_iterations(uint64_t num_iter, TaskGroup_t* task_group, uint64_t n,
                       uint64_t& remaining)
{
    if(remaining <= 0 || !task_group)
        return;

    if(num_iter > remaining)
        num_iter = remaining;
    remaining -= num_iter;

    // add an element of randomness
    static std::atomic_uintmax_t _counter;
    uintmax_t                    _seed = get_seed() + (++_counter * 10000);
    get_engine().seed(_seed);

    std::stringstream ss;
    ss << cprefix << "Submitting " << num_iter << " tasks computing \"fibonacci(" << n
       << ")\" to task manager "
       << "(" << remaining << " iterations remaining)..." << std::flush;

    TaskManager* taskManager = TaskRunManager::GetMasterRunManager()->GetTaskManager();

    Timer t;
    t.Start();
    for(uint32_t i = 0; i < num_iter; ++i)
    {
        int offset = get_random_int();
        taskManager->exec(*task_group, fibonacci, n + offset);
    }
    t.Stop();
    ss << " " << t << endl;

    AutoLock l(TypeMutex<decltype(std::cout)>());
    cout << ss.str();
}

//============================================================================//

void
execute_gpu_iterations(uint64_t num_iter, TaskGroup_t* task_group, uint64_t n)
{
    if(!task_group)
        return;

    // add an element of randomness
    static std::atomic_uintmax_t _counter;
    uintmax_t                    _seed = get_seed() + (++_counter * 10000);
    get_engine().seed(_seed);

    TaskManager* taskManager = GetGpuManager();

    std::stringstream ss;
    ss << cprefix << "Submitting " << num_iter << " tasks computing \"run_gpu(" << n
       << ")\" to task manager "
       << "..." << std::flush;

    Timer t;
    t.Start();
    for(uint32_t i = 0; i < num_iter; ++i)
    {
        int offset_a = get_random_int();
        int offset_b = get_random_int();
        taskManager->exec(*task_group, run_gpu, n + (offset_a * offset_b));
    }
    t.Stop();
    ss << " " << t << endl;

    AutoLock l(TypeMutex<decltype(std::cout)>());
    cout << ss.str();
}

//============================================================================//

int
main(int argc, char** argv)
{
    _pause_collection;  // for VTune

#if defined(PTL_USE_TIMEMORY)
    tim::manager* manager = tim::manager::instance();
#endif

    ConsumeParameters(argc, argv);

    auto hwthreads        = std::thread::hardware_concurrency();
    auto default_fib      = 42;
    auto default_tg       = 2;
    auto default_gpu      = 52;
    auto default_nthreads = (useTBB) ? (2 * hwthreads) : hwthreads;

    // default environment controls but don't overwrite
    setenv("NUM_THREADS", std::to_string(default_nthreads).c_str(), 0);
    setenv("FIBONACCI", std::to_string(default_fib).c_str(), 0);
    setenv("GPU_RANGE", std::to_string(default_gpu).c_str(), 0);
    setenv("GRAINSIZE", std::to_string(hwthreads).c_str(), 0);
    setenv("NUM_TASKS", std::to_string(hwthreads * hwthreads).c_str(), 0);
    setenv("NUM_TASK_GROUPS", std::to_string(default_tg).c_str(), 0);

    rng_range           = GetEnv<decltype(rng_range)>("RNG_RANGE", rng_range,
                                            "Setting RNG range to +/- this value");
    unsigned numThreads = GetEnv<unsigned>("NUM_THREADS", default_nthreads,
                                           "Getting the number of threads");
    uint64_t nfib       = GetEnv<uint64_t>("FIBONACCI", default_fib,
                                     "Setting the centerpoint of fib work distribution");
    uint64_t ngpu =
        GetEnv<uint64_t>("GPU_RANGE", default_gpu, "Setting the GPU range centerpoint");
    uint64_t grainsize = GetEnv<uint64_t>(
        "GRAINSIZE", numThreads, "Dividing number of task into grain of this size");
    uint64_t num_iter = GetEnv<uint64_t>("NUM_TASKS", numThreads * numThreads,
                                         "Setting the number of total tasks");
    uint64_t num_groups =
        GetEnv<uint64_t>("NUM_TASK_GROUPS", 4, "Setting the number of task groups");
    PrintEnv();

    Timer total_timer;
    total_timer.Start();

    // Construct the default run manager
    TaskRunManager* runManager = new TaskRunManager(useTBB);
    // TaskRunManager* runManager = TaskRunManager::GetMasterRunManager(useTBB);
    runManager->Initialize(numThreads);
    message(runManager);

    // the TaskManager is a utility that wraps the
    // function calls into tasks for the ThreadPool
    TaskManager* taskManager = runManager->GetTaskManager();

    ThreadPool*&  gpu_tp  = GetGpuPool();
    TaskManager*& gpu_man = GetGpuManager();

    if(!useTBB)
    {
        gpu_tp  = new ThreadPool(numThreads * numThreads, nullptr, false);
        gpu_man = new TaskManager(gpu_tp);
    }
    else
    {
        gpu_tp  = runManager->GetThreadPool();
        gpu_man = taskManager;
    }

    //------------------------------------------------------------------------//
    //                                                                        //
    //                      Asynchronous examples/tests                       //
    //                                                                        //
    //------------------------------------------------------------------------//
    {
        auto     fib_async = taskManager->async<uint64_t>(fibonacci, nfib);
        uint64_t fib_n     = fib_async->get();
        std::cout << prefix << "[async test] fibonacci(" << nfib << " +/- " << rng_range
                  << ") = " << fib_n << std::endl;
        std::cout << std::endl;
    }

    //------------------------------------------------------------------------//
    //                                                                        //
    //                          Task-group example/test                       //
    //                                                                        //
    //------------------------------------------------------------------------//
    std::atomic_uintmax_t true_answer(0);

    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                     PRIMARY TASKING SECTION                          ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///
    // this function joins task results
    auto cpu_join = [&](Array_t& ref, const uint64_t& thread_local_solution) {
        true_answer += thread_local_solution;
        ref.push_back(thread_local_solution);
        return ref;
    };
    //------------------------------------------------------------------------//
    // this function joins task results
    auto gpu_join = [](Array_t& ref, const uint64_t& thread_local_solution) {
        ref.push_back(thread_local_solution);
        return ref;
    };
    //------------------------------------------------------------------------//
    // this function deletes task groups
    auto del = [](TaskGroup_t*& _task_group) {
        delete _task_group;
        _task_group = nullptr;
    };
    //------------------------------------------------------------------------//
    // create a task group
    auto cpu_create = [=](TaskGroup_t*& _task_group) {
        if(!_task_group)
            _task_group = new TaskGroup_t(cpu_join);
    };
    //------------------------------------------------------------------------//
    // create a task group
    auto gpu_create = [=](TaskGroup_t*& _task_group) {
        if(!_task_group)
            _task_group = new TaskGroup_t(gpu_join, gpu_tp);
    };
    //------------------------------------------------------------------------//
    std::vector<TaskGroup_t*> cpu_task_groups(num_groups, nullptr);
    std::vector<TaskGroup_t*> gpu_task_groups(num_groups, nullptr);
    uint64_t                  remaining = num_iter;

    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                         FAKE SECTION                                 ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///

    std::cout << cprefix << "BEGIN OF FAKE RUN" << std::endl;
    //------------------------------------------------------------------------//
    for(uint64_t i = 0; i < cpu_task_groups.size(); ++i)
    {
        // create the task group
        cpu_create(cpu_task_groups[i]);
        gpu_create(gpu_task_groups[i]);
        // submit task with first task group
        execute_cpu_iterations(hwthreads, cpu_task_groups[i], hwthreads, remaining);
        execute_gpu_iterations(hwthreads, gpu_task_groups[i], hwthreads);
    }
    //------------------------------------------------------------------------//
    // make sure all task groups finished (does join)
    for(uint64_t i = 0; i < cpu_task_groups.size(); ++i)
    {
        cpu_task_groups[i]->join();
        del(cpu_task_groups[i]);
        gpu_task_groups[i]->join();
        del(gpu_task_groups[i]);
    }
    //------------------------------------------------------------------------//
    std::cout << cprefix << "END OF FAKE RUN\n" << std::endl;
    //------------------------------------------------------------------------//

    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                         WORK SECTION                                 ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///
    Timer timer;

    //------------------------------------------------------------------------//
    timer.Start();       // start timer for calculation
    _resume_collection;  // for VTune
    //------------------------------------------------------------------------//

    //------------------------------------------------------------------------//
    std::vector<Array_t> cpu_results(num_groups);
    std::vector<Array_t> gpu_results(num_groups);
    remaining = num_iter;
    true_answer.store(0);
    //------------------------------------------------------------------------//

    while(remaining > 0)
    {
        for(uint64_t i = 0; i < cpu_task_groups.size(); ++i)
        {
            // create the task group
            cpu_create(cpu_task_groups[i]);
            gpu_create(gpu_task_groups[i]);
            // submit task with first task group
            execute_cpu_iterations(grainsize, cpu_task_groups[i], nfib, remaining);
            execute_gpu_iterations(grainsize, gpu_task_groups[i], ngpu);
        }
    }

    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                         JOIN RESULTS                                 ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///
    // make sure all task groups finished (does join)
    for(uint64_t i = 0; i < cpu_task_groups.size(); ++i)
        append(cpu_results[i], cpu_task_groups[i]);
    Timer diff;
    diff.Start();
    std::cout << prefix << "CPU completed" << std::endl;

    for(uint64_t i = 0; i < gpu_task_groups.size(); ++i)
        append(gpu_results[i], gpu_task_groups[i]);
    diff.Stop();
    std::cout << cprefix << "GPU completed" << std::endl;
    std::cout << cprefix << "CPU vs. GPU imbalance:\t\t" << diff << std::endl;

    // compute the anser
    uint64_t cpu_answer = 0;
    uint64_t gpu_answer = 0;
    for(uint64_t i = 0; i < cpu_task_groups.size(); ++i)
    {
        cpu_answer += compute_sum(cpu_results[i]);
        gpu_answer += compute_sum(gpu_results[i]);
    }

    //------------------------------------------------------------------------//
    _pause_collection;  // for VTune
    timer.Stop();       // stop timer for fibonacci
    //------------------------------------------------------------------------//

    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                 END OF PRIMARY TASKING SECTION                       ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///

    std::cout << prefix << "[task group] fibonacci(" << nfib << " +/- " << rng_range
              << ") * " << num_iter << " = " << cpu_answer << std::endl;
    std::cout << cprefix << "[task group] run_gpu(" << ngpu << " +/- " << rng_range
              << ") * " << num_iter << " = " << gpu_answer << std::endl;
    std::cout << cprefix << "[atomic]     gpu-tasking answer: " << true_answer
              << std::endl;

    std::stringstream fibprefix;
    fibprefix << "gpu-tasking(...) calculation time: ";
    int32_t _w = fibprefix.str().length() + 2;

    std::cout << prefix << std::setw(_w) << fibprefix.str() << "\t" << timer << std::endl;

    // KNL hangs somewhere between finishing calculations and total_timer
    Timer del_timer;
    del_timer.Start();

    for(uint64_t i = 0; i < cpu_task_groups.size(); ++i)
        del(cpu_task_groups[i]);
    for(uint64_t i = 0; i < gpu_task_groups.size(); ++i)
        del(gpu_task_groups[i]);

    del_timer.Stop();
    std::cout << cprefix << std::setw(_w) << "Task group deletion time: "
              << "\t" << del_timer << std::endl;

    // print the time for the calculation
    total_timer.Stop();
    std::cout << cprefix << std::setw(_w) << "Total time: "
              << "\t" << total_timer << std::endl;

    uintmax_t ret = (true_answer - cpu_answer);
    if(ret == 0 && num_iter == gpu_answer)
        std::cout << prefix << "Successful MT gpu-tasking calculation" << std::endl;
    else
        std::cout << prefix << "Failure combining MT gpu-tasking calculation "
                  << std::endl;

    std::cout << std::endl;

    delete runManager;

#if defined(PTL_USE_TIMEMORY)
    std::string fname(argv[0]);
    manager->report(std::cout, true);
    manager->write_report(fname + ".out");
    manager->write_json(fname + ".json");
#endif

    return static_cast<int>(ret);
}

//============================================================================//
