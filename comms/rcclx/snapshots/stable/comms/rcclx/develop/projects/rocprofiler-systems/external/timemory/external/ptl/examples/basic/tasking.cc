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

#include "common/utils.hh"

//============================================================================//

void
execute_cpu_iterations(int64_t num_iter, TaskGroup_t* task_group, int64_t n,
                       int64_t& remaining, bool verbose = true)
{
    if(!task_group)
        return;

    if(num_iter > remaining)
        num_iter = remaining;
    remaining -= num_iter;

    // add an element of randomness
    static std::atomic<uint32_t> _counter;
    uint32_t                     _seed = get_seed() + (++_counter * 10000);
    get_engine().seed(_seed);

    std::stringstream ss;
    if(verbose)
    {
        ss << cprefix << "Submitting " << num_iter << " tasks computing \"fibonacci(" << n
           << ")\" to task manager "
           << "(" << remaining << " iterations remaining)..." << std::flush;
    }

    auto* taskManager = TaskRunManager::GetMasterRunManager()->GetTaskManager();

    Timer t;
    t.Start();
    for(uint32_t i = 0; i < num_iter; ++i)
    {
        taskManager->exec(*task_group, fibonacci, n + get_random_int());
    }
    t.Stop();
    if(verbose)
    {
        ss << " " << t << endl;
        AutoLock l(TypeMutex<decltype(std::cout)>());
        cout << ss.str();
    }
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

    auto hwthreads      = std::thread::hardware_concurrency();
    auto default_fib    = 30;
    auto default_tg     = 1;
    auto default_grain  = std::pow(16, 3);
    auto default_ntasks = std::pow(16, 4);
    ;
    auto default_nthreads = hwthreads;
    // cutoff fields
    auto cutoff_high  = 40;
    auto cutoff_low   = 25;
    auto cutoff_incr  = 5;
    auto cutoff_tasks = 1;
    long cutoff_value = 44;  // greater than 45 answer exceeds INT_MAX

    // default environment controls but don't overwrite
    setenv("NUM_THREADS", std::to_string(hwthreads).c_str(), 0);
    setenv("FIBONACCI", std::to_string(default_fib).c_str(), 0);
    setenv("GRAINSIZE", std::to_string(default_grain).c_str(), 0);
    setenv("NUM_TASKS", std::to_string(default_ntasks).c_str(), 0);
    setenv("NUM_TASK_GROUPS", std::to_string(default_tg).c_str(), 0);

    rng_range           = GetEnv<decltype(rng_range)>("RNG_RANGE", rng_range,
                                            "Setting RNG range to +/- this value");
    unsigned numThreads = GetEnv<unsigned>("NUM_THREADS", default_nthreads,
                                           "Getting the number of threads");
    int64_t  nfib       = GetEnv<int64_t>("FIBONACCI", default_fib,
                                   "Setting the centerpoint of fib work distribution");
    int64_t  grainsize  = GetEnv<int64_t>(
        "GRAINSIZE", numThreads, "Dividing number of task into grain of this size");
    int64_t num_iter = GetEnv<int64_t>("NUM_TASKS", numThreads * numThreads,
                                       "Setting the number of total tasks");
    int64_t num_groups =
        GetEnv<int64_t>("NUM_TASK_GROUPS", 4, "Setting the number of task groups");

    cutoff_high  = GetEnv<int>("CUTOFF_HIGH", cutoff_high);
    cutoff_incr  = GetEnv<int>("CUTOFF_INCR", cutoff_incr);
    cutoff_low   = GetEnv<int>("CUTOFF_LOW", cutoff_low);
    cutoff_tasks = GetEnv<int>("CUTOFF_TASKS", cutoff_tasks);
    cutoff_value = GetEnv<long>("CUTOFF_VALUE", cutoff_value);

    PrintEnv();

    Timer total_timer;
    total_timer.Start();

    // Construct the default run manager
    TaskRunManager* runManager = new TaskRunManager(useTBB);
    runManager->Initialize(numThreads);
    message(runManager);

    // the TaskManager is a utility that wraps the
    // function calls into tasks for the ThreadPool
    TaskManager* taskManager = runManager->GetTaskManager();

    //------------------------------------------------------------------------//
    //                                                                        //
    //                      Asynchronous examples/tests                       //
    //                                                                        //
    //------------------------------------------------------------------------//
    {
        auto    fib_async = taskManager->async<int64_t>(fibonacci, nfib);
        int64_t fib_n     = fib_async->get();
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
    auto cpu_join = [&](Array_t& ref, const int64_t& thread_local_solution) {
        true_answer += thread_local_solution;
        // ref.push_back(thread_local_solution);
        ref.emplace_back(thread_local_solution);
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
        {
            _task_group = new TaskGroup_t(cpu_join);
            _task_group->reserve(grainsize);
        }
    };
    //------------------------------------------------------------------------//
    std::vector<TaskGroup_t*> cpu_task_groups(num_groups, nullptr);
    int64_t                   remaining = num_iter;

    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                         FAKE SECTION                                 ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///

    std::cout << cprefix << "BEGIN OF FAKE RUN" << std::endl;
    //------------------------------------------------------------------------//
    for(auto& itr : cpu_task_groups)
    {
        // create the task group
        cpu_create(itr);
        // submit task with first task group
        execute_cpu_iterations(hwthreads, itr, 10, remaining, false);
    }
    //------------------------------------------------------------------------//
    // make sure all task groups finished (does join)
    for(auto& itr : cpu_task_groups)
    {
        // join task group
        itr->join();
        // delete task groups
        del(itr);
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

#if defined(USE_TBB_TASKS)
    cout << prefix << "Running with TBB backend..." << std::endl;
#else
    cout << prefix << "Running with PTL backend..." << std::endl;
#endif

    Timer timer;

    //------------------------------------------------------------------------//
    timer.Start();       // start timer for calculation
    _resume_collection;  // for VTune
    //------------------------------------------------------------------------//

    //------------------------------------------------------------------------//
    std::vector<Array_t> cpu_results(num_groups);
    remaining = num_iter;
    true_answer.store(0);
    //------------------------------------------------------------------------//

    while(remaining > 0)
    {
        for(size_t i = 0; i < cpu_task_groups.size(); ++i)
        {
            // create the task group
            cpu_create(cpu_task_groups[i]);
            // submit task with first task group
            execute_cpu_iterations(grainsize, cpu_task_groups[i], nfib, remaining);
            // make sure all task groups finished (does join)
            append(cpu_results[i], cpu_task_groups[i]);
        }
    }

    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                         JOIN RESULTS                                 ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///
    std::cout << prefix << "CPU completed" << std::endl;

    // compute the anser
    int64_t cpu_answer = 0;
    for(auto& itr : cpu_results)
    {
        cpu_answer += compute_sum(itr);
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

    cout << prefix << "[task group] fibonacci(" << nfib << " +/- " << rng_range
         << ") = " << cpu_answer << endl;
    cout << cprefix << "  [atomic]   fibonacci(" << nfib << " +/- " << rng_range
         << ") = " << true_answer << endl;

    std::stringstream fibprefix;
    fibprefix << "fibonacci(" << nfib << " +/- " << rng_range << ") calculation time: ";
    int32_t _w = static_cast<int32_t>(fibprefix.str().length()) + 2;

    cout << prefix << std::setw(_w) << fibprefix.str() << "\t" << timer << endl;

    for(auto& itr : cpu_task_groups)
        del(itr);

    // print the time for the calculation
    total_timer.Stop();
    cout << cprefix << std::setw(_w) << "Total time: "
         << "\t" << total_timer << endl;

    int64_t     ret = (true_answer - cpu_answer);
    std::string msg = (ret == 0) ? "Successful MT" : "Failure combining";
    cout << prefix << msg << " fibonacci calculation" << endl << endl;

    delete runManager;

    return ret;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
