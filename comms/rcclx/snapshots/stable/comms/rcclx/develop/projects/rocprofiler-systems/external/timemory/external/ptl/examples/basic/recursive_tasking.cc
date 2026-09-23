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
/// \file recursive_tasking.cc
/// \brief Example showing the usage of recursive tasking
//

#include "common/utils.hh"

#if defined(PTL_USE_GPERF)
#    include <gperftools/profiler.h>
#endif

//============================================================================//

template <typename TaskGroup_t>
int64_t
task_fibonacci(int64_t n, int64_t cutoff)
{
    if(n < 2)
        return n;

    int64_t     x = 0;
    int64_t     y = 0;
    TaskGroup_t g{};
    ++task_group_cnt();
    if(n >= cutoff)
    {
        g.run([&x, n, cutoff]() { x = task_fibonacci<TaskGroup_t>(n - 1, cutoff); });
        g.run([&y, n, cutoff]() { y = task_fibonacci<TaskGroup_t>(n - 2, cutoff); });
    }
    else
    {
        g.run([&x, n]() { x = fibonacci(n - 1); });
        g.run([&y, n]() { y = fibonacci(n - 2); });
    }
    // wait for both tasks to complete
    g.wait();
    return x + y;
}

//============================================================================//

void
execute_iterations(int64_t num_iter, TaskGroup_t* task_group, int64_t n,
                   int64_t& remaining)
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

    cout << cprefix << "Submitting " << num_iter << " tasks computing \"fibonacci(" << n
         << ")\" to task manager "
         << "(" << remaining << " iterations remaining)..." << std::flush;

    Timer t;
    t.Start();
    for(uint32_t i = 0; i < num_iter; ++i)
    {
        task_group->exec(fibonacci, n + get_random_int());
    }
    t.Stop();
    cout << " " << t << endl;
}

//============================================================================//

int
main(int argc, char** argv)
{
#if defined(PTL_USE_TIMEMORY)
    tim::enable_signal_detection();
#endif

    _pause_collection;  // VTune
    //_heap_profiler_start(get_gperf_filename(argv[0], "heap").c_str());  //
    // gperf

#if defined(PTL_USE_TIMEMORY)
    tim::manager* manager = tim::manager::instance();
#endif

    ConsumeParameters(argc, argv);

    auto hwthreads        = std::thread::hardware_concurrency();
    auto default_fib      = 20;
    auto default_tg       = 1;
    auto default_grain    = pow(16, 1);
    auto default_ntasks   = pow(16, 1);
    auto default_nthreads = hwthreads;
    // cutoff fields
    long cutoff_value = 30;  // greater than 45 answer exceeds INT_MAX
    auto cutoff_high  = cutoff_value;
    auto cutoff_low   = 15;
    auto cutoff_incr  = 5;
    auto cutoff_tasks = 1;

    // default environment controls but don't overwrite
    setenv("NUM_THREADS", std::to_string(hwthreads).c_str(), 0);
    setenv("FIBONACCI", std::to_string(default_fib).c_str(), 0);
    setenv("GRAINSIZE", std::to_string(default_grain).c_str(), 0);
    setenv("NUM_TASKS", std::to_string(default_ntasks).c_str(), 0);
    setenv("NUM_TASK_GROUPS", std::to_string(default_tg).c_str(), 0);

    rng_range           = GetEnv<decltype(rng_range)>("RNG_RANGE", rng_range + 6,
                                            "Setting RNG range to +/- this value");
    unsigned numThreads = GetEnv<unsigned>("NUM_THREADS", default_nthreads,
                                           "Getting the number of threads");
    int64_t  nfib       = GetEnv<int64_t>("FIBONACCI", default_fib,
                                   "Setting the centerpoint of fib work distribution");
    int64_t  grainsize  = GetEnv<int64_t>(
        "GRAINSIZE", numThreads, "Dividing number of task into grain of this size");
    int64_t num_iter = numThreads * numThreads;
    int64_t num_groups =
        GetEnv<int64_t>("NUM_TASK_GROUPS", 4, "Setting the number of task groups");

    cutoff_value = GetEnv<long>("CUTOFF_VALUE", cutoff_value);
    cutoff_high  = GetEnv<int>("CUTOFF_HIGH", cutoff_value);
    cutoff_low   = GetEnv<int>("CUTOFF_LOW", cutoff_low);
    cutoff_incr  = GetEnv<int>("CUTOFF_INCR", cutoff_incr);
    cutoff_tasks = GetEnv<int>("CUTOFF_TASKS", cutoff_tasks);

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
    //                Asynchronous and Recursion examples/tests               //
    //                                                                        //
    //------------------------------------------------------------------------//
    Timer singleTimer;
    // run with async
    int64_t fib_async = 0;
    {
        singleTimer.Start();
        auto fib_tmp = taskManager->async<intmax_t>(fibonacci, cutoff_value);
        fib_async    = fib_tmp->get();
        singleTimer.Stop();

        cout << prefix << "[async test] fibonacci(" << cutoff_value << ") * "
             << cutoff_tasks << " = " << fib_async << " ... " << singleTimer << endl;
    }

#if defined(USE_TBB_TASKS)
    cout << prefix << "Running with TBB task_group..." << std::endl;
#else
    cout << prefix << "Running with PTL task_group..." << std::endl;
#endif

    std::vector<int> cutoffs;
    for(int i = cutoff_high; i >= cutoff_low; i -= cutoff_incr)
        cutoffs.push_back(i);

    //------------------------------------------------------------------------//
    auto run_recursive = [=](LongGroup_t& fib_tmp, int cutoff) {
        fib_tmp.exec(task_fibonacci<VoidGroup_t>, cutoff_value, cutoff);
    };
    //------------------------------------------------------------------------//

    std::map<int, Measurement*> measurements;
    // run with recursive
    Timer measureTimer;
    measureTimer.Start();
    for(int i = 0; i < cutoff_tasks; ++i)
    {
        cout << cprefix << "iteration #" << i << " of " << cutoff_tasks << "..." << endl;
        for(auto cutoff : cutoffs)
        {
            int64_t fib_recur = 0;
            task_group_cnt().store(0);

            singleTimer.Start();

            if(cutoff == cutoff_high)
            {
                _resume_collection;  // for VTune
            }

            LongGroup_t fib_tmp([](long& _ref, long _i) { return _ref += _i; });
            run_recursive(fib_tmp, cutoff);
            fib_recur = fib_tmp.join();

            if(cutoff == cutoff_high)
            {
                _pause_collection;  // for VTune
            }

            singleTimer.Stop();

            auto num_task_groups = task_group_cnt().load();

            Measurement* measurement = nullptr;
            if(measurements.find(num_task_groups) != measurements.end())
                measurement = measurements.find(num_task_groups)->second;
            if(!measurement)
            {
                measurement =
                    new Measurement(cutoff, num_task_groups, taskManager->size());
                measurements[num_task_groups] = measurement;
            }

            if(measurement)
                *measurement += singleTimer;

            cout << cprefix << "[recur test] fibonacci(" << cutoff_value << ") * " << i
                 << " = " << fib_recur << " ... " << singleTimer << " ... [# task grp] "
                 << num_task_groups << " (cutoff = " << cutoff
                 << ") "
                 //<< measurement->real
                 << endl;

            if(fib_async != fib_recur)
            {
                cerr << cprefix << "Warning! async != recursive: " << fib_async
                     << " != " << fib_recur << endl;
            }
        }
    }
    measureTimer.Stop();
    std::cout << prefix << "Total measurement time: " << measureTimer << std::endl;
    std::stringstream ss;
    ss << argv[0] << "_recursive.dat";
    std::ofstream ofs(ss.str().c_str());
    if(ofs)
    {
        std::set<Measurement> _measurements;
        for(auto itr : measurements)
            _measurements.insert(*(itr.second));
        for(const auto& itr : _measurements)
            ofs << itr << endl;
    }
    ofs.close();
    for(auto itr : measurements)
        delete itr.second;
    measurements.clear();

    cout << endl;

    //------------------------------------------------------------------------//
    //                                                                        //
    //                          Task-group example/test                       //
    //                                                                        //
    //------------------------------------------------------------------------//
    std::atomic_uintmax_t true_answer(0);

    // start timer for calculation
    Timer timer;
    timer.Start();

    _resume_collection;  // for VTune

    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                     PRIMARY TASKING SECTION                          ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///
    // this function joins task results
    auto join = [&](Array_t& ref, const int64_t& thread_local_solution) {
        true_answer += thread_local_solution;
        // ref.push_back(thread_local_solution);
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
    auto create = [=](TaskGroup_t*& _task_group) {
        if(!_task_group)
            _task_group = new TaskGroup_t(join);
    };
    //------------------------------------------------------------------------//
    std::vector<TaskGroup_t*> task_groups(num_groups, nullptr);
    std::vector<Array_t>      results(num_groups);
    int64_t                   remaining = num_iter;

    while(remaining > 0)
    {
        for(size_t i = 0; i < task_groups.size(); ++i)
        {
            // wait for task group to finish (does join) before delete + create
            append(results[i], task_groups[i]);

            // create the task group
            create(task_groups[i]);

            // submit task with first task group
            execute_iterations(grainsize, task_groups[i], nfib, remaining);

            // wait for old task groups to finish (does join)
            if(i + 1 < static_cast<size_t>(num_groups))
                append(results[i + 1], task_groups[i + 1]);

            if(remaining == 0)
                break;
        }
    }

    // make sure all task groups finished (does join)
    for(size_t i = 0; i < task_groups.size(); ++i)
        append(results[i], task_groups[i]);

    // compute the anser
    int64_t answer = 0;
    for(auto& itr : results)
    {
        answer += compute_sum(itr);
    }
    ///======================================================================///
    ///                                                                      ///
    ///                                                                      ///
    ///                 END OF PRIMARY TASKING SECTION                       ///
    ///                                                                      ///
    ///                                                                      ///
    ///======================================================================///

    _pause_collection;  // for VTune

    // stop timer for fibonacci
    timer.Stop();

    cout << prefix << "[task group] fibonacci(" << nfib << " +/- " << rng_range
         << ") = " << answer << endl;
    cout << cprefix << "  [atomic]   fibonacci(" << nfib << " +/- " << rng_range
         << ") = " << true_answer << endl;

    std::stringstream fibprefix;
    fibprefix << "fibonacci(" << nfib << " +/- " << rng_range << ") calculation time: ";
    int32_t _w = static_cast<int32_t>(fibprefix.str().length()) + 2;

    cout << prefix << std::setw(_w) << fibprefix.str() << "\t" << timer << endl;

    // KNL hangs somewhere between finishing calculations and total_timer
    Timer del_timer;
    del_timer.Start();

    for(auto& itr : task_groups)
        del(itr);

    del_timer.Stop();
    cout << cprefix << std::setw(_w) << "Task group deletion time: "
         << "\t" << del_timer << endl;

    // print the time for the calculation
    total_timer.Stop();
    cout << cprefix << std::setw(_w) << "Total time: "
         << "\t" << total_timer << endl;

    int64_t ret = (true_answer - answer);
    if(ret == 0)
    {
        cout << prefix << "Successful MT fibonacci calculation" << endl;
    }
    else
    {
        cout << prefix << "Failure combining MT fibonacci calculation " << endl;
    }

    cout << endl;

    delete runManager;

    //_heap_profiler_stop;
#if defined(PTL_USE_TIMEMORY)
    tim::disable_signal_detection();
#endif

    return ret;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
