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
//  PTL common header for examples
//
//
//============================================================================//

#include "PTL/Task.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
#include "PTL/Threading.hh"
#include "PTL/TiMemory.hh"
#include "PTL/Timer.hh"
#include "PTL/Utility.hh"

using namespace PTL;

// C headers
#include <cstdlib>  // setenv
#include <stdlib.h>

// C++ headers
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

#if defined(PTL_USE_GPERF)
#    include <gperftools/heap-checker.h>
#    include <gperftools/heap-profiler.h>
#    include <gperftools/profiler.h>
#endif

using std::cerr;
using std::cout;
using std::endl;
using std::string;

//============================================================================//

#if defined(PTL_USE_ITTNOTIFY)
#    include <ittnotify.h>
#    define _pause_collection __itt_pause()
#    define _resume_collection __itt_resume()
#else
#    define _pause_collection
#    define _resume_collection
#endif

#if defined(PTL_USE_GPERF)
#    define _cpu_profiler_start(fname) ProfilerStart(fname)
#    define _cpu_profiler_flush ProfilerFlush()
#    define _cpu_profiler_stop ProfilerStop()
#    define _heap_profiler_start(fname) HeapProfilerStart(fname)
#    define _heap_profiler_flush HeapProfilerFlush()
#    define _heap_profiler_stop HeapProfilerStop()
#else
#    define _cpu_profiler_start(fname)
#    define _cpu_profiler_flush
#    define _cpu_profiler_stop
#    define _heap_profiler_start(fname)
#    define _heap_profiler_flush
#    define _heap_profiler_stop
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

//============================================================================//

// some typedefs to simplify declarations
typedef std::vector<int64_t>       Array_t;
typedef std::default_random_engine random_engine_t;
typedef std::vector<float>         farray_t;
typedef std::vector<int64_t>       iarray_t;

//============================================================================//

// some constants
const string   prefix    = "\n\t### ==> ";
const string   cprefix   = "\t### ==> ";
static int16_t rng_range = 2;

//============================================================================//
//
// the first template parameter is the result type, the second
// template parameter is optional. It will default to the first
// template parameter is not specified. It is available for
// when the results of individual tasks need to be combined into
// a different data type. In the fibonacci calculation of order 43
// the result using int will overflow the max value for int,
// hence why I am using it here

#if defined(USE_TBB_TASKS)
const bool                                    useTBB = true;
typedef TBBTaskGroup<Array_t, const int64_t&> TaskGroup_t;
typedef tbb::task_group                       VoidGroup_t;
typedef TBBTaskGroup<long>                    LongGroup_t;
#else
const bool                                 useTBB = false;
typedef TaskGroup<Array_t, const int64_t&> TaskGroup_t;
typedef TaskGroup<void, void, 10>          VoidGroup_t;
typedef TaskGroup<long>                    LongGroup_t;
#endif

//============================================================================//

struct Measurement
{
    long   cutoff;
    long   num_task_groups;
    long   nthreads;
    double ncount         = 0.0;
    double real           = 0.0;
    double cpu            = 0.0;
    double cpu_per_thread = 0.0;
    double cpu_util       = 0.0;

    Measurement(long _cutoff, long _ntg, long _nthreads)
    : cutoff(_cutoff)
    , num_task_groups(_ntg)
    , nthreads(_nthreads)
    {}

    bool operator==(const Measurement& rhs) const
    {
        return num_task_groups == rhs.num_task_groups;
    }

    bool operator!=(const Measurement& rhs) const { return !(*this == rhs); }

    bool operator()(const Measurement& rhs) const
    {
        return num_task_groups < rhs.num_task_groups;
    }

    bool operator<(const Measurement& rhs) const
    {
        return num_task_groups < rhs.num_task_groups;
    }

    bool operator>(const Measurement& rhs) const
    {
        return !(*this < rhs || *this == rhs);
    }

    bool operator>=(const Measurement& rhs) const { return !(*this < rhs); }

    bool operator<=(const Measurement& rhs) const
    {
        return (*this < rhs || *this == rhs);
    }

    Measurement& operator+=(const Timer& _timer)
    {
        real += _timer.GetRealElapsed();
        double _cpu = _timer.GetUserElapsed() + _timer.GetSystemElapsed();
        cpu += _cpu;
        cpu_per_thread += _cpu / nthreads;
        cpu_util += (_cpu / _timer.GetRealElapsed()) * 100.0;
        ncount += 1.0;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Measurement& m)
    {
        os << m.cutoff << ", " << m.num_task_groups << ", " << (m.real / m.ncount) << ", "
           << (m.cpu / m.ncount) << ", " << (m.cpu_per_thread / m.ncount) << ", "
           << (m.cpu_util / m.ncount) << ", " << m.ncount;
        return os;
    }
};

//============================================================================//

inline void
message(TaskRunManager* runmanager)
{
    cout << "\n\t--> Running in multithreaded mode with "
         << runmanager->GetNumberOfThreads() << " threads\n"
         << endl;
}

//============================================================================//

inline uint32_t
get_seed()
{
    static const uint32_t        seed_base   = 6734525;
    static const uint32_t        seed_factor = 1000;
    static std::atomic<uint32_t> _counter;
    static thread_local uint32_t _tid = ++_counter;
    return seed_base + (_tid * seed_factor);
}

//============================================================================//

inline random_engine_t&
get_engine()
{
    static thread_local random_engine_t* _engine = new random_engine_t(get_seed());
    return (*_engine);
}

//============================================================================//

template <typename _Tp = double>
_Tp
get_random()
{
    return std::generate_canonical<_Tp, std::numeric_limits<_Tp>::digits>(get_engine());
}

//============================================================================//

inline int64_t
get_random_int(int64_t _range = rng_range)
{
    static thread_local std::uniform_int_distribution<int64_t>* _instance =
        new std::uniform_int_distribution<int64_t>(-_range, _range);
    return (*_instance)(get_engine());
}

//============================================================================//

inline int64_t
fibonacci(int64_t n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

//============================================================================//

inline std::atomic_uintmax_t&
task_group_cnt()
{
    static std::atomic_uintmax_t _instance(0);
    return _instance;
}

//============================================================================//

inline int64_t
compute_sum(const Array_t& arr)
{
    int64_t _sum = 0;
    for(const auto& itr : arr)
    {
        _sum += itr;
    }
    return _sum;
}

//============================================================================//

inline void
append(Array_t& lhs, TaskGroup_t* rhs)
{
    if(rhs)
        for(auto& itr : rhs->join())
            lhs.push_back(itr);
}

//============================================================================//

inline std::string
get_gperf_filename(const char* arg0, const std::string& ftype)
{
    uintmax_t   n     = 0;
    std::string fname = "";
    while(fname.length() == 0)
    {
        std::ifstream     in;
        std::stringstream ss;
        ss << arg0 << ".gperf." << ftype << "." << n++;
        in.open(ss.str().c_str());
        if(!in)
            fname = ss.str();
    }
    return fname;
}
//============================================================================//
