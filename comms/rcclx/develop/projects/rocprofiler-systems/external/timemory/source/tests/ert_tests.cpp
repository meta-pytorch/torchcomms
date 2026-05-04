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

#define TIMEMORY_USER_ERT_FLOPS 2

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/ert.hpp"
#include "timemory/timemory.hpp"

#include <cstdint>
#include <set>

namespace dmp    = tim::dmp;
namespace gpu    = tim::gpu;
namespace ert    = tim::ert;
namespace device = tim::device;

using settings       = tim::settings;
using counter_type   = tim::component::wall_clock;
using fp16_t         = tim::gpu::fp16_t;
using ert_data_t     = ert::exec_data<counter_type>;
using ert_data_ptr_t = std::shared_ptr<ert_data_t>;
using init_list_t    = std::set<uint64_t>;

//--------------------------------------------------------------------------------------//

namespace details
{
//  Get the current tests name
inline std::string
get_test_name()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// run ert
template <typename Tp, typename DeviceT>
void
run_ert(ert_data_ptr_t, int64_t num_threads, int64_t min_size, int64_t max_data,
        int64_t num_streams = 0, int64_t block_size = 0, int64_t num_gpus = 0);

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
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class ert_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

TEST_F(ert_tests, run)
{
    auto data  = std::make_shared<ert_data_t>();
    auto nproc = dmp::size();

    auto cpu_min_size = 64;
    auto cpu_max_data = ert::cache_size::get_max() / 4;

    init_list_t cpu_num_threads;

    auto default_thread_init_list = init_list_t({ 1 });

    for(auto itr : default_thread_init_list)
    {
        auto entry = itr / nproc;
        if(entry > 0)
            cpu_num_threads.insert(entry);
    }

    if(cpu_num_threads.empty())
        cpu_num_threads.insert(1);

    TIMEMORY_BLANK_AUTO_TIMER("run_ert");

    for(int i = 0; i < tim::get_env<int>("NUM_ITER", 1); ++i)
    {
#if !defined(USE_CUDA)

        // execute the single-precision ERT calculations
        for(auto nthread : cpu_num_threads)
            details::run_ert<float, device::cpu>(data, nthread, cpu_min_size,
                                                 cpu_max_data);

        // execute the double-precision ERT calculations
        for(auto nthread : cpu_num_threads)
            details::run_ert<double, device::cpu>(data, nthread, cpu_min_size,
                                                  cpu_max_data);

#else

        auto gpu_min_size = 1 * units::megabyte;
        auto gpu_max_data = 500 * units::megabyte;

        // determine how many GPUs to execute on (or on CPU if zero)
        int         num_gpus        = gpu::device_count();
        init_list_t gpu_num_streams = { 1 };
        init_list_t gpu_block_sizes = { 32, 128, 256, 512, 1024 };

        if(num_gpus < 1)
        {
            // execute the single-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                details::run_ert<float, device::cpu>(data, nthread, cpu_min_size,
                                                     cpu_max_data);

            // execute the double-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                details::run_ert<double, device::cpu>(data, nthread, cpu_min_size,
                                                      cpu_max_data);
        }
        else  // num_gpus >= 1
        {
#    if defined(TIMEMORY_USE_CUDA_HALF)
            // execute the half-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!tim::get_env("ERT_GPU_FP16", false))
                            continue;
                        details::run_ert<fp16_t, device::gpu>(data, nthread, gpu_min_size,
                                                              gpu_max_data, nstream,
                                                              block, num_gpus);
                    }
#    endif
            // execute the single-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!tim::get_env("ERT_GPU_FP32", true))
                            continue;
                        details::run_ert<float, device::gpu>(data, nthread, gpu_min_size,
                                                             gpu_max_data, nstream, block,
                                                             num_gpus);
                    }

            // execute the double-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!tim::get_env("ERT_GPU_FP64", true))
                            continue;
                        details::run_ert<double, device::gpu>(data, nthread, gpu_min_size,
                                                              gpu_max_data, nstream,
                                                              block, num_gpus);
                    }
        }
#endif
    }

    std::string fname = "ert_results";

    printf("\n");
    ert::serialize(fname, *data);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT>
void
details::run_ert(ert_data_ptr_t data, int64_t num_threads, int64_t min_size,
                 int64_t max_data, int64_t num_streams, int64_t block_size,
                 int64_t num_gpus)
{
    // create a label for this test
    auto dtype = tim::demangle(typeid(Tp).name());
    auto htype = DeviceT::name();
    auto label =
        TIMEMORY_JOIN("", __FUNCTION__, '/', dtype, '/', htype, '/', num_threads,
                      "-threads", '/', min_size, "-min-ws", '/', max_data, "-max-size");

    if(std::is_same<DeviceT, device::gpu>::value)
    {
        label = TIMEMORY_JOIN("_", label, num_gpus, "gpus", num_streams, "streams",
                              block_size, "thr-per-blk");
    }

    printf("\n[ert-example]> Executing %s...\n", label.c_str());

    using ert_executor_type = ert::executor<DeviceT, Tp, counter_type>;
    using ert_config_type   = typename ert_executor_type::configuration_type;
    using ert_counter_type  = ert::counter<DeviceT, Tp, counter_type>;

    //
    // simple modifications to override method number of threads, number of streams,
    // block size, minimum working set size, and max data size
    //
    ert_config_type::get_num_threads()      = [=]() { return num_threads; };
    ert_config_type::get_num_streams()      = [=]() { return num_streams; };
    ert_config_type::get_block_size()       = [=]() { return block_size; };
    ert_config_type::get_min_working_size() = [=]() { return min_size; };
    ert_config_type::get_max_data_size()    = [=]() { return max_data; };

    //
    // create a callback function that sets the device based on the thread-id
    //
    auto set_counter_device = [=](uint64_t tid, ert_counter_type&) {
        if(num_gpus > 0)
            gpu::set_device(tid % num_gpus);
    };

    //
    // create a configuration object -- this handles a lot of the setup
    //
    ert_config_type config{};
    //
    // start generic timemory timer
    //
    TIMEMORY_BLANK_CALIPER(config, tim::auto_timer, label);
    TIMEMORY_CALIPER_APPLY(config, report_at_exit, true);
    //
    // "construct" an ert::executor that executes the configuration and inserts
    // into the data object.
    //
    // NOTE: the ert::executor has callbacks that allows one to customize the
    //       the ERT execution
    //
    // check the operations were performed
    {
        auto _counter = config.executor(data);
        _counter.set_callback(set_counter_device);
        auto _executed = ert_executor_type::template execute<1, 4>(
            _counter, { "scalar_add", "vector_fma" });
        EXPECT_TRUE(_executed);
    }
    // check that operations were skipped
    {
        auto _counter = config.executor(data);
        _counter.add_skip_ops(1);
        _counter.add_skip_ops(4);
        _counter.set_callback(set_counter_device);
        auto _executed = ert_executor_type::template execute<1, 4>(
            _counter, { "scalar_add", "vector_fma" });
        EXPECT_FALSE(_executed);
    }

    if(data && (settings::verbose() > 0 || settings::debug()))
        std::cout << "\n" << *(data) << std::endl;
    printf("\n");
}

//======================================================================================//
