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

#include "timemory-ert.hpp"

#include "timemory/components/rusage/components.hpp"
#include "timemory/components/timing/components.hpp"
#include "timemory/components/timing/ert_timer.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/config.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/operations/definition.hpp"
#include "timemory/settings.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/signals.hpp"
#include "timemory/variadic/lightweight_tuple.cpp"
#include "timemory/variadic/lightweight_tuple.hpp"

namespace units = tim::units;
namespace comp  = tim::component;

using bundle_t = tim::lightweight_tuple<comp::wall_clock, comp::cpu_clock, comp::cpu_util,
                                        comp::peak_rss>;

//--------------------------------------------------------------------------------------//

bool
get_verbose()
{
    return (settings::verbose() > 0 || settings::debug());
}

void*
start_generic(const std::string& _label)
{
    return static_cast<void*>(new bundle_t{ _label });
}

void
stop_generic(void* _ptr)
{
    if(!_ptr)
        return;
    bundle_t* _bundle = static_cast<bundle_t*>(_ptr);
    if(get_verbose())
        std::cerr << *_bundle << std::endl;
    delete _bundle;
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    size_t      num_iter         = 1;
    std::string fname            = "ert_results";
    auto        init_num_threads = std::set<uint64_t>{ 1, 2 };

    bool    enable_cpu      = false;
    int64_t cpu_min_size    = 64;
    int64_t cpu_max_data    = 2 * ert::cache_size::get_max();
    auto    cpu_num_threads = std::set<uint64_t>{};
    auto    cpu_ftypes      = std::set<std::string>{ "fp32", "fp64" };

    bool    enable_gpu      = false;
    int64_t num_gpus        = gpu::device_count();
    int64_t gpu_min_size    = 1 * units::megabyte;
    int64_t gpu_max_data    = 500 * units::megabyte;
    auto    gpu_num_threads = std::set<uint64_t>{};
    auto    gpu_num_streams = std::set<uint64_t>{ 1 };
    auto    gpu_block_sizes = std::set<uint64_t>{ 32, 128, 256, 512, 1024 };
#if defined(TIMEMORY_USE_CUDA_HALF)
    auto gpu_ftypes = std::set<std::string>{ "fp16", "fp32", "fp64" };
#else
    auto gpu_ftypes = std::set<std::string>{ "fp32", "fp64" };
#endif

    auto _compute_num_threads = [](const std::set<uint64_t>& _v) -> std::set<uint64_t> {
        std::set<uint64_t> _c{};
        for(auto itr : _v)
        {
            auto entry = itr / dmp::size();
            if(entry > 0)
                _c.insert(entry);
            else
                _c.insert(itr);
        }
        return _c;
    };

    using parser_t = tim::argparse::argument_parser;
    parser_t parser("ex_ert");

    parser.enable_help();
    parser.add_argument({ "-n", "--num-iter" }, "Number of iterations")
        .count(1)
        .action([&](parser_t& p) { num_iter = p.get<size_t>("num-iter"); });
    parser.add_argument({ "-o", "--output" }, "Output filename")
        .count(1)
        .action([&](parser_t& p) { fname = p.get<std::string>("output"); });
    parser.add_argument({ "-t", "--num-threads" }, "Number of threads")
        .set_default(init_num_threads)
        .action([&](parser_t& p) {
            if(cpu_num_threads.empty())
                cpu_num_threads =
                    _compute_num_threads(p.get<std::set<uint64_t>>("num-threads"));
            if(gpu_num_threads.empty())
                gpu_num_threads =
                    _compute_num_threads(p.get<std::set<uint64_t>>("num-threads"));
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[CPU]" }, "");
    parser.add_argument({ "--cpu" }, "Run CPU ERT").max_count(1).action([&](parser_t& p) {
        enable_cpu = p.get<bool>("cpu");
    });
    parser.add_argument({ "--cpu-types" }, "CPU floating point types")
        .choices(cpu_ftypes)
        .max_count(cpu_ftypes.size())
        .min_count(1)
        .action(
            [&](parser_t& p) { cpu_ftypes = p.get<std::set<std::string>>("cpu-types"); });
    parser.add_argument({ "--cpu-num-threads" }, "CPU Number of threads")
        .action([&](parser_t& p) {
            cpu_num_threads =
                _compute_num_threads(p.get<std::set<uint64_t>>("cpu-num-threads"));
        });
    parser.add_argument()
        .names({ "--cpu-min-size" })
        .description("Starting data size for CPU ERT (in bytes)")
        .max_count(1)
        .dtype("bytes")
        .action([&](parser_t& p) {
            cpu_min_size = p.get<int64_t>("cpu-min-size") * units::byte;
        });
    parser.add_argument()
        .names({ "--cpu-max-size" })
        .description("Maximum data size for CPU ERT (in KB)")
        .max_count(1)
        .dtype("KB")
        .action([&](parser_t& p) {
            cpu_max_data = p.get<int64_t>("cpu-max-size") * units::kilobyte;
        });

    if(num_gpus > 0)
    {
        std::set<int64_t> num_gpus_choices = {};
        for(int64_t i = 0; i < num_gpus; ++i)
            num_gpus_choices.insert(i + 1);
        parser.add_argument({ "" }, "");
        parser.add_argument({ "[GPU]" }, "");
        parser.add_argument({ "--gpu" }, "Run GPU ERT")
            .max_count(1)
            .action([&](parser_t& p) { enable_gpu = p.get<bool>("cpu"); });
        parser.add_argument({ "--num-gpus" }, "Number of GPUs to use in ERT calculation")
            .max_count(1)
            .choices(num_gpus_choices)
            .action([&](parser_t& p) { num_gpus = p.get<int64_t>("num-gpu"); });
        parser.add_argument({ "--gpu-types" }, "GPU floating point types")
            .choices(gpu_ftypes)
            .max_count(gpu_ftypes.size())
            .min_count(1)
            .action([&](parser_t& p) {
                gpu_ftypes = p.get<std::set<std::string>>("gpu-types");
            });
        parser
            .add_argument({ "--gpu-num-threads" },
                          "Number of CPU threads launching kernels")
            .action([&](parser_t& p) {
                gpu_num_threads =
                    _compute_num_threads(p.get<std::set<uint64_t>>("gpu-num-threads"));
            });
        parser
            .add_argument({ "--gpu-num-streams" },
                          "Number of streams to use when launching kernels")
            .action([&](parser_t& p) {
                gpu_num_streams = p.get<std::set<uint64_t>>("gpu-num-streams");
            });
        parser
            .add_argument({ "--gpu-num-threads-per-block" },
                          "GPU number of threads per block in a kernel, e.g. block size")
            .action([&](parser_t& p) {
                gpu_block_sizes = p.get<std::set<uint64_t>>("gpu-num-threads-per-block");
            });
        parser.add_argument()
            .names({ "--gpu-min-size" })
            .description("Starting data size for GPU ERT (in KB)")
            .max_count(1)
            .dtype("KB")
            .action([&](parser_t& p) {
                gpu_min_size = p.get<int64_t>("gpu-min-size") * units::kilobyte;
            });
        parser.add_argument()
            .names({ "--gpu-max-size" })
            .description("Maximum data size for GPU ERT (in KB)")
            .max_count(1)
            .dtype("KB")
            .action([&](parser_t& p) {
                gpu_max_data = p.get<int64_t>("gpu-max-size") * units::kilobyte;
            });
    }

    auto err = parser.parse(argc, argv);
    if(err)
    {
        std::cerr << err << std::endl;
        parser.print_help("-- <CMD> <ARGS>");
        return EXIT_FAILURE;
    }

    if(parser.exists("h") || parser.exists("help"))
    {
        parser.print_help("-- <CMD> <ARGS>");
        return 0;
    }

    settings::verbose() = 0;
    dmp::initialize(argc, argv);
    tim::timemory_init(argc, argv, "");
    tim::signals::enable_signal_detection();

    auto data = std::make_shared<ert_data_t>();

    if(!enable_cpu && !enable_gpu)
    {
        if(num_gpus > 0)
            enable_gpu = true;
        else
            enable_cpu = true;
    }

    void* _timer = start_generic("run_ert");

    for(size_t i = 0; i < num_iter; ++i)
    {
        // execute the single-precision ERT calculations
        for(auto nthread : cpu_num_threads)
        {
            if(!enable_cpu)
                continue;
            if(cpu_ftypes.count("fp32") == 0)
                continue;
            run_ert<float, device::cpu>(data, nthread, cpu_min_size, cpu_max_data);
        }

        // execute the double-precision ERT calculations
        for(auto nthread : cpu_num_threads)
        {
            if(!enable_cpu)
                continue;
            if(cpu_ftypes.count("fp64") == 0)
                continue;
            run_ert<double, device::cpu>(data, nthread, cpu_min_size, cpu_max_data);
        }

        if(num_gpus > 0)
        {
            // execute the half-precision ERT calculations
            for(auto nthread : gpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!enable_gpu)
                            continue;
                        if(gpu_ftypes.count("fp16") == 0)
                            continue;
                        run_ert<fp16_t, device::gpu>(data, nthread, gpu_min_size,
                                                     gpu_max_data, nstream, block,
                                                     num_gpus);
                    }
            // execute the single-precision ERT calculations
            for(auto nthread : gpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!enable_gpu)
                            continue;
                        if(gpu_ftypes.count("fp32") == 0)
                            continue;
                        run_ert<float, device::gpu>(data, nthread, gpu_min_size,
                                                    gpu_max_data, nstream, block,
                                                    num_gpus);
                    }

            // execute the double-precision ERT calculations
            for(auto nthread : gpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!enable_gpu)
                            continue;
                        if(gpu_ftypes.count("fp64") == 0)
                            continue;
                        run_ert<double, device::gpu>(data, nthread, gpu_min_size,
                                                     gpu_max_data, nstream, block,
                                                     num_gpus);
                    }
        }
    }

    if(dmp::rank() == 0)
        printf("\n");
    ert::serialize(fname, *data);

    stop_generic(_timer);
    tim::timemory_finalize();
    dmp::finalize();
    return 0;
}

//======================================================================================//
