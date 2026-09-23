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

#include "nvml.hpp"

#include "timemory/general/serialization.hpp"
#include "timemory/mpl/policy.hpp"

#include <fstream>

//--------------------------------------------------------------------------------------//

int
execute(std::vector<nvml_device_info>&, int argc, char** argv);

int
main(int argc, char** argv)
{
    parse_args_and_configure(argc, argv);

    int                           ec           = 0;
    unsigned int                  device_count = 0;
    std::vector<nvml_device_info> unit_device_vec{};

    TIMEMORY_NVML_RUNTIME_CHECK_ERROR(nvmlInit_v2(), goto Error)
    TIMEMORY_NVML_RUNTIME_CHECK_ERROR(nvmlDeviceGetCount(&device_count), goto Error);

    unit_device_vec.resize(device_count);
    for(unsigned int i = 0; i < device_count; ++i)
    {
        auto& device       = unit_device_vec.at(i).device;
        auto& name         = unit_device_vec.at(i).name;
        auto& pci          = unit_device_vec.at(i).pci;
        auto& compute_mode = unit_device_vec.at(i).compute_mode;

        auto result = nvmlDeviceGetHandleByIndex(i, &device);
        if(NVML_SUCCESS != result)
        {
            fprintf(stderr, "Failed to get handle for device %u: %s\n", i,
                    nvmlErrorString(result));
            goto Error;
        }

        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if(NVML_SUCCESS != result)
        {
            fprintf(stderr, "Failed to get name of device %u: %s\n", i,
                    nvmlErrorString(result));
            goto Error;
        }

        result = nvmlDeviceGetPciInfo(device, &pci);
        if(NVML_SUCCESS != result)
        {
            fprintf(stderr, "Failed to get pci info for device %u: %s\n", i,
                    nvmlErrorString(result));
            goto Error;
        }

        printf("%u. %s [%s]\n", i, name, pci.busId);

        result = nvmlDeviceGetComputeMode(device, &compute_mode);
        if(NVML_ERROR_NOT_SUPPORTED == result)
        {
            fprintf(stderr, "\t This is not CUDA capable device\n");
        }
        else if(NVML_SUCCESS != result)
        {
            fprintf(stderr, "Failed to get compute mode for device %u: %s\n", i,
                    nvmlErrorString(result));
            goto Error;
        }
    }

    for(size_t i = 0; i < unit_device_vec.size(); ++i)
        unit_device_vec.at(i).index = static_cast<int>(i);

    ec = execute(unit_device_vec, argc, argv);

    TIMEMORY_NVML_RUNTIME_CHECK_ERROR(nvmlShutdown(), {})
    tim::timemory_finalize();
    return ec;

Error:
    auto result = nvmlShutdown();
    if(NVML_SUCCESS != result)
        fprintf(stderr, "Failed to shutdown NVML: %s\n", nvmlErrorString(result));

    tim::timemory_finalize();
    fprintf(stderr, "Press ENTER to continue...\n");
    getchar();
    return EXIT_FAILURE;
}

//--------------------------------------------------------------------------------------//

int
execute(std::vector<nvml_device_info>& unit_device_vec, int argc, char** argv)
{
    if(argc == 1)
    {
        monitor(unit_device_vec);
        return EXIT_SUCCESS;
    }

    int  ec       = 0;
    auto _execute = [&ec, argc, argv]() {
        std::vector<char*> _argv(argc, nullptr);
        argvector().clear();
        std::ostringstream _cmdstring{};
        for(int i = 1; i < argc; ++i)
        {
            argvector().emplace_back(argv[i]);
            _argv.at(i - 1) = argv[i];
            _cmdstring << " " << argvector().back();
        }
        auto _cmd = tim::popen::popen(_argv.at(0), _argv.data());
        tim::popen::flush_output(std::cout, _cmd);
        ec = tim::popen::pclose(_cmd);
        if(ec != 0)
            perror("Error in timemory_fork");
        finished() = true;
    };

    std::thread _t{ _execute };
    monitor(unit_device_vec);
    _t.join();
    return ec;
}

//--------------------------------------------------------------------------------------//
