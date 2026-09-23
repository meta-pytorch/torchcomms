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

#pragma once

#include "timemory/general.hpp"
#include "timemory/sampling.hpp"
#include "timemory/timemory.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <nvml.h>
#include <thread>
#include <vector>

template <typename Tp>
using vector_t       = std::vector<Tp>;
using stringstream_t = std::stringstream;

using namespace tim::component;

struct nvml_device_info
{
    int               index = -1;
    nvmlDevice_t      device;
    char              name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlPciInfo_t     pci;
    nvmlComputeMode_t compute_mode;
};

struct nvml_config
{
    bool   debug           = tim::get_env("TIMEMORY_NVML_DEBUG", false);
    bool   finished        = false;
    int    verbose         = tim::get_env("TIMEMORY_NVML_VERBOSE", 0);
    size_t buffer_count    = tim::get_env<size_t>("TIMEMORY_NVML_BUFFER_COUNT", 1000);
    size_t max_samples     = tim::get_env<double>("TIMEMORY_NVML_MAX_SAMPLES", 0);
    size_t dump_interval   = tim::get_env<size_t>("TIMEMORY_NVML_DUMP_INTERVAL", 1000);
    double sample_interval = tim::get_env<double>("TIMEMORY_NVML_SAMPLE_INTERVAL", 1.0);
    std::string output_file =
        tim::get_env<std::string>("TIMEMORY_NVML_OUTPUT", "timemory-nvml-output");
    std::string time_format =
        tim::get_env<std::string>("TIMEMORY_NVML_TIME_FORMAT", "%F_%I:%M:%S_%p");
    std::vector<std::string> argvector = {};

    template <typename Archive>
    void serialize(Archive& ar, unsigned int);

    std::string get_output_filename(std::string inp = {});
};

nvml_config&
get_config();

void
parse_args_and_configure(int&, char**&);

void
monitor(const std::vector<nvml_device_info>&);

template <typename Archive>
void
nvml_config::serialize(Archive& ar, unsigned int)
{
    ar(tim::cereal::make_nvp("debug", debug), tim::cereal::make_nvp("verbose", verbose),
       tim::cereal::make_nvp("buffer_count", buffer_count),
       tim::cereal::make_nvp("max_samples", max_samples),
       tim::cereal::make_nvp("dump_interval", dump_interval),
       tim::cereal::make_nvp("sample_interval", sample_interval),
       tim::cereal::make_nvp("time_format", time_format),
       tim::cereal::make_nvp("argvector", argvector));
}

#define NVML_CONFIG_FUNCTION(NAME)                                                       \
    inline auto& NAME() { return get_config().NAME; }

NVML_CONFIG_FUNCTION(debug)
NVML_CONFIG_FUNCTION(finished)
NVML_CONFIG_FUNCTION(verbose)
NVML_CONFIG_FUNCTION(buffer_count)
NVML_CONFIG_FUNCTION(max_samples)
NVML_CONFIG_FUNCTION(dump_interval)
NVML_CONFIG_FUNCTION(sample_interval)
NVML_CONFIG_FUNCTION(output_file)
NVML_CONFIG_FUNCTION(time_format)
NVML_CONFIG_FUNCTION(argvector)
