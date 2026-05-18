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

#include "timemory/utility/macros.hpp"

#include <cstddef>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

namespace tim
{
namespace procfs
{
namespace cpuinfo
{
struct freq
{
    TIMEMORY_DEFAULT_OBJECT(freq)

    auto     operator()(size_t _idx) const;
    explicit operator bool() const;

    static double   get(size_t _idx);
    static size_t   size() { return std::thread::hardware_concurrency(); }
    static auto&    get_ifstream()
    {
        static thread_local auto _v = []() {
            auto _ifs =
                std::make_unique<std::ifstream>("/proc/cpuinfo", std::ifstream::binary);
            if(!_ifs->is_open())
                _ifs = std::unique_ptr<std::ifstream>{};
            return _ifs;
        }();
        return _v;
    }
};

inline freq::operator bool() const { return (get_ifstream() != nullptr); }

inline double
freq::get(size_t _idx)
{
    std::ifstream ifs("/proc/cpuinfo");
    if(!ifs) return 0.0;

    std::string line;
    size_t current_cpu = 0;

    while(std::getline(ifs, line))
    {
        if(line.find("processor") == 0)
        {
            size_t idx;
            if(sscanf(line.c_str(), "processor : %zu", &idx) == 1)
            {
                current_cpu = idx;
            }
        }
        if(current_cpu == _idx && line.find("cpu MHz") == 0)
        {
            double freq = 0.0;
            size_t pos = line.find(':');
            if(pos != std::string::npos)
            {
                std::string value = line.substr(pos + 1);
                freq = std::stod(value);
                return freq;
            }
        }
    }

    return 0.0;
}

inline auto
freq::operator()(size_t _idx) const
{
    return freq::get(_idx % size());
}

}  // namespace cpuinfo
}  // namespace procfs
}  // namespace tim