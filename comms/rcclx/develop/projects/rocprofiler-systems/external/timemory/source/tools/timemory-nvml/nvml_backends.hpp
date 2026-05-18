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

#include "timemory/components/base/types.hpp"
#include "timemory/math/compute.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/types.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <nvml.h>
#include <ostream>
#include <sys/types.h>
#include <unistd.h>

namespace tim
{
namespace backends
{
struct nvml_process_data
{
    using value_type = unsigned long long;
    using pid_map_t  = std::unordered_map<unsigned long, std::string>;

    static pid_map_t& get_pid_name_map()
    {
        static pid_map_t _instance{};
        return _instance;
    }

    static void add_pid_name(unsigned int _pid)
    {
        if(get_pid_name_map().count(_pid) == 0)
        {
            constexpr int buffer_size = 64;
            char          name[buffer_size + 1];
            memset(name, '\0', buffer_size + 1);
            auto ret = nvmlSystemGetProcessName(_pid, name, buffer_size);
            if(ret == NVML_SUCCESS)
                get_pid_name_map()[_pid] = std::string{ name };
        }
    }

    TIMEMORY_DEFAULT_OBJECT(nvml_process_data)

    nvml_process_data(const nvmlProcessInfo_t& rhs)
    : pid{ rhs.pid }
    , used_gpu_memory{ rhs.usedGpuMemory }
    {
        add_pid_name(pid);
    }

    nvml_process_data& operator=(const nvmlProcessInfo_t& rhs)
    {
        pid             = rhs.pid;
        used_gpu_memory = rhs.usedGpuMemory;
        add_pid_name(pid);
        return *this;
    }

    nvml_process_data& operator+=(const nvml_process_data& rhs)
    {
        assert(pid == rhs.pid);
        used_gpu_memory += rhs.used_gpu_memory;
        return *this;
    }

    nvml_process_data& operator-=(const nvml_process_data& rhs)
    {
        assert(pid == rhs.pid);
        used_gpu_memory -= rhs.used_gpu_memory;
        return *this;
    }

    template <typename Tp, enable_if_t<std::is_integral<Tp>::value> = 0>
    nvml_process_data& operator/=(Tp rhs)
    {
        assert(pid == rhs.pid);
        used_gpu_memory /= static_cast<value_type>(rhs);
        return *this;
    }

    template <typename Tp, enable_if_t<std::is_integral<Tp>::value> = 0>
    friend nvml_process_data operator/(nvml_process_data lhs, Tp rhs)
    {
        lhs.used_gpu_memory /= static_cast<value_type>(rhs);
        return lhs;
    }

    auto percent_diff(const nvml_process_data& rhs) const
    {
        auto lhs            = *this;
        lhs.used_gpu_memory = math::compute<value_type>::percent_diff(
            lhs.used_gpu_memory, rhs.used_gpu_memory);
        return lhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const nvml_process_data& rhs)
    {
        if(get_pid_name_map().count(rhs.pid) > 0)
            os << get_pid_name_map()[rhs.pid] << " :: " << rhs.used_gpu_memory;
        else
            os << rhs.pid << " :: " << rhs.used_gpu_memory;
        return os;
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int)
    {
        if(get_pid_name_map().count(pid) > 0)
            ar(cereal::make_nvp("pid", pid),
               cereal::make_nvp("name", get_pid_name_map()[pid]),
               cereal::make_nvp("used_gpu_memory", used_gpu_memory));
        else
            ar(cereal::make_nvp("pid", pid),
               cereal::make_nvp("used_gpu_memory", used_gpu_memory));
    }

private:
    unsigned int pid             = 0;
    value_type   used_gpu_memory = 0;
};

struct nvml_memory_info_data
{
    using value_type = unsigned long long;
    using array_type = std::array<value_type, 3>;

    TIMEMORY_DEFAULT_OBJECT(nvml_memory_info_data)

    nvml_memory_info_data(const nvmlMemory_t& rhs)
    : m_data{ rhs.free, rhs.total, rhs.used }
    {}

    nvml_memory_info_data& operator=(const nvmlMemory_t& rhs)
    {
        m_data = { rhs.free, rhs.total, rhs.used };
        return *this;
    }

    nvml_memory_info_data& operator+=(const nvml_memory_info_data& rhs)
    {
        using namespace tim::component::operators;
        m_data += rhs.m_data;
        return *this;
    }

    nvml_memory_info_data& operator-=(const nvml_memory_info_data& rhs)
    {
        using namespace tim::component::operators;
        m_data -= rhs.m_data;
        return *this;
    }

    template <typename Tp, enable_if_t<std::is_integral<Tp>::value> = 0>
    nvml_memory_info_data& operator/=(Tp rhs)
    {
        using namespace tim::component::operators;
        m_data /= rhs;
        return *this;
    }

    auto percent_diff(const nvml_memory_info_data& rhs) const
    {
        auto lhs   = *this;
        lhs.m_data = math::compute<array_type>::percent_diff(lhs.m_data, rhs.m_data);
        return lhs;
    }

    template <typename Tp, enable_if_t<std::is_integral<Tp>::value> = 0>
    friend nvml_memory_info_data operator/(nvml_memory_info_data lhs, Tp rhs)
    {
        using namespace tim::component::operators;
        lhs.m_data /= static_cast<value_type>(rhs);
        return lhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const nvml_memory_info_data& rhs)
    {
        os << "free: " << rhs.free() << ", used: " << rhs.used()
           << ", total: " << rhs.total();
        return os;
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int)
    {
        ar(cereal::make_nvp("free", free()), cereal::make_nvp("total", total()),
           cereal::make_nvp("used", used()));
    }

    array_type  data() const { return m_data; }
    array_type& data() { return m_data; }

    value_type  free() const { return m_data.at(0); }
    value_type  total() const { return m_data.at(1); }
    value_type  used() const { return m_data.at(2); }
    value_type& free() { return m_data.at(0); }
    value_type& total() { return m_data.at(1); }
    value_type& used() { return m_data.at(2); }

private:
    array_type m_data{};
};

struct nvml_utilization_rate_data
{
    using value_type = unsigned int;
    using array_type = std::array<value_type, 2>;

    TIMEMORY_DEFAULT_OBJECT(nvml_utilization_rate_data)

    nvml_utilization_rate_data(const nvmlUtilization_t& rhs)
    : m_data{ rhs.gpu, rhs.memory }
    {}

    nvml_utilization_rate_data& operator=(const nvmlUtilization_t& rhs)
    {
        m_data = { rhs.gpu, rhs.memory };
        return *this;
    }

    nvml_utilization_rate_data& operator+=(const nvml_utilization_rate_data& rhs)
    {
        using namespace tim::component::operators;
        m_data += rhs.m_data;
        return *this;
    }

    nvml_utilization_rate_data& operator-=(const nvml_utilization_rate_data& rhs)
    {
        using namespace tim::component::operators;
        m_data -= rhs.m_data;
        return *this;
    }

    template <typename Tp, enable_if_t<std::is_integral<Tp>::value> = 0>
    nvml_utilization_rate_data& operator/=(Tp rhs)
    {
        using namespace tim::component::operators;
        m_data /= rhs;
        return *this;
    }

    auto percent_diff(const nvml_utilization_rate_data& rhs) const
    {
        auto lhs   = *this;
        lhs.m_data = math::compute<array_type>::percent_diff(lhs.m_data, rhs.m_data);
        return lhs;
    }

    template <typename Tp, enable_if_t<std::is_integral<Tp>::value> = 0>
    friend nvml_utilization_rate_data operator/(nvml_utilization_rate_data lhs, Tp rhs)
    {
        using namespace tim::component::operators;
        lhs.m_data /= static_cast<value_type>(rhs);
        return lhs;
    }

    friend std::ostream& operator<<(std::ostream&                     os,
                                    const nvml_utilization_rate_data& rhs)
    {
        os << "gpu utilization: " << rhs.gpu()
           << "%, memory utilization: " << rhs.memory() << "%";
        return os;
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned int)
    {
        ar(cereal::make_nvp("gpu_utilization", gpu()),
           cereal::make_nvp("memory_utilization", memory()));
    }

    array_type  data() const { return m_data; }
    array_type& data() { return m_data; }

    value_type  gpu() const { return m_data.at(0); }
    value_type  memory() const { return m_data.at(1); }
    value_type& gpu() { return m_data.at(0); }
    value_type& memory() { return m_data.at(1); }

private:
    array_type m_data{};
};

}  // namespace backends
}  // namespace tim
