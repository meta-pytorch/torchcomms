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

/** \file timemory/ert/data.hpp
 * \headerfile timemory/ert/data.hpp "timemory/ert/data.hpp"
 * Provides execution data for ERT
 *
 */

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/components/cuda/backends.hpp"
#include "timemory/components/timing/ert_timer.hpp"
#include "timemory/ert/aligned_allocator.hpp"
#include "timemory/ert/barrier.hpp"
#include "timemory/ert/cache_size.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tim
{
namespace ert
{
using std::size_t;
using namespace std::placeholders;

//--------------------------------------------------------------------------------------//
//  execution params
//
struct exec_params
{
    explicit exec_params(uint64_t _work_set = 16,
                         uint64_t mem_max   = 8 * cache_size::get_max(),
                         uint64_t _nthread = 1, uint64_t _nstream = 1,
                         uint64_t _grid_size = 0, uint64_t _block_size = 32)
    : working_set_min(_work_set)
    , memory_max(mem_max)
    , nthreads(_nthread)
    , nstreams(_nstream)
    , grid_size(_grid_size)
    , block_size(_block_size)
    {}

    ~exec_params()                      = default;
    exec_params(const exec_params&)     = default;
    exec_params(exec_params&&) noexcept = default;
    exec_params& operator=(const exec_params&) = default;
    exec_params& operator=(exec_params&&) noexcept = default;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("working_set_min", working_set_min),
           cereal::make_nvp("memory_max", memory_max),
           cereal::make_nvp("nthreads", nthreads), cereal::make_nvp("nrank", nrank),
           cereal::make_nvp("nproc", nproc), cereal::make_nvp("nstreams", nstreams),
           cereal::make_nvp("grid_size", grid_size),
           cereal::make_nvp("block_size", block_size),
           cereal::make_nvp("shmem_size", shmem_size));
    }

    friend std::ostream& operator<<(std::ostream& os, const exec_params& obj)
    {
        std::stringstream ss;
        ss << "working_set_min = " << obj.working_set_min << ", "
           << "memory_max = " << obj.memory_max << ", "
           << "nthreads = " << obj.nthreads << ", "
           << "nrank = " << obj.nrank << ", "
           << "nproc = " << obj.nproc << ", "
           << "nstreams = " << obj.nstreams << ", "
           << "grid_size = " << obj.grid_size << ", "
           << "block_size = " << obj.block_size << ", "
           << "shmem_size = " << obj.shmem_size;
        os << ss.str();
        return os;
    }

    uint64_t working_set_min = 16;                         // NOLINT NOLINTNEXTLINE
    uint64_t memory_max      = 8 * cache_size::get_max();  // default is 8 * L3 cache size
    uint64_t nthreads        = 1;                          // NOLINT
    uint64_t nrank           = tim::dmp::rank();           // NOLINT
    uint64_t nproc           = tim::dmp::size();           // NOLINT
    uint64_t nstreams        = 1;                          // NOLINT
    uint64_t grid_size       = 0;                          // NOLINT
    uint64_t block_size      = 32;                         // NOLINT
    uint64_t shmem_size      = 0;                          // NOLINT
};

//--------------------------------------------------------------------------------------//
//  execution data -- reuse this for multiple types
//
template <typename Tp>
class exec_data
{
public:
    using value_type     = std::tuple<std::string, uint64_t, uint64_t, uint64_t, uint64_t,
                                  uint64_t, Tp, std::string, std::string, exec_params>;
    using labels_type    = std::array<string_t, std::tuple_size<value_type>::value>;
    using value_array    = std::vector<value_type>;
    using size_type      = typename value_array::size_type;
    using iterator       = typename value_array::iterator;
    using const_iterator = typename value_array::const_iterator;
    using this_type      = exec_data<Tp>;

    //----------------------------------------------------------------------------------//
    //
    exec_data()                     = default;
    ~exec_data()                    = default;
    exec_data(const exec_data&)     = delete;
    exec_data(exec_data&&) noexcept = default;

    exec_data& operator=(const exec_data&) = delete;
    exec_data& operator=(exec_data&&) noexcept = default;

public:
    //----------------------------------------------------------------------------------//
    //
    void set_labels(const labels_type& _labels) { m_labels = _labels; }
    TIMEMORY_NODISCARD labels_type    get_labels() const { return m_labels; }
    size_type                         size() { return m_values.size(); }
    iterator                          begin() { return m_values.begin(); }
    TIMEMORY_NODISCARD const_iterator begin() const { return m_values.begin(); }
    iterator                          end() { return m_values.end(); }
    TIMEMORY_NODISCARD const_iterator end() const { return m_values.end(); }

public:
    //----------------------------------------------------------------------------------//
    //
    exec_data& operator+=(const value_type& entry)
    {
        m_values.resize(m_values.size() + 1);
        m_values.back() = entry;
        // m_values.push_back(entry);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    //
    exec_data& operator+=(const exec_data& rhs)
    {
        for(const auto& itr : rhs.m_values)
            m_values.push_back(itr);
        return *this;
    }

public:
    //----------------------------------------------------------------------------------//
    //
    friend std::ostream& operator<<(std::ostream& os, const exec_data& obj)
    {
        std::stringstream ss;
        for(const auto& itr : obj.m_values)
        {
            ss << std::setw(24) << std::get<0>(itr) << " (device: " << std::get<7>(itr)
               << ", dtype = " << std::get<8>(itr) << "): ";
            obj.write<1>(ss, itr, ", ", 10);
            obj.write<2>(ss, itr, ", ", 6);
            obj.write<3>(ss, itr, ", ", 12);
            obj.write<4>(ss, itr, ", ", 12);
            obj.write<5>(ss, itr, ", ", 12);
            obj.write<6>(ss, itr, "\n", 12);
        }
        os << ss.str();
        return os;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        constexpr auto sz = std::tuple_size<value_type>::value;
        ar(cereal::make_nvp("entries", m_values.size()));

        ar.setNextName("ert");
        ar.startNode();
        ar.makeArray();
        for(const auto& itr : m_values)
        {
            ar.startNode();
            _save(ar, itr, make_index_sequence<sz>{});
            ar.finishNode();
        }
        ar.finishNode();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        constexpr auto sz    = std::tuple_size<value_type>::value;
        auto           _size = 0;
        ar(cereal::make_nvp("entries", _size));
        m_values.resize(_size);

        ar.setNextName("ert");
        ar.startNode();
        for(auto& itr : m_values)
        {
            ar.startNode();
            _load(ar, itr, make_index_sequence<sz>{});
            ar.finishNode();
        }
        ar.finishNode();
    }

private:
    labels_type m_labels = { { "label", "working-set", "trials", "total-bytes",
                               "total-ops", "ops-per-set", "counter", "device", "dtype",
                               "exec-params" } };
    value_array m_values = {};

private:
    //----------------------------------------------------------------------------------//
    //
    template <size_t N>
    void write(std::ostream& os, const value_type& ret, const string_t& _trailing,
               int32_t _width) const
    {
        os << std::setw(10) << std::get<N>(m_labels) << " = " << std::setw(_width)
           << std::get<N>(ret) << _trailing;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive, size_t... Idx>
    void _save(Archive& ar, const value_type& _tuple, index_sequence<Idx...>) const
    {
        ar(cereal::make_nvp(std::get<Idx>(m_labels), std::get<Idx>(_tuple))...);
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive, size_t... Idx>
    void _load(Archive& ar, value_type& _tuple, index_sequence<Idx...>)
    {
        ar(cereal::make_nvp(std::get<Idx>(m_labels), std::get<Idx>(_tuple))...);
    }
};

//--------------------------------------------------------------------------------------//
//
//      CPU -- initialize buffer
//
//--------------------------------------------------------------------------------------//

template <typename DeviceT, typename Tp, typename Intp = int32_t,
          device::enable_if_cpu_t<DeviceT> = 0>
void
initialize_buffer(Tp* A, const Tp& value, const Intp& nsize)
{
    auto range = device::grid_strided_range<DeviceT, 0, Intp>(nsize);
    for(auto i = range.begin(); i < range.end(); i += range.stride())
        A[i] = value;
}

//--------------------------------------------------------------------------------------//
//
//      GPU -- initialize buffer
//
//--------------------------------------------------------------------------------------//

template <typename DeviceT, typename Tp, typename Intp = int32_t,
          device::enable_if_gpu_t<DeviceT> = 0>
TIMEMORY_GLOBAL_FUNCTION void
initialize_buffer(Tp* A, Tp value, Intp nsize)
{
    auto range = device::grid_strided_range<DeviceT, 0, Intp>(nsize);
    for(auto i = range.begin(); i < range.end(); i += range.stride())
        A[i] = value;
}

//--------------------------------------------------------------------------------------//

}  // namespace ert
}  // namespace tim
