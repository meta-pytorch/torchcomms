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

#include "timemory/components/base.hpp"
#include "timemory/components/papi/backends.hpp"
#include "timemory/components/papi/macros.hpp"
#include "timemory/components/papi/papi_common.hpp"
#include "timemory/components/papi/papi_config.hpp"
#include "timemory/components/papi/types.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef TIMEMORY_COMPONENTS_PAPI_PAPI_VECTOR_HPP_
#    define TIMEMORY_COMPONENTS_PAPI_PAPI_VECTOR_HPP_ 1

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                          Array of PAPI counters
//
//--------------------------------------------------------------------------------------//
//
struct papi_vector
: public base<papi_vector, std::vector<long long>>
, private papi_common<void>
{
    template <typename... T>
    friend struct cpu_roofline;

    template <typename... T>
    friend struct gpu_roofline;

    using size_type    = size_t;
    using event_list   = std::vector<int>;
    using value_type   = std::vector<long long>;
    using entry_type   = typename value_type::value_type;
    using this_type    = papi_vector;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using common_type  = papi_common<void>;

    static constexpr short precision = 3;
    static constexpr short width     = 8;

    static auto& get_initializer()
    {
        static auto _v = common_type::initializer_t{};
        return _v;
    }

    static void configure(papi_config* = common_type::get_config());
    static void initialize(papi_config* _cfg = common_type::get_config());
    static void shutdown(papi_config* = common_type::get_config());

    static void thread_init();
    static void thread_finalize();

    static std::string label();
    static std::string description();

    papi_vector();
    papi_vector(papi_config*);
    ~papi_vector()                      = default;
    papi_vector(const papi_vector&)     = default;
    papi_vector(papi_vector&&) noexcept = default;
    papi_vector& operator=(const papi_vector&) = default;
    papi_vector& operator=(papi_vector&&) noexcept = default;
    papi_vector& operator+=(const papi_vector& rhs);
    papi_vector& operator-=(const papi_vector& rhs);

    size_t                   size() const;
    value_type               record();
    void                     sample();
    void                     start();
    void                     stop();
    entry_type               get_display(int evt_type) const;
    std::ostream&            write(std::ostream&) const;
    std::vector<std::string> label_array() const;
    std::vector<std::string> description_array() const;
    std::vector<std::string> display_unit_array() const;
    std::vector<int64_t>     unit_array() const;
    std::string              get_display() const;

    using base_type::load;

    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum));
    }

    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        auto _data = get<double>();
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("repr_data", _data),
           cereal::make_nvp("value", value), cereal::make_nvp("accum", accum),
           cereal::make_nvp("display", _data));
    }

    template <typename Tp = double>
    std::vector<Tp> get() const
    {
        auto        values = std::vector<Tp>{};
        const auto& _data  = load();
        values.reserve(_data.size());
        for(const auto& itr : _data)
            values.emplace_back(itr);
        values.resize(size(), Tp{ 0 });
        return values;
    }

    friend std::ostream& operator<<(std::ostream& os, const papi_vector& obj)
    {
        return obj.write(os);
    }

protected:
    using common_type::m_thr;
    papi_config* m_config           = common_type::get_config();
    bool         m_config_is_common = true;
};

}  // namespace component
}  // namespace tim

#endif

#if defined(TIMEMORY_PAPI_COMPONENT_HEADER_MODE) &&                                      \
    TIMEMORY_PAPI_COMPONENT_HEADER_MODE > 0
#    include "timemory/components/papi/papi_vector.cpp"
#endif
