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

#ifndef TIMEMORY_COMPONENTS_CUPTI_CUPTI_PCSAMPLING_HPP_
#    define TIMEMORY_COMPONENTS_CUPTI_CUPTI_PCSAMPLING_HPP_ 1
#endif

#include "timemory/components/base.hpp"
#include "timemory/components/cupti/backends.hpp"
#include "timemory/components/cupti/types.hpp"
#include "timemory/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/variadic/lightweight_tuple.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"

#    include <cuda.h>
#    include <cupti.h>
#    if defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
#        include <cupti_pcsampling.h>
#    endif
#endif

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//              CUPTI Program Counter (PC) sampling component
//
//--------------------------------------------------------------------------------------//
/// \struct tim::component::cupti_pcsampling
/// \brief The PC Sampling gives the number of samples for each source and assembly
/// line with various stall reasons. Using this information, you can pinpoint portions
/// of your kernel that are introducing latencies and the reason for the latency.
///
struct cupti_pcsampling
: public base<cupti_pcsampling, cupti::pcsample>
, private policy::instance_tracker<cupti_pcsampling, false>
{
    // component-specific aliases
    using data_type    = cupti::pcdata;
    using size_type    = std::size_t;
    using tracker_type = policy::instance_tracker<cupti_pcsampling, false>;
#if defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
    using config_type =
        std::tuple<CUcontext, CUpti_PCSamplingEnableParams,
                   CUpti_PCSamplingGetNumStallReasonsParams,
                   CUpti_PCSamplingGetStallReasonsParams, CUpti_PCSamplingData,
                   std::vector<CUpti_PCSamplingConfigurationInfo>,
                   CUpti_PCSamplingConfigurationInfoParams, CUpti_PCSamplingStartParams,
                   CUpti_PCSamplingStopParams, size_t, size_t>;
#else
    using config_type = std::tuple<null_type, null_type, null_type, null_type, null_type,
                                   std::vector<null_type>, null_type, null_type,
                                   null_type, size_t, size_t>;
#endif

    // required aliases
    using value_type = cupti::pcsample;
    using this_type  = cupti_pcsampling;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "cupti_pcsampling"; }
    static std::string description() { return "CUpti Program Counter (PC) Sampling API"; }
    static void        global_init() { initialize(); }
    static void        global_finalize() { finalize(); }
    static data_type   record();
    static void        sample();

    TIMEMORY_DEFAULT_OBJECT(cupti_pcsampling)

    void        store(const value_type& _data);
    void        store(value_type&& _data);
    void        start();
    void        stop();
    void        set_started();
    void        set_stopped();
    std::string get_display() const;

    auto get_laps() const { return value.totalSamples; }

    std::vector<int64_t>            get() const;
    static std::vector<std::string> label_array();

    static void cleanup() { cupti::pcstall::allocate_arrays(0); }

#if defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
    static CUpti_PCSamplingData get_pcsampling_data(size_t numStallReasons,
                                                    size_t numPcsToCollect);
    static void                 free_pcsampling_data(CUpti_PCSamplingData);
#endif

protected:
    const char* m_prefix = nullptr;

public:
    static config_type configure();
    static void        initialize();
    static void        finalize();
    static auto&       get_configuration_data()
    {
        static persistent_data _instance{};
        return _instance;
    }

private:
    static std::unordered_set<cupti_pcsampling*>& get_stack()
    {
        static thread_local std::unordered_set<cupti_pcsampling*> _instance{};
        return _instance;
    }

private:
    struct persistent_data
    {
        bool        enabled       = false;
        bool        region_totals = true;
        config_type data          = {};

        // create ref and cr member functions e.g.:
        //  auto&       context()       { return get<0>(data); }
        //  const auto& context() const { return get<0>(data); }
        TIMEMORY_TUPLE_ACCESSOR(0, data, context)
        TIMEMORY_TUPLE_ACCESSOR(1, data, enable_params)
        TIMEMORY_TUPLE_ACCESSOR(2, data, num_stall_reasons_params)
        TIMEMORY_TUPLE_ACCESSOR(3, data, stall_reasons_params)
        TIMEMORY_TUPLE_ACCESSOR(4, data, sampling_data)
        TIMEMORY_TUPLE_ACCESSOR(5, data, info)
        TIMEMORY_TUPLE_ACCESSOR(6, data, info_params)
        TIMEMORY_TUPLE_ACCESSOR(7, data, start_params)
        TIMEMORY_TUPLE_ACCESSOR(8, data, stop_params)
        TIMEMORY_TUPLE_ACCESSOR(9, data, num_stall_reasons)
        TIMEMORY_TUPLE_ACCESSOR(10, data, num_collect_pcs)
    };
};
//
}  // namespace component
//
//--------------------------------------------------------------------------------------//
//
namespace cupti
{
//
template <typename Archive>
void
pcsample::save(Archive& ar, const unsigned int) const
{
    std::string _fname = functionName;
    ar(cereal::make_nvp("samples", totalSamples), cereal::make_nvp("cubin_id", cubinCrc),
       cereal::make_nvp("pc_offset", pcOffset),
       cereal::make_nvp("func_index", functionIndex),
       cereal::make_nvp("func_name", _fname), cereal::make_nvp("stalls", stalls));
}
//
template <typename Archive>
void
pcsample::load(Archive& ar, const unsigned int)
{
    // memory leak
    auto* _fname = new std::string{};
    ar(cereal::make_nvp("samples", totalSamples), cereal::make_nvp("cubin_id", cubinCrc),
       cereal::make_nvp("pc_offset", pcOffset),
       cereal::make_nvp("func_index", functionIndex),
       cereal::make_nvp("func_name", *_fname), cereal::make_nvp("stalls", stalls));
    functionName = _fname->c_str();
}
//
template <typename Archive>
inline void
pcstall::save(Archive& ar, const unsigned int) const
{
    auto        _idx     = index;
    auto        _samples = samples;
    std::string _name    = name();
    ar(cereal::make_nvp("index", _idx), cereal::make_nvp("name", _name),
       cereal::make_nvp("samples", _samples));
}
//
template <typename Archive>
inline void
pcstall::load(Archive& ar, const unsigned int)
{
    auto _idx     = index;
    auto _samples = samples;
    ar(cereal::make_nvp("index", _idx), cereal::make_nvp("samples", _samples));
}
//
}  // namespace cupti
}  // namespace tim

// #endif  // TIMEMORY_USE_CUPTI_PCSAMPLING

#if defined(TIMEMORY_CUPTI_COMPONENT_HEADER_MODE)
#    include "timemory/components/cupti/cupti_pcsampling.cpp"
#endif
