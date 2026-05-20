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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "timemory/backends/types/cupti.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/backends/cupti.hpp"

#    include <cuda.h>
#    include <cupti.h>
#endif

#if defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
#    include <cupti_pcsampling.h>
#endif

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

// for disabling a category at runtime
// no statistics ever
//
#if !defined(TIMEMORY_CUPTI_API_CALL)
#    define TIMEMORY_CUPTI_API_CALL(...) TIMEMORY_CUPTI_CALL(__VA_ARGS__)
#endif

#if !defined(ACTIVITY_RECORD_ALIGNMENT)
#    define ACTIVITY_RECORD_ALIGNMENT 8
#endif

#if !defined(PACKED_ALIGNMENT)
#    if defined(_WIN32)  // Windows 32- and 64-bit
#        define PACKED_ALIGNMENT __declspec(align(ACTIVITY_RECORD_ALIGNMENT))
#    elif defined(__GNUC__)  // GCC
#        define PACKED_ALIGNMENT                                                         \
            __attribute__((__packed__))                                                  \
                __attribute__((aligned(ACTIVITY_RECORD_ALIGNMENT)))
#    else  // all other compilers
#        define PACKED_ALIGNMENT
#    endif
#endif

#if !defined(TIMEMORY_CUPTI_CALLOC)
#    define TIMEMORY_CUPTI_CALLOC(TYPE, SIZE)                                            \
        (SIZE == 0) ? nullptr : static_cast<TYPE*>(calloc(SIZE, sizeof(TYPE)))
#endif

#if !defined(CUPTI_STALL_REASON_STRING_SIZE)
#    define CUPTI_STALL_REASON_STRING_SIZE 128
#endif

#if !defined(TIMEMORY_CUPTIAPI)
#    if defined(CUPTIAPI)
#        define TIMEMORY_CUPTIAPI CUPTIAPI
#    else
#        define TIMEMORY_CUPTIAPI
#    endif
#endif

#if !defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
extern "C"
{
    struct PACKED_ALIGNMENT CUpti_PCSamplingStallReason_t;
    struct PACKED_ALIGNMENT CUpti_PCSamplingPCData_t;
    struct PACKED_ALIGNMENT CUpti_PCSamplingData_t;
}
#endif

//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace cupti
{
#if !defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
struct CUpti_PCSamplingStallReason_t
{};
struct CUpti_PCSamplingPCData_t
{};
struct CUpti_PCSamplingData_t
{};
#else
using CUpti_PCSamplingStallReason_t = CUpti_PCSamplingStallReason;
using CUpti_PCSamplingPCData_t      = CUpti_PCSamplingPCData;
using CUpti_PCSamplingData_t        = CUpti_PCSamplingData;
#endif

/// \struct tim::component::cupti::pcstall
/// \brief Timemory's version of CUpti_PCSamplingStallReason
///
struct PACKED_ALIGNMENT pcstall
{
    // Collected stall reason index
    uint32_t index = 0;
    // Number of times the PC was sampled with the stallReason.
    uint32_t samples = 0;

    TIMEMORY_DEFAULT_OBJECT(pcstall)

    pcstall(const CUpti_PCSamplingStallReason_t&);
    pcstall(uint32_t _index, uint32_t _samples);

    pcstall& operator+=(const pcstall& rhs);
    pcstall& operator-=(const pcstall& rhs);

    bool operator==(const pcstall& rhs) const { return index == rhs.index; }
    bool operator!=(const pcstall& rhs) const { return !(*this == rhs); }
    bool operator<(const pcstall& rhs) const { return index < rhs.index; }
    bool operator>(const pcstall& rhs) const { return index > rhs.index; }
    bool operator<=(const pcstall& rhs) const { return index <= rhs.index; }
    bool operator>=(const pcstall& rhs) const { return index >= rhs.index; }

    const char*        name() const { return name(index); }
    static const char* name(uint32_t index);
    static bool        enabled(uint32_t index);

    // serialization
    template <typename Archive>
    void save(Archive& ar, const unsigned int) const;
    template <typename Archive>
    void load(Archive& ar, const unsigned int);

    friend std::ostream& operator<<(std::ostream& os, const pcstall& obj)
    {
        os << obj.name() << "=" << obj.samples;
        return os;
    }

    static char**& get_name_array()
    {
        static char** _instance = nullptr;
        return _instance;
    }

    static uint32_t*& get_index_array()
    {
        static uint32_t* _instance = nullptr;
        return _instance;
    }

    static bool*& get_bool_array()
    {
        static bool* _instance = nullptr;
        return _instance;
    }

    static size_t& get_size()
    {
        static size_t _instance = 0;
        return _instance;
    }

    static void allocate_arrays(size_t _n)
    {
        // last data size allocated
        size_t _last_n = get_size();

        // references to arrays
        uint32_t*& _idx  = get_index_array();
        char**&    _desc = get_name_array();
        bool*&     _on   = get_bool_array();

        // cleanup
        if(_idx)
            free(_idx);

        if(_on)
            free(_on);

        if(_desc)
        {
            for(size_t i = 0; i < _last_n; i++)
                free(_desc[i]);
            free(_desc);
        }

        // allocate, value of zero for _n sets to nullptr
        _on   = TIMEMORY_CUPTI_CALLOC(bool, _n);
        _idx  = TIMEMORY_CUPTI_CALLOC(uint32_t, _n);
        _desc = TIMEMORY_CUPTI_CALLOC(char*, _n);
        for(size_t i = 0; i < _n; i++)
        {
            _desc[i]    = TIMEMORY_CUPTI_CALLOC(char, CUPTI_STALL_REASON_STRING_SIZE);
            _desc[i][0] = '\0';
        }

        get_size() = _n;
    }
};

/// \struct tim::component::cupti::pcsample
/// \brief Timemory's version of CUpti_PCSamplingPCData
///
struct pcsample
{
    // used by name()
    template <typename KeyT, typename MappedT, typename... Tail>
    using uomap_t = std::unordered_map<KeyT, MappedT, Tail...>;

    // used to set array size
    static constexpr int32_t stall_reasons_size = 38;

    // Number of samples collected across all PCs. It includes all dropped samples.
    mutable uint64_t totalSamples = 0;
    // Unique cubin id
    uint64_t cubinCrc = 0;
    // PC offset
    uint32_t pcOffset = 0;
    // The function's unique symbol index in the module.
    uint32_t functionIndex = 0;
    // Function name
    const char* functionName = "<unknown>";
    // Collected stall reasons
    mutable std::array<pcstall, stall_reasons_size> stalls{};

    pcsample(const pcsample&)     = default;
    pcsample(pcsample&&) noexcept = default;
    pcsample& operator=(const pcsample&) = default;
    pcsample& operator=(pcsample&&) noexcept = default;

    pcsample();
    explicit pcsample(const CUpti_PCSamplingPCData_t&);

    const pcsample& operator+=(const pcsample& rhs) const;
    const pcsample& operator-=(const pcsample& rhs) const;
    pcsample&       operator/=(uint64_t) { return *this; }
    const pcsample& operator/=(uint64_t) const { return *this; }

    bool operator==(const pcsample& rhs) const;
    bool operator<(const pcsample& rhs) const;
    bool operator<=(const pcsample& rhs) const;
    bool operator!=(const pcsample& rhs) const { return !(*this == rhs); }
    bool operator>=(const pcsample& rhs) const { return !(*this < rhs); }
    bool operator>(const pcsample& rhs) const { return !(*this <= rhs); }

    std::string name() const;

    // serialization
    template <typename Archive>
    void save(Archive& ar, const unsigned int) const;
    template <typename Archive>
    void load(Archive& ar, const unsigned int);

    friend std::ostream& operator<<(std::ostream& os, const pcsample& obj)
    {
        os << "samples=" << obj.totalSamples << ", cubinCrc=" << obj.cubinCrc
           << ", pc-offset=" << obj.pcOffset << ", functionIndex=" << obj.functionIndex;
        for(const auto& itr : obj.stalls)
        {
            if(itr.samples > 0)
                os << ", " << itr;
        }
        return os;
    }

    static auto& get_cubin_map()
    {
        static std::unordered_map<uint64_t, std::tuple<const void*, uint32_t>>
            _instance{};
        return _instance;
    }

    static void TIMEMORY_CUPTIAPI compute_cubin_crc(const void* cubin, size_t cubinSize,
                                                    uint64_t* cubinCrc)
    {
        auto&    _map = get_cubin_map();
        uint64_t _idx = _map.size();
        _map.insert({ _idx, { cubin, cubinSize } });
        *cubinCrc = _idx;
    }
};
///
struct pcdata
{
    // Number of PCs collected
    size_t totalNumPcs = 0;
    // Number of PCs available for collection
    size_t remainingNumPcs = 0;
    // Unique identifier for each range. Data collected across multiple ranges in multiple
    // buffers can be identified using range id.
    uint64_t rangeId = 0;
    // Profiled PC data
    std::set<pcsample> samples{};

    TIMEMORY_DEFAULT_OBJECT(pcdata)

    explicit pcdata(CUpti_PCSamplingData_t&& _data);

    pcdata& operator+=(const pcdata& rhs);
    pcdata& operator+=(pcdata&& rhs);
    pcdata& operator-=(const pcdata& rhs);

    bool operator==(const pcdata& rhs) const { return rangeId == rhs.rangeId; }
    bool operator!=(const pcdata& rhs) const { return rangeId != rhs.rangeId; }
    bool operator<(const pcdata& rhs) const { return rangeId < rhs.rangeId; }
    bool operator>(const pcdata& rhs) const { return rangeId > rhs.rangeId; }
    bool operator<=(const pcdata& rhs) const { return rangeId <= rhs.rangeId; }
    bool operator>=(const pcdata& rhs) const { return rangeId >= rhs.rangeId; }

    // serialization
    template <typename Archive>
    void save(Archive& ar, const unsigned int) const;
    template <typename Archive>
    void load(Archive& ar, const unsigned int);

    bool append(const pcsample& rhs);
    bool append(pcsample&& rhs);
};
//
}  // namespace cupti
//
inline namespace stl
{
//
inline cupti::profiler::results_t&
operator+=(cupti::profiler::results_t& lhs, const cupti::profiler::results_t& rhs)
{
    assert(lhs.size() == rhs.size());
    const auto N = ::std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < N; ++i)
        lhs[i] += rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//

inline cupti::profiler::results_t
operator-(const cupti::profiler::results_t& lhs, const cupti::profiler::results_t& rhs)
{
    assert(lhs.size() == rhs.size());
    cupti::profiler::results_t tmp = lhs;
    const auto                 N   = ::std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < N; ++i)
        tmp[i] -= rhs[i];
    return tmp;
}
//
}  // namespace stl
//
//--------------------------------------------------------------------------------------//
//
namespace cupti
{
/// \struct tim::cupti::perfworks_translation
/// \brief provides the mapping from the old CUPTI events and metrics to the perfworks
/// metrics
///
struct perfworks_translation
{
    using map_t = std::map<std::string, std::vector<std::string>>;

    static map_t& get_metrics()
    {
        static map_t _instance = {
            { "achieved_occupancy",
              { "sm__warps_active.avg.pct_of_peak_sustained_active" } },
            { "atomic_transactions",
              { "l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum", "+",
                "l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum", "+",
                "l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum", "+",
                "l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum" } },
            { "cf_executed",
              { "smsp__inst_executed_pipe_cbu.sum", "+",
                "smsp__inst_executed_pipe_adu.sum" } },
            { "double_precision_fu_utilization",
              { "smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active" } },
            { "dram_read_bytes", { "dram__bytes_read.sum" } },
            { "dram_read_throughput", { "dram__bytes_read.sum.per_second" } },
            { "dram_read_transactions", { "dram__sectors_read.sum" } },
            { "dram_utilization",
              { "dram__throughput.avg.pct_of_peak_sustained_elapsed" } },
            { "dram_write_bytes", { "dram__bytes_write.sum" } },
            { "dram_write_throughput", { "dram__bytes_write.sum.per_second" } },
            { "dram_write_transactions", { "dram__sectors_write.sum" } },
            { "eligible_warps_per_cycle",
              { "smsp__warps_eligible.sum.per_cycle_active" } },
            { "flop_count_dp",
              { "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", "+",
                "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum", "+",
                "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", "*", "2" } },
            { "flop_count_dp_add",
              { "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum" } },
            { "flop_count_dp_fma",
              { "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum" } },
            { "flop_count_dp_mul",
              { "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum" } },
            { "flop_count_hp",
              { "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum", "+",
                "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum", "+",
                "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", "*", "2" } },
            { "flop_count_hp_add",
              { "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum" } },
            { "flop_count_hp_fma",
              { "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum" } },
            { "flop_count_hp_mul",
              { "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum" } },
            { "flop_count_sp",
              { "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", "+",
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", "+",
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "*", "2" } },
            { "flop_count_sp_add",
              { "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum" } },
            { "flop_count_sp_fma",
              { "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum" } },
            { "flop_count_sp_mul",
              { "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum" } },
            { "flop_count_sp_special",
              { "smsp__sass_thread_inst_executed_op_mufu_pred_on.sum" } },
            { "gld_throughput",
              { "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second" } },
            { "gld_transactions", { "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum" } },
            { "global_atomic_requests",
              { "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum" } },
            { "global_hit_rate",
              { "l1tex__t_sectors_pipe_lsu_mem_global_op_{op}_lookup_hit.sum", "/",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum" } },
            { "global_load_requests",
              { "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum" } },
            { "global_reduction_requests",
              { "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum" } },
            { "global_store_requests",
              { "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum" } },
            { "gst_throughput",
              { "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second" } },
            { "gst_transactions", { "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum" } },
            { "half_precision_fu_utilization", { "smsp__inst_executed_pipe_fp16.sum" } },
            { "inst_bit_convert",
              { "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum" } },
            { "inst_compute_ld_st",
              { "smsp__sass_thread_inst_executed_op_memory_pred_on.sum" } },
            { "inst_control",
              { "smsp__sass_thread_inst_executed_op_control_pred_on.sum" } },
            { "inst_executed", { "smsp__inst_executed.sum" } },
            { "inst_executed_global_loads", { "smsp__inst_executed_op_global_ld.sum" } },
            { "inst_executed_global_reductions",
              { "smsp__inst_executed_op_global_red.sum" } },
            { "inst_executed_global_stores", { "smsp__inst_executed_op_global_st.sum" } },
            { "inst_executed_local_loads", { "smsp__inst_executed_op_local_ld.sum" } },
            { "inst_executed_local_stores", { "smsp__inst_executed_op_local_st.sum" } },
            { "inst_executed_shared_atomics",
              { "smsp__inst_executed_op_shared_atom.sum", "+",
                "smsp__inst_executed_op_shared_atom_dot_alu.sum", "+",
                "smsp__inst_executed_op_shared_atom_dot_cas.sum" } },
            { "inst_executed_shared_loads", { "smsp__inst_executed_op_shared_ld.sum" } },
            { "inst_executed_shared_stores", { "smsp__inst_executed_op_shared_st.sum" } },
            { "inst_executed_surface_atomics",
              { "smsp__inst_executed_op_surface_atom.sum" } },
            { "inst_executed_surface_loads",
              { "smsp__inst_executed_op_surface_ld.sum", "+",
                "smsp__inst_executed_op_shared_atom_dot_alu.sum", "+",
                "smsp__inst_executed_op_shared_atom_dot_cas.sum" } },
            { "inst_executed_surface_reductions",
              { "smsp__inst_executed_op_surface_red.sum" } },
            { "inst_executed_surface_stores",
              { "smsp__inst_executed_op_surface_st.sum" } },
            { "inst_executed_tex_ops", { "smsp__inst_executed_op_texture.sum" } },
            { "inst_fp_16", { "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum" } },
            { "inst_fp_32", { "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum" } },
            { "inst_fp_64", { "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum" } },
            { "inst_integer",
              { "smsp__sass_thread_inst_executed_op_integer_pred_on.sum" } },
            { "inst_inter_thread_communication",
              { "smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on."
                "sum" } },
            { "inst_issued", { "smsp__inst_issued.sum" } },
            { "inst_misc", { "smsp__sass_thread_inst_executed_op_misc_pred_on.sum" } },
            { "inst_per_warp", { "smsp__average_inst_executed_per_warp.ratio" } },
            { "ipc", { "smsp__inst_executed.avg.per_cycle_active" } },
            { "issue_slot_utilization",
              { "smsp__issue_active.avg.pct_of_peak_sustained_active" } },
            { "issue_slots", { "smsp__inst_issued.sum" } },
            { "issued_ipc", { "smsp__inst_issued.avg.per_cycle_active" } },
            { "l1_sm_lg_utilization", { "l1tex__lsu_writeback_active.sum" } },
            { "l2_atomic_throughput",
              { "lts__t_sectors_srcunit_l1_op_atom.sum.per_second" } },
            { "l2_atomic_transactions", { "lts__t_sectors_srcunit_l1_op_atom.sum" } },
            { "l2_global_atomic_store_bytes",
              { "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_atom.sum" } },
            { "l2_global_load_bytes",
              { "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum" } },
            { "l2_local_global_store_bytes",
              { "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum" } },
            { "l2_local_load_bytes",
              { "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum" } },
            { "l2_read_throughput", { "lts__t_sectors_op_read.sum.per_second" } },
            { "l2_read_transactions", { "lts__t_sectors_op_read.sum" } },
            { "l2_surface_load_bytes",
              { "lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_surface_op_ld.sum" } },
            { "l2_surface_store_bytes",
              { "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_surface_op_st.sum" } },
            { "l2_tex_hit_rate", { "lts__t_sector_hit_rate.pct" } },
            { "l2_tex_read_hit_rate", { "lts__t_sector_op_read_hit_rate.pct" } },
            { "l2_tex_read_throughput",
              { "lts__t_sectors_srcunit_tex_op_read.sum.per_second" } },
            { "l2_tex_read_transactions", { "lts__t_sectors_srcunit_tex_op_read.sum" } },
            { "l2_tex_write_hit_rate", { "lts__t_sector_op_write_hit_rate.pct" } },
            { "l2_tex_write_throughput",
              { "lts__t_sectors_srcunit_tex_op_read.sum.per_second" } },
            { "l2_tex_write_transactions", { "lts__t_sectors_srcunit_tex_op_read.sum" } },
            { "l2_utilization", { "lts__t_sectors.avg.pct_of_peak_sustained_elapsed" } },
            { "l2_write_throughput", { "lts__t_sectors_op_write.sum.per_second" } },
            { "l2_write_transactions", { "lts__t_sectors_op_write.sum" } },
            { "ldst_fu_utilization",
              { "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active" } },
            { "local_load_requests",
              { "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum" } },
            { "local_load_throughput",
              { "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum.per_second" } },
            { "local_load_transactions",
              { "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum" } },
            { "local_store_requests",
              { "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum" } },
            { "local_store_throughput",
              { "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum.per_second" } },
            { "local_store_transactions",
              { "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum" } },
            { "shared_load_throughput",
              { "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second" } },
            { "shared_load_transactions",
              { "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum" } },
            { "shared_store_throughput",
              { "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second" } },
            { "shared_store_transactions",
              { "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum" } },
            { "shared_utilization",
              { "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_"
                "elapsed" } },
            { "single_precision_fu_utilization",
              { "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active" } },
            { "sm_efficiency",
              { "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed" } },
            { "sm_tex_utilization",
              { "l1tex__texin_sm2tex_req_cycles_active.avg.pct_of_peak_sustained_"
                "elapsed" } },
            { "special_fu_utilization",
              { "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active" } },
            { "stall_constant_memory_dependency",
              { "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct" } },
            { "stall_exec_dependency",
              { "smsp__warp_issue_stalled_short_scoreboard_miss_per_warp_active.pct" } },
            { "stall_inst_fetch",
              { "smsp__warp_issue_stalled_no_instruction_miss_per_warp_active.pct" } },
            { "stall_memory_dependency",
              { "smsp__warp_issue_stalled_long_scoreboard_miss_per_warp_active.pct" } },
            { "stall_memory_throttle",
              { "smsp__warp_issue_stalled_drain_miss_per_warp_active.pct" } },
            { "stall_not_selected",
              { "smsp__warp_issue_stalled_not_selected_miss_per_warp_active.pct" } },
            { "stall_other",
              { "smsp__warp_issue_stalled_misc_miss_per_warp_active.pct" } },
            { "stall_pipe_busy",
              { "smsp__warp_issue_stalled_misc_mio_throttle_per_warp_active.pct" } },
            { "stall_sleeping",
              { "smsp__warp_issue_stalled_misc_sleeping_per_warp_active.pct" } },
            { "stall_sync",
              { "smsp__warp_issue_stalled_misc_membar_per_warp_active.pct" } },
            { "stall_texture",
              { "smsp__warp_issue_stalled_misc_tex_throttle_per_warp_active.pct" } },
            { "surface_atomic_requests",
              { "l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum" } },
            { "surface_load_requests",
              { "l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum" } },
            { "surface_reduction_requests",
              { "l1tex__t_requests_pipe_tex_mem_surface_op_red.sum" } },
            { "surface_store_requests",
              { "l1tex__t_requests_pipe_tex_mem_surface_op_st.sum" } },
            { "sysmem_read_throughput",
              { "lts__t_sectors_aperture_sysmem_op_read.sum.per_second" } },
            { "sysmem_read_transactions",
              { "lts__t_sectors_aperture_sysmem_op_read.sum" } },
            { "sysmem_write_throughput",
              { "lts__t_sectors_aperture_sysmem_op_write.sum.per_second" } },
            { "sysmem_write_transactions",
              { "lts__t_sectors_aperture_sysmem_op_write.sum" } },
            { "tensor_precision_fu_utilization",
              { "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active" } },
            { "tex_cache_hit_rate", { "l1tex__t_sector_hit_rate.pct" } },
            { "tex_cache_transactions",
              { "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active", "+",
                "l1tex__tex_writeback_active.avg.pct_of_peak_sustained_active" } },
            { "tex_fu_utilization",
              { "smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active" } },
            { "tex_sm_tex_utilization",
              { "l1tex__f_tex2sm_cycles_active.avg.pct_of_peak_sustained_elapsed" } },
            { "tex_sm_utilization",
              { "sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed" } },
            { "texture_load_requests", { "l1tex__t_requests_pipe_tex_mem_texture.sum" } },
            { "warp_execution_efficiency",
              { "smsp__thread_inst_executed_per_inst_executed.ratio" } },
            { "warp_nonpred_execution_efficiency",
              { "smsp__thread_inst_executed_per_inst_executed.pct" } },
        };
        return _instance;
    }

    static map_t& get_events()
    {
        static map_t _instance = {
            { "active_cycles",
              {
                  "sm__cycles_active.sum",
              } },
            { "active_cycles_pm",
              {
                  "sm__cycles_active.sum",
              } },
            { "active_cycles_sys",
              {
                  "sys__cycles_active.sum",
              } },
            { "active_warps",
              {
                  "sm__warps_active.sum",
              } },
            { "active_warps_pm",
              {
                  "sm__warps_active.sum",
              } },
            { "atom_count",
              {
                  "smsp__inst_executed_op_generic_atom_dot_alu.sum",
              } },
            { "elapsed_cycles_pm",
              {
                  "sm__cycles_elapsed.sum",
              } },
            { "elapsed_cycles_sm",
              {
                  "sm__cycles_elapsed.sum",
              } },
            { "elapsed_cycles_sys",
              {
                  "sys__cycles_elapsed.sum",
              } },
            { "fb_subp0_read_sectors",
              {
                  "dram__sectors_read.sum.sum",
              } },
            { "fb_subp1_read_sectors",
              {
                  "dram__sectors_read.sum",
              } },
            { "fb_subp0_write_sectors",
              {
                  "dram__sectors_write.sum",
              } },
            { "fb_subp1_write_sectors",
              {
                  "dram__sectors_write.sum",
              } },
            { "global_atom_cas",
              {
                  "smsp__inst_executed_op_generic_atom_dot_cas.sum",
              } },
            { "gred_count",
              {
                  "smsp__inst_executed_op_global_red.sum",
              } },
            { "inst_executed",
              {
                  "sm__inst_executed.sum",
              } },
            { "inst_executed_fma_pipe_s0",
              {
                  "smsp__inst_executed_pipe_fma.sum",
              } },
            { "inst_executed_fma_pipe_s1",
              {
                  "smsp__inst_executed_pipe_fma.sum",
              } },
            { "inst_executed_fma_pipe_s2",
              {
                  "smsp__inst_executed_pipe_fma.sum",
              } },
            { "inst_executed_fma_pipe_s3",
              {
                  "smsp__inst_executed_pipe_fma.sum",
              } },
            { "inst_executed_fp16_pipe_s0",
              {
                  "smsp__inst_executed_pipe_fp16.sum",
              } },
            { "inst_executed_fp16_pipe_s1",
              {
                  "smsp__inst_executed_pipe_fp16.sum",
              } },
            { "inst_executed_fp16_pipe_s2",
              {
                  "smsp__inst_executed_pipe_fp16.sum",
              } },
            { "inst_executed_fp16_pipe_s3",
              {
                  "smsp__inst_executed_pipe_fp16.sum",
              } },
            { "inst_executed_fp64_pipe_s0",
              {
                  "smsp__inst_executed_pipe_fp64.sum",
              } },
            { "inst_executed_fp64_pipe_s1",
              {
                  "smsp__inst_executed_pipe_fp64.sum",
              } },
            { "inst_executed_fp64_pipe_s2",
              {
                  "smsp__inst_executed_pipe_fp64.sum",
              } },
            { "inst_executed_fp64_pipe_s3",
              {
                  "smsp__inst_executed_pipe_fp64.sum",
              } },
            { "inst_issued1",
              {
                  "sm__inst_issued.sum",
              } },
            { "l2_subp0_read_sector_misses",
              {
                  "lts__t_sectors_op_read_lookup_miss.sum",
              } },
            { "l2_subp1_read_sector_misses",
              {
                  "lts__t_sectors_op_read_lookup_miss.sum",
              } },
            { "l2_subp0_read_sysmem_sector_queries",
              {
                  "lts__t_sectors_aperture_sysmem_op_read.sum",
              } },
            { "l2_subp1_read_sysmem_sector_queries",
              {
                  "lts__t_sectors_aperture_sysmem_op_read.sum",
              } },
            { "l2_subp0_read_tex_hit_sectors",
              {
                  "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
              } },
            { "l2_subp1_read_tex_hit_sectors",
              {
                  "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
              } },
            { "l2_subp0_read_tex_sector_queries",
              {
                  "lts__t_sectors_srcunit_tex_op_read.sum",
              } },
            { "l2_subp1_read_tex_sector_queries",
              {
                  "lts__t_sectors_srcunit_tex_op_read.sum",
              } },
            { "l2_subp0_total_read_sector_queries",
              {
                  "lts__t_sectors_op_read.sum",
                  "+",
                  "lts__t_sectors_op_atom.sum",
                  "+",
                  "lts__t_sectors_op_red.sum",
              } },
            { "l2_subp1_total_read_sector_queries",
              {
                  "lts__t_sectors_op_read.sum",
                  "+",
                  "lts__t_sectors_op_atom.sum",
                  "+",
                  "lts__t_sectors_op_red.sum",
              } },
            { "l2_subp0_total_write_sector_queries",
              {
                  "lts__t_sectors_op_write.sum",
                  "+",
                  "lts__t_sectors_op_atom.sum",
                  "+",
                  "lts__t_sectors_op_red.sum",
              } },
            { "l2_subp1_total_write_sector_queries",
              {
                  "lts__t_sectors_op_write.sum",
                  "+",
                  "lts__t_sectors_op_atom.sum",
                  "+",
                  "lts__t_sectors_op_red.sum",
              } },
            { "l2_subp0_write_sector_misses",
              {
                  "lts__t_sectors_op_write_lookup_miss.sum",
              } },
            { "l2_subp1_write_sector_misses",
              {
                  "lts__t_sectors_op_write_lookup_miss.sum",
              } },
            { "l2_subp0_write_sysmem_sector_queries",
              {
                  "lts__t_sectors_aperture_sysmem_op_write.sum",
              } },
            { "l2_subp1_write_sysmem_sector_queries",
              {
                  "lts__t_sectors_aperture_sysmem_op_write.sum",
              } },
            { "l2_subp0_write_tex_hit_sectors",
              {
                  "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum",
              } },
            { "l2_subp1_write_tex_hit_sectors",
              {
                  "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum",
              } },
            { "l2_subp0_write_tex_sector_queries",
              {
                  "lts__t_sectors_srcunit_tex_op_write.sum",
              } },
            { "l2_subp1_write_tex_sector_queries",
              {
                  "lts__t_sectors_srcunit_tex_op_write.sum",
              } },
            { "not_predicated_off_thread_inst_executed",
              {
                  "smsp__thread_inst_executed_pred_on.sum",
              } },
            { "inst_issued0",
              {
                  "smsp__issue_inst0.sum",
              } },
            { "sm_cta_launched",
              {
                  "sm__ctas_launched.sum",
              } },
            { "shared_load",
              {
                  "smsp__inst_executed_op_shared_ld.sum",
              } },
            { "shared_store",
              {
                  "smsp__inst_executed_op_shared_st.sum",
              } },
            { "generic_load",
              {
                  "smsp__inst_executed_op_generic_ld.sum",
              } },
            { "generic_store",
              {
                  "smsp__inst_executed_op_generic_st.sum",
              } },
            { "global_load",
              {
                  "smsp__inst_executed_op_global_ld.sum",
              } },
            { "global_store",
              {
                  "smsp__inst_executed_op_global_st.sum",
              } },
            { "local_load",
              {
                  "smsp__inst_executed_op_local_ld.sum",
              } },
            { "local_store",
              {
                  "smsp__inst_executed_op_local_st.sum",
              } },
            { "shared_atom",
              {
                  "smsp__inst_executed_op_shared_atom.sum",
              } },
            { "shared_atom_cas",
              {
                  "smsp__inst_executed_shared_atom_dot_cas.sum",
              } },
            { "shared_ld_bank_conflict",
              {
                  "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
              } },
            { "shared_st_bank_conflict",
              {
                  "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
              } },
            { "shared_ld_transactions",
              {
                  "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
              } },
            { "shared_st_transactions",
              {
                  "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
              } },
            { "tensor_pipe_active_cycles_s0",
              {
                  "smsp__pipe_tensor_cycles_active.sum",
              } },
            { "tensor_pipe_active_cycles_s1",
              {
                  "smsp__pipe_tensor_cycles_active.sum",
              } },
            { "tensor_pipe_active_cycles_s2",
              {
                  "smsp__pipe_tensor_cycles_active.sum",
              } },
            { "tensor_pipe_active_cycles_s3",
              {
                  "smsp__pipe_tensor_cycles_active.sum",
              } },
            { "thread_inst_executed",
              {
                  "smsp__thread_inst_executed.sum",
              } },
            { "warps_launched",
              {
                  "smsp__warps_launched.sum",
              } }
        };
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace cupti
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//

#if defined(TIMEMORY_CUPTI_COMPONENT_HEADER_MODE)
#    include "timemory/components/cupti/backends.cpp"
#endif
