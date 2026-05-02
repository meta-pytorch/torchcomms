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

#include "timemory/components/cupti/types.hpp"

#if !defined(TIMEMORY_CUPTI_COMPONENT_HEADER_MODE)
#    include "timemory/components/cupti/backends.hpp"
#endif

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
namespace cupti
{
//
TIMEMORY_CUPTI_INLINE pcdata&
pcdata::operator+=(const pcdata& rhs)
{
    assert(rangeId == rhs.rangeId);
    totalNumPcs     = std::max(totalNumPcs, rhs.totalNumPcs);
    remainingNumPcs = std::min(remainingNumPcs, rhs.remainingNumPcs);
    for(const auto& itr : rhs.samples)
        append(itr);
    return *this;
}
//
TIMEMORY_CUPTI_INLINE pcdata&
pcdata::operator+=(pcdata&& rhs)
{
    assert(rangeId == rhs.rangeId);
    totalNumPcs     = std::max(totalNumPcs, rhs.totalNumPcs);
    remainingNumPcs = std::min(remainingNumPcs, rhs.remainingNumPcs);
    for(auto&& itr : rhs.samples)
        append(std::move(itr));
    return *this;
}
//
TIMEMORY_CUPTI_INLINE pcdata&
pcdata::operator-=(const pcdata& rhs)
{
    assert(rangeId == rhs.rangeId);
    for(const auto& ritr : rhs.samples)
    {
        for(auto& itr : samples)
        {
            if(ritr == itr)  // match found
            {
                itr -= ritr;
                break;
            }
        }
    }
    return *this;
}
//
TIMEMORY_CUPTI_INLINE bool
pcdata::append(const pcsample& _sample)
{
    for(auto& itr : samples)
    {
        if(_sample == itr)  // match found
        {
            itr += _sample;
            return false;
        }
    }
    samples.insert(_sample);
    return true;
}
//
TIMEMORY_CUPTI_INLINE bool
pcdata::append(pcsample&& _sample)
{
    for(auto& itr : samples)
    {
        if(_sample == itr)  // match found
        {
            itr += _sample;
            return false;
        }
    }
    samples.insert(std::move(_sample));
    return true;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_CUPTI_INLINE
pcsample::pcsample()
{
    for(size_t i = 0; i < stalls.size(); ++i)
        stalls[i].index = i;
}
//---------------------------------------//
#if defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
//---------------------------------------//
TIMEMORY_CUPTI_INLINE
pcsample::pcsample(const CUpti_PCSamplingPCData_t& _pcdata)
: cubinCrc(_pcdata.cubinCrc)
, pcOffset(_pcdata.pcOffset)
, functionIndex(_pcdata.functionIndex)
, functionName(_pcdata.functionName)
{
    for(size_t i = 0; i < stalls.size(); ++i)
        stalls[i].index = i;
    for(size_t i = 0; i < _pcdata.stallReasonCount; ++i)
    {
        const auto& _stall = _pcdata.stallReason[i];
        auto        ridx   = _stall.pcSamplingStallReasonIndex;
        stalls[ridx]       = pcstall{ _stall };
    }

    for(auto& itr : stalls)
        totalSamples += itr.samples;
}
//---------------------------------------//
#else
//---------------------------------------//
TIMEMORY_CUPTI_INLINE
pcsample::pcsample(const CUpti_PCSamplingPCData_t&) {}
//---------------------------------------//
#endif
//
TIMEMORY_CUPTI_INLINE const pcsample&
pcsample::operator+=(const pcsample& rhs) const
{
    for(int32_t i = 0; i < stall_reasons_size; ++i)
        stalls[i] += rhs.stalls[i];
    for(int32_t i = 0; i < stall_reasons_size; ++i)
        totalSamples += rhs.stalls[i].samples;
    return *this;
}
//
TIMEMORY_CUPTI_INLINE const pcsample&
pcsample::operator-=(const pcsample& rhs) const
{
    for(int32_t i = 0; i < stall_reasons_size; ++i)
        stalls[i] -= rhs.stalls[i];
    for(int32_t i = 0; i < stall_reasons_size; ++i)
        totalSamples -= rhs.stalls[i].samples;
    return *this;
}
//
TIMEMORY_CUPTI_INLINE bool
pcsample::operator==(const pcsample& rhs) const
{
    return std::tie(cubinCrc, pcOffset, functionIndex) ==
           std::tie(rhs.cubinCrc, rhs.pcOffset, rhs.functionIndex);
}
//
TIMEMORY_CUPTI_INLINE bool
pcsample::operator<(const pcsample& rhs) const
{
    return (cubinCrc < rhs.cubinCrc) || (pcOffset < rhs.pcOffset) ||
           (functionIndex < rhs.functionIndex);
}
//
TIMEMORY_CUPTI_INLINE bool
pcsample::operator<=(const pcsample& rhs) const
{
    return (*this == rhs) || (*this < rhs);
}
//
TIMEMORY_CUPTI_INLINE std::string
                      pcsample::name() const
{
#if defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
    // involves a look-up so cache this result
    static auto _per_line = settings::instance()->get<bool>("cupti_pcsampling_per_line");

    if(_per_line)
    {
        static uomap_t<uint32_t, uomap_t<uint32_t, uomap_t<uint64_t, std::string>>>
             _sass2src{};
        auto itr = _sass2src[functionIndex][pcOffset].find(cubinCrc);
        if(itr == _sass2src[functionIndex][pcOffset].end())
        {
            CUpti_GetSassToSourceCorrelationParams sassToSourceParams = {};
            sassToSourceParams.size      = sizeof(CUpti_GetSassToSourceCorrelationParams);
            sassToSourceParams.cubin     = std::get<0>(get_cubin_map().at(cubinCrc));
            sassToSourceParams.cubinSize = std::get<1>(get_cubin_map().at(cubinCrc));
            sassToSourceParams.functionName = functionName;
            sassToSourceParams.pcOffset     = pcOffset;
            TIMEMORY_CUPTI_API_CALL(cuptiGetSassToSourceCorrelation(&sassToSourceParams));
            if(sassToSourceParams.fileName)
            {
                auto _fname = string_view_t{ sassToSourceParams.fileName };
                auto _line  = sassToSourceParams.lineNumber;
                _sass2src[functionIndex][pcOffset][cubinCrc] =
                    TIMEMORY_JOIN("", demangle(functionName), '/', _fname, ':', _line);
                free(sassToSourceParams.fileName);
                free(sassToSourceParams.dirName);
            }
            else
            {
                _sass2src[functionIndex][pcOffset][cubinCrc] = functionName;
            }
            itr = _sass2src[functionIndex][pcOffset].find(cubinCrc);
        }
        return itr->second;
    }
#endif

    return functionName;
}
//
//--------------------------------------------------------------------------------------//
//
//---------------------------------------//
#if defined(TIMEMORY_USE_CUPTI_PCSAMPLING)
//---------------------------------------//
TIMEMORY_CUPTI_INLINE
pcstall::pcstall(const CUpti_PCSamplingStallReason_t& _obj)
: index(_obj.pcSamplingStallReasonIndex)
, samples(_obj.samples)
{}
//---------------------------------------//
#else
//---------------------------------------//
TIMEMORY_CUPTI_INLINE
pcstall::pcstall(const CUpti_PCSamplingStallReason_t&) {}
//---------------------------------------//
#endif
//
TIMEMORY_CUPTI_INLINE
pcstall::pcstall(uint32_t _index, uint32_t _samples)
: index(_index)
, samples(_samples)
{}
//
TIMEMORY_CUPTI_INLINE pcstall&
pcstall::operator+=(const pcstall& rhs)
{
    samples += rhs.samples;
    return *this;
}
//
TIMEMORY_CUPTI_INLINE pcstall&
pcstall::operator-=(const pcstall& rhs)
{
    samples -= rhs.samples;
    return *this;
}
//
TIMEMORY_CUPTI_INLINE const char*
pcstall::name(uint32_t _index)
{
    if(_index >= get_size())
        return "<unknown>";
    for(uint32_t i = 0; i < get_size(); ++i)
    {
        auto _idx = get_index_array()[i];
        if(_idx == _index)
            return get_name_array()[i];
    }
    return "<unknown>";
}
//
TIMEMORY_CUPTI_INLINE bool
pcstall::enabled(uint32_t _index)
{
    if(_index >= get_size())
        return false;
    for(uint32_t i = 0; i < get_size(); ++i)
    {
        auto _idx = get_index_array()[i];
        if(_idx == _index)
            return get_bool_array()[i];
    }
    return false;
}

}  // namespace cupti
}  // namespace tim
