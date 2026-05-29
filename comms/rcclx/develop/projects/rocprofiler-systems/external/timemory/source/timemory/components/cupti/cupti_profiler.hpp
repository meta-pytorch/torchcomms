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

/**
 * \headerfile "timemory/components/cupti/cupti_profiler.hpp"
 * Provides implementation of CUPTI Profiler API
 *
 */

#pragma once

#if defined(TIMEMORY_USE_CUPTI_NVPERF)

#    include "timemory/components/base.hpp"
#    include "timemory/components/cupti/backends.hpp"
#    include "timemory/components/cupti/types.hpp"
#    include "timemory/macros.hpp"
#    include "timemory/settings/declaration.hpp"

#    include <nvperf_cuda_host.h>
#    include <nvperf_host.h>
#    include <nvperf_target.h>
//
#    include <cupti_profiler_target.h>
#    include <cupti_target.h>
//
#    include <cstdint>
#    include <cuda.h>
#    include <fstream>
#    include <iostream>
#    include <string>
#    include <vector>
//
//======================================================================================//
//
#    if !defined(TIMEMORY_CUPTI_API_CALL)
#        define TIMEMORY_CUPTI_API_CALL(...) TIMEMORY_CUPTI_CALL(__VA_ARGS__)
#    endif
//
//======================================================================================//
//
#    if !defined(TIMEMORY_NVPW_API_CALL)
#        define TIMEMORY_NVPW_API_CALL(apiFuncCall)                                      \
            do                                                                           \
            {                                                                            \
                NVPA_Status _status = apiFuncCall;                                       \
                if(_status != NVPA_STATUS_SUCCESS)                                       \
                {                                                                        \
                    TIMEMORY_PRINTF_FATAL(                                               \
                        stderr, "%s:%d: error: function %s failed with error %d.\n",     \
                        __FILE__, __LINE__, #apiFuncCall, _status);                      \
                    exit(EXIT_FAILURE);                                                  \
                }                                                                        \
            } while(0)
#    endif
//
//======================================================================================//
//
#    define TIMEMORY_RETURN_IF_NVPW_ERROR(retval, actual)                                \
        do                                                                               \
        {                                                                                \
            if(NVPA_STATUS_SUCCESS != actual)                                            \
            {                                                                            \
                TIMEMORY_PRINTF_WARNING(stderr, "FAILED: %s\n", #actual);                \
                return retval;                                                           \
            }                                                                            \
        } while(0)
//
//======================================================================================//
//
template <typename T>
class ScopeExit
{
public:
    ScopeExit(T _t)
    : t(_t)
    {}
    ~ScopeExit() { t(); }
    T t;
};
//
//======================================================================================//
//
template <typename T>
ScopeExit<T>
MoveScopeExit(T t)
{
    return ScopeExit<T>(t);
}
//
//======================================================================================//
//
#    define NV_ANONYMOUS_VARIABLE_DIRECT(name, line) name##line
#    define NV_ANONYMOUS_VARIABLE_INDIRECT(name, line)                                   \
        NV_ANONYMOUS_VARIABLE_DIRECT(name, line)
#    define SCOPE_EXIT(func)                                                             \
        const auto NV_ANONYMOUS_VARIABLE_INDIRECT(EXIT, __LINE__) =                      \
            MoveScopeExit([=]() { func; })
//
//======================================================================================//
//
namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          CUPTI Profiler component
//
//--------------------------------------------------------------------------------------//
/// \struct tim::component::cupti_profiler
/// \brief Hardware counters via the CUpti profiling API. The profiling API is only
/// available with newer NVIDIA hardware and CUDA 10+. This component currently has issues
/// with nesting.
///
struct cupti_profiler : public base<cupti_profiler, std::map<std::string, double>>
{
protected:
    struct MetricNameValue;

public:
    // required aliases
    using value_type = std::map<std::string, double>;
    using this_type  = cupti_profiler;
    using base_type  = base<this_type, value_type>;
    using data_type  = std::vector<MetricNameValue>;

    // component-specific aliases
    using size_type = std::size_t;
    using string_t  = std::string;

    static std::string label() { return "cupti_profiler"; }
    static std::string description() { return "CUpti Profiler API"; }

    static void configure(int device = 0);
    static void finalize();

    static void global_init() { configure(); }

    static void global_finalize() { finalize(); }

    TIMEMORY_DEFAULT_OBJECT(cupti_profiler)

    value_type record()
    {
        auto&      chipName         = get_persistent_data().chipName;
        auto&      counterDataImage = get_persistent_data().counterDataImage;
        auto&      metricNames      = get_persistent_data().metricNames;
        data_type  _data;
        value_type _tmp;
        GetMetricGpuValue(chipName, counterDataImage, metricNames, _data);
        if(settings::verbose() > 0)
            TIMEMORY_PRINT_HERE("METRIC_GPU_VALUE size: %li", (long int) _data.size());
        for(const auto& itr : _data)
        {
            auto _prefix = itr.metricName + ".";
            if(settings::verbose() > 0)
                TIMEMORY_PRINT_HERE("    METRIC[%s] size: %li", itr.metricName.c_str(),
                                    (long int) itr.rangeNameMetricValueMap.size());
            for(const auto& vitr : itr.rangeNameMetricValueMap)
                _tmp[_prefix + vitr.first] = vitr.second;
        }
        return _tmp;
    }

    void start()
    {
        auto _count = get_counter()++;
        if(_count == 0)
        {
            // enable();
            TIMEMORY_CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
            TIMEMORY_CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
        }

        value = record();
        TIMEMORY_CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
    }

    void stop()
    {
        using namespace tim::component::operators;
        TIMEMORY_CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));
        auto _count = --get_counter();
        if(_count == 0)
        {
            TIMEMORY_CUPTI_API_CALL(
                cuptiProfilerDisableProfiling(&disableProfilingParams));
            TIMEMORY_CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
            // disable();
        }

        cuda::stream_sync(0);
        cuda::device_sync();

        TIMEMORY_CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));

        auto _tmp = record();
        for(auto& itr : _tmp)
        {
            auto& vitr = value[itr.first];
            vitr       = (itr.second - vitr);
            accum[itr.first] += vitr;
        }
    }

    void set_prefix(const std::string& _prefix)
    {
        pushRangeParams.pRangeName = _prefix.c_str();
    }

    std::vector<double> get() const
    {
        std::vector<double> data;
        for(const auto& itr : accum)
            data.emplace_back(itr.second);
        return data;
    }

    string_t get_display() const
    {
        auto _get_display = [&](std::ostream& os, const auto& obj) {
            auto _label = obj.first;
            auto _prec  = base_type::get_precision();
            auto _width = base_type::get_width();
            auto _flags = base_type::get_format_flags();

            std::stringstream ssv, ssi;
            ssv.setf(_flags);
            ssv << std::setw(_width) << std::setprecision(_prec) << obj.second;
            if(!_label.empty())
                ssi << " " << _label;
            os << ssv.str() << ssi.str();
        };

        const auto&       _data = load();
        std::stringstream ss;
        for(size_type i = 0; i < _data.size(); ++i)
        {
            auto itr = _data.begin();
            std::advance(itr, i);
            _get_display(ss, *itr);
            if(i + 1 < _data.size())
                ss << ", ";
        }
        return ss.str();
    }

    static std::vector<string_t> label_array()
    {
        auto ret = get_persistent_data().metricNames;
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    static std::vector<string_t> description_array() { return label_array(); }

    static std::vector<string_t> display_unit_array()
    {
        return std::vector<string_t>(get_persistent_data().metricNames.size(), "");
    }

    static std::vector<int64_t> unit_array()
    {
        return std::vector<int64_t>(get_persistent_data().metricNames.size(), 1);
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto _get = [&](const value_type& _data) {
            std::vector<double> values;
            for(auto itr : _data)
                values.push_back(itr.second);
            return values;
        };
        std::vector<double> _disp  = _get(accum);
        std::vector<double> _value = _get(value);
        std::vector<double> _accum = _get(accum);
        ar(cereal::make_nvp("laps", laps), cereal::make_nvp("repr_data", _disp),
           cereal::make_nvp("value", _value), cereal::make_nvp("accum", _accum),
           cereal::make_nvp("display", _disp));
        // ar(cereal::make_nvp("units", unit_array()),
        //   cereal::make_nvp("display_units", display_unit_array()));
    }

public:
    this_type& operator+=(const this_type& rhs)
    {
        for(const auto& itr : rhs.value)
        {
            value[itr.first] += itr.second;
        }

        for(const auto& itr : rhs.accum)
        {
            accum[itr.first] += itr.second;
        }

        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        for(const auto& itr : rhs.value)
        {
            if(value.find(itr.first) != value.end())
                value[itr.first] -= itr.second;
        }

        for(const auto& itr : rhs.accum)
        {
            if(accum.find(itr.first) != accum.end())
                accum[itr.first] -= itr.second;
        }

        return *this;
    }

public:
    static bool WriteBinaryFile(const char* pFileName, const std::vector<uint8_t>& data);
    static bool ReadBinaryFile(const char* pFileName, std::vector<uint8_t>& image);

    static std::set<std::string> ListSupportedChips();
    static std::set<std::string> ListMetrics(const char* chipName, bool listSubMetrics);
    static std::string           GetHwUnit(const std::string& metricName);

    static bool GetMetricGpuValue(std::string                   chipName,
                                  std::vector<uint8_t>          counterDataImage,
                                  std::vector<std::string>      metricNames,
                                  std::vector<MetricNameValue>& metricNameValueMap);

    static bool PrintMetricValues(std::string              chipName,
                                  std::vector<uint8_t>     counterDataImage,
                                  std::vector<std::string> metricNames);

protected:
    static bool create_counter_data_image(std::vector<uint8_t>& counterDataImage,
                                          std::vector<uint8_t>& counterDataScratchBuffer,
                                          std::vector<uint8_t>& counterDataImagePrefix);

    static bool enable();
    static bool disable();

    static bool GetConfigImage(std::string chipName, std::vector<std::string> metricNames,
                               std::vector<uint8_t>& configImage);

    static bool GetCounterDataPrefixImage(std::string              chipName,
                                          std::vector<std::string> metricNames,
                                          std::vector<uint8_t>& counterDataImagePrefix);

    static bool GetRawMetricRequests(
        NVPA_MetricsContext* pMetricsContext, std::vector<std::string> metricNames,
        std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
        std::vector<std::string>&           temp);

    static bool ParseMetricNameString(const std::string& metricName, std::string* reqName,
                                      bool* isolated, bool* keepInstances);

protected:
    std::string* m_prefix = nullptr;

    CUpti_Profiler_PushRange_Params pushRangeParams = {
        CUpti_Profiler_PushRange_Params_STRUCT_SIZE
    };
    CUpti_Profiler_PopRange_Params popRangeParams = {
        CUpti_Profiler_PopRange_Params_STRUCT_SIZE
    };
    CUpti_Profiler_BeginPass_Params beginPassParams = {
        CUpti_Profiler_BeginPass_Params_STRUCT_SIZE
    };
    CUpti_Profiler_EndPass_Params endPassParams = {
        CUpti_Profiler_EndPass_Params_STRUCT_SIZE
    };
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
        CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE
    };
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE
    };
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE
    };

protected:
    struct MetricNameValue
    {
        std::string                                 metricName;
        int                                         numRanges;
        std::vector<std::pair<std::string, double>> rangeNameMetricValueMap;
    };

    struct persistent_data
    {
        std::atomic<int64_t>     instCounter;
        CUdevice                 cuDevice;
        CUcontext                cuContext;
        bool                     enabled                = false;
        int                      deviceCount            = 1;
        int                      deviceNum              = 0;
        int                      numRanges              = 64;
        int                      computeCapabilityMajor = 0;
        int                      computeCapabilityMinor = 0;
        CUpti_ProfilerReplayMode profilerReplayMode     = CUPTI_UserReplay;
        CUpti_ProfilerRange      profilerRange          = CUPTI_UserRange;
        std::string              chipName               = "";
        std::string              CounterDataFileName    = "SimpleCupti.counterdata";
        std::string              CounterDataSBFileName  = "SimpleCupti.counterdataSB";
        std::vector<uint8_t>     counterDataImagePrefix;
        std::vector<uint8_t>     configImage;
        std::vector<uint8_t>     counterDataImage;
        std::vector<uint8_t>     counterDataScratchBuffer;
        std::vector<std::string> metricNames;
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }

    static std::atomic<int64_t>& get_counter()
    {
        return get_persistent_data().instCounter;
    }
};
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::create_counter_data_image(std::vector<uint8_t>& counterDataImage,
                                          std::vector<uint8_t>& counterDataScratchBuffer,
                                          std::vector<uint8_t>& counterDataImagePrefix)
{
    auto& numRanges = get_persistent_data().numRanges;

    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix    = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges          = numRanges;
    counterDataImageOptions.maxNumRangeTreeNodes  = numRanges;
    counterDataImageOptions.maxRangeNameLength    = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
        CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE
    };

    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions =
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    TIMEMORY_CUPTI_API_CALL(
        cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
        CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE
    };

    initializeParams.sizeofCounterDataImageOptions =
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions             = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];

    TIMEMORY_CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
        scratchBufferSizeParams = {
            CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE
        };

    scratchBufferSizeParams.counterDataImageSize =
        calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;

    TIMEMORY_CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
        &scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
        initScratchBufferParams = {
            CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE
        };

    initScratchBufferParams.counterDataImageSize =
        calculateSizeParams.counterDataImageSize;
    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize =
        scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];

    TIMEMORY_CUPTI_API_CALL(
        cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline void
cupti_profiler::configure(int device)
{
    auto& cuDevice                 = get_persistent_data().cuDevice;
    auto& metricNames              = get_persistent_data().metricNames;
    auto& counterDataImagePrefix   = get_persistent_data().counterDataImagePrefix;
    auto& configImage              = get_persistent_data().configImage;
    auto& counterDataImage         = get_persistent_data().counterDataImage;
    auto& counterDataScratchBuffer = get_persistent_data().counterDataScratchBuffer;
    auto& profilerReplayMode       = get_persistent_data().profilerReplayMode;
    auto& profilerRange            = get_persistent_data().profilerRange;
    auto& deviceCount              = get_persistent_data().deviceCount;
    auto& deviceNum                = get_persistent_data().deviceNum;
    auto& computeCapabilityMajor   = get_persistent_data().computeCapabilityMajor;
    auto& computeCapabilityMinor   = get_persistent_data().computeCapabilityMinor;
    auto& chipName                 = get_persistent_data().chipName;

    TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if(deviceCount == 0)
    {
        TIMEMORY_PRINTF_WARNING(stderr, "There is no device supporting CUDA.\n");
        return;
    }

    deviceNum = device;
    TIMEMORY_PRINTF(stderr, "CUDA Device Number: %d\n", deviceNum);

    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGetAttribute(
        &computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGetAttribute(
        &computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

    TIMEMORY_PRINTF(stderr, "Compute Capability of Device: %d.%d\n",
                    computeCapabilityMajor, computeCapabilityMinor);

    if(computeCapabilityMajor < 7)
    {
        TIMEMORY_PRINTF(stderr,
                        "Sample unsupported on Device with compute capability < 7.0\n");
        return;
    }

    // Get the names of the metrics to collect
    metricNames =
        tim::delimit(settings::cupti_events() + "," + settings::cupti_metrics(), " ;,");

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE
    };
    TIMEMORY_CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE
    };

    getChipNameParams.deviceIndex = deviceNum;

    TIMEMORY_CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    chipName = getChipNameParams.pChipName;

    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = {
        NVPW_InitializeHost_Params_STRUCT_SIZE
    };

    TIMEMORY_NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

    if(metricNames.size())
    {
        if(!GetConfigImage(chipName, metricNames, configImage))
        {
            std::cerr << "Failed to create configImage" << std::endl;
            return;
        }
        if(!GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix))
        {
            std::cerr << "Failed to create counterDataImagePrefix" << std::endl;
            return;
        }
    }
    else
    {
        std::cerr << "No metrics provided to profile" << std::endl;
        return;
    }

    if(!create_counter_data_image(counterDataImage, counterDataScratchBuffer,
                                  counterDataImagePrefix))
    {
        std::cerr << "Failed to create counterDataImage" << std::endl;
        return;
    }

    auto& enabled   = get_persistent_data().enabled;
    auto& numRanges = get_persistent_data().numRanges;
    auto& cuContext = get_persistent_data().cuContext;

    CUpti_Profiler_BeginSession_Params beginSessionParams = {
        CUpti_Profiler_BeginSession_Params_STRUCT_SIZE
    };

    CUpti_Profiler_SetConfig_Params setConfigParams = {
        CUpti_Profiler_SetConfig_Params_STRUCT_SIZE
    };

    TIMEMORY_CUDA_DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

    beginSessionParams.ctx                          = cuContext;
    beginSessionParams.counterDataImageSize         = counterDataImage.size();
    beginSessionParams.pCounterDataImage            = &counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer    = &counterDataScratchBuffer[0];
    beginSessionParams.range                        = profilerRange;
    beginSessionParams.replayMode                   = profilerReplayMode;
    beginSessionParams.maxRangesPerPass             = numRanges;
    beginSessionParams.maxLaunchesPerPass           = numRanges;

    setConfigParams.pConfig          = &configImage[0];
    setConfigParams.configSize       = configImage.size();
    setConfigParams.passIndex        = 0;
    setConfigParams.minNestingLevel  = 1;
    setConfigParams.numNestingLevels = 1;

    TIMEMORY_CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
    TIMEMORY_CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

    enabled = true;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::enable()
{
    auto& enabled = get_persistent_data().enabled;
    if(!enabled)
        configure();

    auto& cuContext                = get_persistent_data().cuContext;
    auto& numRanges                = get_persistent_data().numRanges;
    auto& cuDevice                 = get_persistent_data().cuDevice;
    auto& metricNames              = get_persistent_data().metricNames;
    auto& counterDataImagePrefix   = get_persistent_data().counterDataImagePrefix;
    auto& configImage              = get_persistent_data().configImage;
    auto& counterDataImage         = get_persistent_data().counterDataImage;
    auto& counterDataScratchBuffer = get_persistent_data().counterDataScratchBuffer;
    auto& profilerReplayMode       = get_persistent_data().profilerReplayMode;
    auto& profilerRange            = get_persistent_data().profilerRange;
    auto& deviceCount              = get_persistent_data().deviceCount;
    auto& deviceNum                = get_persistent_data().deviceNum;
    auto& computeCapabilityMajor   = get_persistent_data().computeCapabilityMajor;
    auto& computeCapabilityMinor   = get_persistent_data().computeCapabilityMinor;
    auto& chipName                 = get_persistent_data().chipName;

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE
    };

    CUpti_Profiler_BeginPass_Params beginPassParams = {
        CUpti_Profiler_BeginPass_Params_STRUCT_SIZE
    };

    TIMEMORY_CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
    TIMEMORY_CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline void
cupti_profiler::finalize()
{
    auto& chipName                 = get_persistent_data().chipName;
    auto& counterDataImage         = get_persistent_data().counterDataImage;
    auto& counterDataScratchBuffer = get_persistent_data().counterDataScratchBuffer;
    auto& CounterDataFileName      = get_persistent_data().CounterDataFileName;
    auto& CounterDataSBFileName    = get_persistent_data().CounterDataSBFileName;
    auto& metricNames              = get_persistent_data().metricNames;
    auto& cuContext                = get_persistent_data().cuContext;

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
        CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE
    };

    CUpti_Profiler_EndSession_Params endSessionParams = {
        CUpti_Profiler_EndSession_Params_STRUCT_SIZE
    };

    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
        CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE
    };

    TIMEMORY_CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
    TIMEMORY_CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuCtxDestroy(cuContext));
    TIMEMORY_CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

    /* Dump counterDataImage in file */
    WriteBinaryFile(CounterDataFileName.c_str(), counterDataImage);
    WriteBinaryFile(CounterDataSBFileName.c_str(), counterDataScratchBuffer);

    /* Evaluation of metrics collected in counterDataImage, this can also be done
     * offline*/
    PrintMetricValues(chipName, counterDataImage, metricNames);
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::disable()
{
    auto& enabled   = get_persistent_data().enabled;
    auto& cuContext = get_persistent_data().cuContext;

    if(!enabled)
        return false;

    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE
    };

    CUpti_Profiler_EndPass_Params endPassParams = {
        CUpti_Profiler_EndPass_Params_STRUCT_SIZE
    };

    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
        CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE
    };

    TIMEMORY_CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
    TIMEMORY_CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
    TIMEMORY_CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));

    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline std::string
cupti_profiler::GetHwUnit(const std::string& metricName)
{
    return metricName.substr(0, metricName.find("__", 0));
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::GetMetricGpuValue(std::string                   chipName,
                                  std::vector<uint8_t>          counterDataImage,
                                  std::vector<std::string>      metricNames,
                                  std::vector<MetricNameValue>& metricNameValueMap)
{
    if(!counterDataImage.size())
    {
        std::cout << "Counter Data Image is empty!\n";
        return false;
    }

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE
    };

    metricsContextCreateParams.pChipName = chipName.c_str();
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
        NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE
    };

    metricsContextDestroyParams.pMetricsContext =
        metricsContextCreateParams.pMetricsContext;

    SCOPE_EXIT([&]() {
        NVPW_MetricsContext_Destroy(
            (NVPW_MetricsContext_Destroy_Params*) &metricsContextDestroyParams);
    });

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = {
        NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE
    };

    getNumRangesParams.pCounterDataImage = &counterDataImage[0];
    TIMEMORY_RETURN_IF_NVPW_ERROR(false,
                                  NVPW_CounterData_GetNumRanges(&getNumRangesParams));

    std::vector<std::string> reqName;
    reqName.resize(metricNames.size());

    bool                     isolated      = true;
    bool                     keepInstances = true;
    std::vector<const char*> metricNamePtrs;
    metricNameValueMap.resize(metricNames.size());

    for(size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
    {
        ParseMetricNameString(metricNames[metricIndex], &reqName[metricIndex], &isolated,
                              &keepInstances);
        metricNamePtrs.push_back(reqName[metricIndex].c_str());
        metricNameValueMap[metricIndex].metricName = metricNames[metricIndex];
        metricNameValueMap[metricIndex].numRanges  = getNumRangesParams.numRanges;
    }

    for(size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
    {
        std::vector<const char*> descriptionPtrs;

        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = {
            NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE
        };
        getRangeDescParams.pCounterDataImage = &counterDataImage[0];
        getRangeDescParams.rangeIndex        = rangeIndex;
        TIMEMORY_RETURN_IF_NVPW_ERROR(
            false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
        descriptionPtrs.resize(getRangeDescParams.numDescriptions);

        getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
        TIMEMORY_RETURN_IF_NVPW_ERROR(
            false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

        std::string rangeName;
        for(size_t descriptionIndex = 0;
            descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
        {
            if(descriptionIndex)
            {
                rangeName += "/";
            }
            rangeName += descriptionPtrs[descriptionIndex];
        }

        std::vector<double> gpuValues;
        gpuValues.resize(metricNames.size());
        NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
            NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE
        };
        setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        setCounterDataParams.pCounterDataImage = &counterDataImage[0];
        setCounterDataParams.isolated          = true;
        setCounterDataParams.rangeIndex        = rangeIndex;
        NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = {
            NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE
        };
        evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        evalToGpuParams.numMetrics      = metricNamePtrs.size();
        evalToGpuParams.ppMetricNames   = &metricNamePtrs[0];
        evalToGpuParams.pMetricValues   = &gpuValues[0];
        NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);
        for(size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
        {
            metricNameValueMap[metricIndex].rangeNameMetricValueMap.push_back(
                std::make_pair(rangeName, gpuValues[metricIndex]));
        }
    }

    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::PrintMetricValues(std::string              chipName,
                                  std::vector<uint8_t>     counterDataImage,
                                  std::vector<std::string> metricNames)
{
    if(!counterDataImage.size())
    {
        std::cout << "Counter Data Image is empty!\n";
        return false;
    }

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE
    };
    metricsContextCreateParams.pChipName = chipName.c_str();
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
        NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE
    };
    metricsContextDestroyParams.pMetricsContext =
        metricsContextCreateParams.pMetricsContext;
    SCOPE_EXIT([&]() {
        NVPW_MetricsContext_Destroy(
            (NVPW_MetricsContext_Destroy_Params*) &metricsContextDestroyParams);
    });

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = {
        NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE
    };
    getNumRangesParams.pCounterDataImage = &counterDataImage[0];
    TIMEMORY_RETURN_IF_NVPW_ERROR(false,
                                  NVPW_CounterData_GetNumRanges(&getNumRangesParams));

    std::vector<std::string> reqName;
    reqName.resize(metricNames.size());
    bool                     isolated      = true;
    bool                     keepInstances = true;
    std::vector<const char*> metricNamePtrs;
    for(size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
    {
        ParseMetricNameString(metricNames[metricIndex], &reqName[metricIndex], &isolated,
                              &keepInstances);
        metricNamePtrs.push_back(reqName[metricIndex].c_str());
    }

    for(size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
    {
        std::vector<const char*> descriptionPtrs;

        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = {
            NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE
        };
        getRangeDescParams.pCounterDataImage = &counterDataImage[0];
        getRangeDescParams.rangeIndex        = rangeIndex;
        TIMEMORY_RETURN_IF_NVPW_ERROR(
            false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

        descriptionPtrs.resize(getRangeDescParams.numDescriptions);

        getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
        TIMEMORY_RETURN_IF_NVPW_ERROR(
            false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

        std::string rangeName;
        for(size_t descriptionIndex = 0;
            descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
        {
            if(descriptionIndex)
            {
                rangeName += "/";
            }
            rangeName += descriptionPtrs[descriptionIndex];
        }

        std::vector<double> gpuValues;
        gpuValues.resize(metricNames.size());

        NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
            NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE
        };
        setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        setCounterDataParams.pCounterDataImage = &counterDataImage[0];
        setCounterDataParams.isolated          = true;
        setCounterDataParams.rangeIndex        = rangeIndex;
        NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = {
            NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE
        };
        evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        evalToGpuParams.numMetrics      = metricNamePtrs.size();
        evalToGpuParams.ppMetricNames   = &metricNamePtrs[0];
        evalToGpuParams.pMetricValues   = &gpuValues[0];
        NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);

        for(size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
        {
            std::cout << "rangeName: " << rangeName
                      << "\tmetricName: " << metricNames[metricIndex]
                      << "\tgpuValue: " << gpuValues[metricIndex] << std::endl;
        }
    }
    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::GetRawMetricRequests(
    NVPA_MetricsContext* pMetricsContext, std::vector<std::string> metricNames,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests, std::vector<std::string>& temp)
{
    std::string reqName;
    bool        isolated      = true;
    bool        keepInstances = true;

    for(auto& metricName : metricNames)
    {
        ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
        /* Bug in collection with collection of metrics without instances, keep it to
         * true*/
        keepInstances = true;
        NVPW_MetricsContext_GetMetricProperties_Begin_Params
            getMetricPropertiesBeginParams = {
                NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE
            };
        getMetricPropertiesBeginParams.pMetricsContext = pMetricsContext;
        getMetricPropertiesBeginParams.pMetricName     = reqName.c_str();

        TIMEMORY_RETURN_IF_NVPW_ERROR(false,
                                      NVPW_MetricsContext_GetMetricProperties_Begin(
                                          &getMetricPropertiesBeginParams));

        for(const char** ppMetricDependencies =
                getMetricPropertiesBeginParams.ppRawMetricDependencies;
            *ppMetricDependencies; ++ppMetricDependencies)
        {
            temp.push_back(*ppMetricDependencies);
        }
        NVPW_MetricsContext_GetMetricProperties_End_Params
            getMetricPropertiesEndParams = {
                NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE
            };
        getMetricPropertiesEndParams.pMetricsContext = pMetricsContext;
        TIMEMORY_RETURN_IF_NVPW_ERROR(false, NVPW_MetricsContext_GetMetricProperties_End(
                                                 &getMetricPropertiesEndParams));
    }

    consume_parameters(isolated);

    for(auto& rawMetricName : temp)
    {
        NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
        metricRequest.pMetricName           = rawMetricName.c_str();
        metricRequest.isolated              = isolated;
        metricRequest.keepInstances         = keepInstances;
        rawMetricRequests.push_back(metricRequest);
    }

    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::GetConfigImage(std::string chipName, std::vector<std::string> metricNames,
                               std::vector<uint8_t>& configImage)
{
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE
    };
    metricsContextCreateParams.pChipName = chipName.c_str();
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
        NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE
    };
    metricsContextDestroyParams.pMetricsContext =
        metricsContextCreateParams.pMetricsContext;
    SCOPE_EXIT([&]() {
        NVPW_MetricsContext_Destroy(
            (NVPW_MetricsContext_Destroy_Params*) &metricsContextDestroyParams);
    });

    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string>           temp;
    GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames,
                         rawMetricRequests, temp);

    NVPA_RawMetricsConfigOptions metricsConfigOptions = {
        NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE
    };
    metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    metricsConfigOptions.pChipName    = chipName.c_str();
    NVPA_RawMetricsConfig* pRawMetricsConfig;
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPA_RawMetricsConfig_Create(&metricsConfigOptions, &pRawMetricsConfig));

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
        NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE
    };
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
    SCOPE_EXIT([&]() {
        NVPW_RawMetricsConfig_Destroy(
            (NVPW_RawMetricsConfig_Destroy_Params*) &rawMetricsConfigDestroyParams);
    });

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE
    };
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE
    };
    addMetricsParams.pRawMetricsConfig  = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
    addMetricsParams.numMetricRequests  = rawMetricRequests.size();
    TIMEMORY_RETURN_IF_NVPW_ERROR(false,
                                  NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE
    };
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = {
        NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE
    };
    generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {
        NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE
    };
    getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    getConfigImageParams.bytesAllocated    = 0;
    getConfigImageParams.pBuffer           = NULL;
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

    configImage.resize(getConfigImageParams.bytesCopied);

    getConfigImageParams.bytesAllocated = configImage.size();
    getConfigImageParams.pBuffer        = &configImage[0];
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::GetCounterDataPrefixImage(std::string              chipName,
                                          std::vector<std::string> metricNames,
                                          std::vector<uint8_t>&    counterDataImagePrefix)
{
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE
    };

    metricsContextCreateParams.pChipName = chipName.c_str();

    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
        NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE
    };

    metricsContextDestroyParams.pMetricsContext =
        metricsContextCreateParams.pMetricsContext;

    SCOPE_EXIT([&]() {
        NVPW_MetricsContext_Destroy(
            (NVPW_MetricsContext_Destroy_Params*) &metricsContextDestroyParams);
    });

    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string>           temp;
    GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames,
                         rawMetricRequests, temp);

    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {
        NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE
    };

    counterDataBuilderCreateParams.pChipName = chipName.c_str();

    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {
        NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE
    };

    counterDataBuilderDestroyParams.pCounterDataBuilder =
        counterDataBuilderCreateParams.pCounterDataBuilder;

    SCOPE_EXIT([&]() {
        NVPW_CounterDataBuilder_Destroy(
            (NVPW_CounterDataBuilder_Destroy_Params*) &counterDataBuilderDestroyParams);
    });

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = {
        NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE
    };

    addMetricsParams.pCounterDataBuilder =
        counterDataBuilderCreateParams.pCounterDataBuilder;
    addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
    addMetricsParams.numMetricRequests  = rawMetricRequests.size();
    TIMEMORY_RETURN_IF_NVPW_ERROR(false,
                                  NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

    // size_t                                              counterDataPrefixSize      = 0;
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = {
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE
    };

    getCounterDataPrefixParams.pCounterDataBuilder =
        counterDataBuilderCreateParams.pCounterDataBuilder;

    getCounterDataPrefixParams.bytesAllocated = 0;
    getCounterDataPrefixParams.pBuffer        = NULL;
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

    counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);

    getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
    getCounterDataPrefixParams.pBuffer        = &counterDataImagePrefix[0];
    TIMEMORY_RETURN_IF_NVPW_ERROR(
        false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline std::set<std::string>
cupti_profiler::ListSupportedChips()
{
    std::set<std::string> _ret;

    NVPW_GetSupportedChipNames_Params getSupportedChipNames = {
        NVPW_GetSupportedChipNames_Params_STRUCT_SIZE
    };
    TIMEMORY_RETURN_IF_NVPW_ERROR(_ret,
                                  NVPW_GetSupportedChipNames(&getSupportedChipNames));

    if(settings::verbose() > 2 || settings::debug())
    {
        std::cout << "\n Number of supported chips : "
                  << getSupportedChipNames.numChipNames;
        std::cout << "\n List of supported chips : \n";
    }

    for(size_t i = 0; i < getSupportedChipNames.numChipNames; i++)
    {
        _ret.insert(getSupportedChipNames.ppChipNames[i]);
        if(settings::verbose() > 2 || settings::debug())
            std::cout << " " << getSupportedChipNames.ppChipNames[i] << "\n";
    }

    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
inline std::set<std::string>
cupti_profiler::ListMetrics(const char* chip, bool listSubMetrics)
{
    std::set<std::string> _ret;

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE
    };

    metricsContextCreateParams.pChipName = chip;

    TIMEMORY_RETURN_IF_NVPW_ERROR(
        _ret, NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
        NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE
    };

    metricsContextDestroyParams.pMetricsContext =
        metricsContextCreateParams.pMetricsContext;

    SCOPE_EXIT([&]() {
        NVPW_MetricsContext_Destroy(
            (NVPW_MetricsContext_Destroy_Params*) &metricsContextDestroyParams);
    });

    NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = {
        NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE
    };

    getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    getMetricNameBeginParams.hidePeakSubMetrics      = !listSubMetrics;
    getMetricNameBeginParams.hidePerCycleSubMetrics  = !listSubMetrics;
    getMetricNameBeginParams.hidePctOfPeakSubMetrics = !listSubMetrics;

    TIMEMORY_RETURN_IF_NVPW_ERROR(
        _ret, NVPW_MetricsContext_GetMetricNames_Begin(&getMetricNameBeginParams));

    NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = {
        NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE
    };

    getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    SCOPE_EXIT([&]() {
        NVPW_MetricsContext_GetMetricNames_End(
            (NVPW_MetricsContext_GetMetricNames_End_Params*) &getMetricNameEndParams);
    });

    if(settings::verbose() > 2 || settings::debug())
        std::cout << getMetricNameBeginParams.numMetrics
                  << " metrics in total on the chip\n Metrics List : \n";

    for(size_t i = 0; i < getMetricNameBeginParams.numMetrics; i++)
    {
        _ret.insert(getMetricNameBeginParams.ppMetricNames[i]);
        if(settings::verbose() > 2 || settings::debug())
            std::cout << getMetricNameBeginParams.ppMetricNames[i] << "\n";
    }

    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::ParseMetricNameString(const std::string& metricName, std::string* reqName,
                                      bool* isolated, bool* keepInstances)
{
    std::string& name = *reqName;
    name              = metricName;
    if(name.empty())
    {
        return false;
    }

    // boost program_options sometimes inserts a \n between the metric name and a '&'
    // at the end
    size_t pos = name.find('\n');
    if(pos != std::string::npos)
    {
        name.erase(pos, 1);
    }

    // trim whitespace
    while(name.back() == ' ')
    {
        name.pop_back();
        if(name.empty())
        {
            return false;
        }
    }

    *keepInstances = false;
    if(name.back() == '+')
    {
        *keepInstances = true;
        name.pop_back();
        if(name.empty())
        {
            return false;
        }
    }

    *isolated = true;
    if(name.back() == '$')
    {
        name.pop_back();
        if(name.empty())
        {
            return false;
        }
    }
    else if(name.back() == '&')
    {
        *isolated = false;
        name.pop_back();
        if(name.empty())
        {
            return false;
        }
    }

    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::WriteBinaryFile(const char* pFileName, const std::vector<uint8_t>& data)
{
    FILE* fp = fopen(pFileName, "wb");
    if(fp)
    {
        if(data.size())
        {
            fwrite(&data[0], 1, data.size(), fp);
        }
        fclose(fp);
    }
    else
    {
        std::cout << "Failed to open " << pFileName << "\n";
        fclose(fp);
        return false;
    }
    return true;
}
//
//--------------------------------------------------------------------------------------//
//
inline bool
cupti_profiler::ReadBinaryFile(const char* pFileName, std::vector<uint8_t>& image)
{
    FILE* fp = fopen(pFileName, "rb");
    if(!fp)
    {
        std::cout << "Failed to open " << pFileName << "\n";
        return false;
    }

    fseek(fp, 0, SEEK_END);
    const long fileLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if(!fileLength)
    {
        std::cout << pFileName << " has zero length\n";
        fclose(fp);
        return false;
    }

    image.resize((size_t) fileLength);
    auto ret = fread(&image[0], 1, image.size(), fp);
    fclose(fp);
    return (ret != image.size()) ? false : true;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim

#endif
