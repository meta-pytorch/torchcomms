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

#if defined(TIMEMORY_USE_CUPTI_PCSAMPLING)

#    include "timemory/components/cupti/types.hpp"

#    if !defined(TIMEMORY_COMPONENTS_CUPTI_CUPTI_PCSAMPLING_HPP_)
#        include "timemory/components/cupti/cupti_pcsampling.hpp"
#    endif

#    include "timemory/operations/types/derive.hpp"
#    include "timemory/operations/types/node.hpp"
#    include "timemory/operations/types/reset.hpp"
#    include "timemory/operations/types/start.hpp"
#    include "timemory/operations/types/stop.hpp"
#    include "timemory/operations/types/store.hpp"
#    include "timemory/variadic/component_tuple.hpp"

namespace tim
{
namespace cupti
{
//
TIMEMORY_CUPTI_INLINE
pcdata::pcdata(CUpti_PCSamplingData&& _data)
: totalNumPcs{ _data.totalNumPcs }
, remainingNumPcs{ _data.remainingNumPcs }
, rangeId{ _data.rangeId }
{
    // samples.reserve(_sz);
    for(size_t i = 0; i < totalNumPcs; ++i)
    {
        if(!_data.pPcData[i].stallReason)
            continue;
        pcsample _sample{ _data.pPcData[i] };
        auto     itr = samples.find(_sample);
        if(itr == samples.end())
            samples.insert(std::move(_sample));
        else
            *itr += _sample;
    }
    component::cupti_pcsampling::free_pcsampling_data(_data);
}
}  // namespace cupti
//
namespace component
{
//
TIMEMORY_CUPTI_INLINE CUpti_PCSamplingData
cupti_pcsampling::get_pcsampling_data(size_t numStallReasons, size_t numPcsToCollect)
{
    // User buffer to hold collected PC Sampling data in PC-To-Counter format
    CUpti_PCSamplingData pcSamplingData = {};
    pcSamplingData.size                 = sizeof(CUpti_PCSamplingData);
    pcSamplingData.collectNumPcs        = numPcsToCollect;
    pcSamplingData.pPcData =
        TIMEMORY_CUPTI_CALLOC(CUpti_PCSamplingPCData, pcSamplingData.collectNumPcs);
    for(size_t i = 0; i < pcSamplingData.collectNumPcs; ++i)
        pcSamplingData.pPcData[i].stallReason =
            TIMEMORY_CUPTI_CALLOC(CUpti_PCSamplingStallReason, numStallReasons);
    return pcSamplingData;
}
//
TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::free_pcsampling_data(CUpti_PCSamplingData pcSamplingData)
{
    // Free memory
    for(size_t i = 0; i < pcSamplingData.collectNumPcs; i++)
    {
        if(pcSamplingData.pPcData[i].stallReason)
        {
            free(pcSamplingData.pPcData[i].stallReason);
            pcSamplingData.pPcData[i].stallReason = nullptr;
        }
    }

    if(pcSamplingData.pPcData)
    {
        free(pcSamplingData.pPcData);
        pcSamplingData.pPcData = nullptr;
    }
}
//
inline const char*
getStallReason(const uint32_t& stallReasonCount,
               const uint32_t& pcSamplingStallReasonIndex, uint32_t* pStallReasonIndex,
               char** pStallReasons)
{
    for(uint32_t i = 0; i < stallReasonCount; i++)
    {
        if(pStallReasonIndex[i] == pcSamplingStallReasonIndex)
        {
            return pStallReasons[i];
        }
    }
    return "ERROR_STALL_REASON_INDEX_NOT_FOUND";
}
//
TIMEMORY_CUPTI_INLINE void
printPCSamplingData(CUpti_PCSamplingData* pPcSamplingData,
                    const uint32_t& stallReasonCount, uint32_t* pStallReasonIndex,
                    char** pStallReasons)
{
    std::cout << "----- PC sampling data for range defined by cuptiPCSamplingStart() and "
                 "cuptiPCSamplingStop() -----"
              << std::endl;
    std::cout << "Number of PCs remaining to be collected: "
              << pPcSamplingData->remainingNumPcs << ", ";
    std::cout << "range id: " << pPcSamplingData->rangeId << ", ";
    std::cout << "total samples: " << pPcSamplingData->totalSamples << ", ";
    std::cout << "dropped samples: " << pPcSamplingData->droppedSamples << std::endl;
    for(size_t i = 0; i < pPcSamplingData->totalNumPcs; i++)
    {
        std::cout << "\tpcOffset : 0x" << std::hex << pPcSamplingData->pPcData[i].pcOffset
                  << ", stallReasonCount: " << std::dec
                  << pPcSamplingData->pPcData[i].stallReasonCount << ", functionName: "
                  << demangle(pPcSamplingData->pPcData[i].functionName);
        for(size_t j = 0; j < pPcSamplingData->pPcData[i].stallReasonCount; j++)
        {
            std::cout << "\n\t\tstallReason: "
                      << getStallReason(stallReasonCount,
                                        pPcSamplingData->pPcData[i]
                                            .stallReason[j]
                                            .pcSamplingStallReasonIndex,
                                        pStallReasonIndex, pStallReasons)
                      << ", samples: "
                      << pPcSamplingData->pPcData[i].stallReason[j].samples;
        }
        std::cout << std::endl;
    }
    std::cout << "-----------------------------------------------------------------------"
                 "---------------------------"
              << std::endl;
}
//
TIMEMORY_CUPTI_INLINE cupti_pcsampling::config_type
                      cupti_pcsampling::configure()
{
    CUcontext      cuCtx;
    CUdevice       cuDevice;
    cudaDeviceProp prop;
    int            deviceCount = 0;
    int            deviceNum   = 0;

    TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if(deviceCount == 0)
    {
        TIMEMORY_PRINTF_FATAL(stderr, "There is no device supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // only check if value is still set to the default
    if(get_configuration_data().region_totals)
        get_configuration_data().region_totals =
            settings::instance()->get<bool>("cupti_pcsampling_region_totals");

    deviceNum = settings::instance()->get<int>("cupti_device");
    if(deviceNum < 0)
    {
        deviceNum = 0;
        TIMEMORY_CUDA_RUNTIME_API_CALL(cudaGetDevice(&deviceNum));
    }

    TIMEMORY_CUDA_RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));
    TIMEMORY_PRINTF(stderr, "Device Name: %s\n", prop.name);
    TIMEMORY_PRINTF(stderr, "Device compute capability: %d.%d\n", prop.major, prop.minor);
    if(prop.major < 7)
    {
        TIMEMORY_PRINTF(
            stderr, "Component is unavailable on this device, supported on devices with "
                    "compute capability 7.0 and higher\n");
        exit(EXIT_FAILURE);
    }

    // cuda::device_sync();

    // Create CUDA context
    // TIMEMORY_CUDA_DRIVER_API_CALL(cuCtxCreate(&cuCtx, 0, cuDevice));
    // Init device, context and setup callback
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
    // TIMEMORY_CUDA_DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDevicePrimaryCtxRetain(&cuCtx, cuDevice));
    // TIMEMORY_CUDA_DRIVER_API_CALL(cuCtxGetCurrent(&cuCtx));
    // TIMEMORY_CUDA_DRIVER_API_CALL(cuCtxCreate(&cuCtx, 0, deviceNum));

    TIMEMORY_CUPTI_API_CALL(
        cuptiRegisterComputeCrcCallback(&cupti::pcsample::compute_cubin_crc));

    //----------------------------------------------------------------------------------//
    // Enable PC Sampling
    CUpti_PCSamplingEnableParams pcSamplingEnableParams = {};
    pcSamplingEnableParams.size = CUpti_PCSamplingEnableParamsSize;
    pcSamplingEnableParams.ctx  = cuCtx;
    TIMEMORY_CUPTI_API_CALL(cuptiPCSamplingEnable(&pcSamplingEnableParams));

    //----------------------------------------------------------------------------------//
    // Get number of supported stall reasons
    size_t                                   numStallReasons       = 0;
    CUpti_PCSamplingGetNumStallReasonsParams numStallReasonsParams = {};
    numStallReasonsParams.size            = CUpti_PCSamplingGetNumStallReasonsParamsSize;
    numStallReasonsParams.ctx             = cuCtx;
    numStallReasonsParams.numStallReasons = &numStallReasons;
    TIMEMORY_CUPTI_API_CALL(cuptiPCSamplingGetNumStallReasons(&numStallReasonsParams));

    //----------------------------------------------------------------------------------//
    // Get number of supported stall reason names and corresponding indexes
    cupti::pcstall::allocate_arrays(numStallReasons);
    uint32_t*& pStallReasonIndex    = cupti::pcstall::get_index_array();
    char**&    pStallReasons        = cupti::pcstall::get_name_array();
    bool*&     pStallReasonsEnabled = cupti::pcstall::get_bool_array();

    CUpti_PCSamplingGetStallReasonsParams stallReasonsParams = {};
    stallReasonsParams.size             = CUpti_PCSamplingGetStallReasonsParamsSize;
    stallReasonsParams.ctx              = cuCtx;
    stallReasonsParams.numStallReasons  = numStallReasons;
    stallReasonsParams.stallReasonIndex = pStallReasonIndex;
    stallReasonsParams.stallReasons     = pStallReasons;
    cuptiPCSamplingGetStallReasons(&stallReasonsParams);

    size_t stallReasonCount = numStallReasons;
    for(size_t i = 0; i < numStallReasons; ++i)
        pStallReasonsEnabled[i] = true;
    auto _stall_reasons =
        delimit(settings::instance()->get<std::string>("cupti_pcsampling_stall_reasons"));
    if(!_stall_reasons.empty() && _stall_reasons.size() < stallReasonCount)
    {
        stallReasonCount = _stall_reasons.size();
        // set all to false
        for(size_t i = 0; i < numStallReasons; ++i)
            pStallReasonsEnabled[i] = false;
        // enable all that match
        for(const auto& itr : _stall_reasons)
        {
            for(size_t i = 0; i < numStallReasons; ++i)
            {
                if(std::regex_search(pStallReasons[i], std::regex(itr + "$")))
                    pStallReasonsEnabled[i] = true;
            }
        }
        // reorder the reasons that are enabled to the beginning
        size_t _idx = 0;
        for(size_t i = 0; i < numStallReasons; ++i)
        {
            if(pStallReasonsEnabled[i] && _idx != i)
            {
                std::swap(pStallReasonsEnabled[i], pStallReasonsEnabled[_idx]);
                std::swap(pStallReasonIndex[i], pStallReasonIndex[_idx]);
                std::swap(pStallReasons[i], pStallReasons[_idx]);
                ++_idx;
            }
        }
    }
    if(settings::debug() || settings::verbose() > 0)
    {
        TIMEMORY_PRINTF(stderr, "[cupti_pcsampling][runtime] numStallReasons = %lu\n",
                        (unsigned long) numStallReasons);
        TIMEMORY_PRINTF(stderr, "[cupti_pcsampling][compile] numStallReasons = %lu\n",
                        (unsigned long) cupti::pcsample::stall_reasons_size);
        for(size_t i = 0; i < stallReasonCount; ++i)
            TIMEMORY_PRINTF(stderr, "[cupti_pcsampling] %s index: %lu\n",
                            pStallReasons[i], (long unsigned) pStallReasonIndex[i]);
        // check that there are not new stall reasons that have not been accounted for
        assert(numStallReasons <= cupti::pcsample::stall_reasons_size);
    }

    //----------------------------------------------------------------------------------//
    // User buffer to hold collected PC Sampling data in PC-To-Counter format
    size_t _num_collect =
        settings::instance()->get<size_t>("cupti_pcsampling_num_collect");
    CUpti_PCSamplingData pcSamplingData =
        get_pcsampling_data(stallReasonCount, _num_collect);

    //----------------------------------------------------------------------------------//
    //              configuration info
    //----------------------------------------------------------------------------------//

    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo{};

    //------------------------ Sampling Period ------------------------//
    auto _period = settings::instance()->get<int>("cupti_pcsampling_period");
    CUpti_PCSamplingConfigurationInfo samplingPeriod = {};
    samplingPeriod.attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    samplingPeriod.attributeData.samplingPeriodData.samplingPeriod = _period;
    pcSamplingConfigurationInfo.push_back(samplingPeriod);

    //------------------------ Stall Reason ------------------------//
    CUpti_PCSamplingConfigurationInfo stallReason = {};
    stallReason.attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
    stallReason.attributeData.stallReasonData.stallReasonCount  = stallReasonCount;
    stallReason.attributeData.stallReasonData.pStallReasonIndex = pStallReasonIndex;
    pcSamplingConfigurationInfo.push_back(stallReason);

    //------------------------ Scratch Buffer ------------------------//
    CUpti_PCSamplingConfigurationInfo scratchBufferSize = {};
    scratchBufferSize.attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    scratchBufferSize.attributeData.scratchBufferSizeData.scratchBufferSize =
        10 * units::MB;
    pcSamplingConfigurationInfo.push_back(scratchBufferSize);

    //------------------------ Collect Mode ------------------------//
    auto _serialized = settings::instance()->get<int>("cupti_pcsampling_serialized");
    CUpti_PCSamplingConfigurationInfo collectionMode = {};
    collectionMode.attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
    collectionMode.attributeData.collectionModeData.collectionMode =
        (_serialized) ? CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED
                      : CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
    pcSamplingConfigurationInfo.push_back(collectionMode);

    //------------------------ Start / Stop ------------------------//
    CUpti_PCSamplingConfigurationInfo enableStartStop = {};
    enableStartStop.attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    enableStartStop.attributeData.enableStartStopControlData.enableStartStopControl =
        true;
    pcSamplingConfigurationInfo.push_back(enableStartStop);

    //------------------------ Output Data Fmt ------------------------//
    CUpti_PCSamplingConfigurationInfo outputDataFormat = {};
    outputDataFormat.attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
    outputDataFormat.attributeData.outputDataFormatData.outputDataFormat =
        CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;
    pcSamplingConfigurationInfo.push_back(outputDataFormat);

    //------------------------ Sample Data Buffer ------------------------//
    CUpti_PCSamplingConfigurationInfo samplingDataBuffer = {};
    samplingDataBuffer.attributeType =
        CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    samplingDataBuffer.attributeData.samplingDataBufferData.samplingDataBuffer =
        (void*) pcSamplingData.pPcData;
    pcSamplingConfigurationInfo.push_back(samplingDataBuffer);

    //------------------------ Config Info Params ------------------------//
    CUpti_PCSamplingConfigurationInfoParams pcSamplingConfigurationInfoParams = {};
    pcSamplingConfigurationInfoParams.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    pcSamplingConfigurationInfoParams.ctx  = cuCtx;
    pcSamplingConfigurationInfoParams.numAttributes = pcSamplingConfigurationInfo.size();
    pcSamplingConfigurationInfoParams.pPCSamplingConfigurationInfo =
        pcSamplingConfigurationInfo.data();
    TIMEMORY_CUPTI_API_CALL(
        cuptiPCSamplingSetConfigurationAttribute(&pcSamplingConfigurationInfoParams));

    for(auto itr : pcSamplingConfigurationInfo)
        TIMEMORY_CUPTI_API_CALL(itr.attributeStatus);

    // At start of code region
    CUpti_PCSamplingStartParams pcSamplingStartParams = {};
    pcSamplingStartParams.size                        = CUpti_PCSamplingStartParamsSize;
    pcSamplingStartParams.ctx                         = cuCtx;
    // cuptiPCSamplingStart(&pcSamplingStartParams);

    // At end of code region
    CUpti_PCSamplingStopParams pcSamplingStopParams = {};
    pcSamplingStopParams.size                       = CUpti_PCSamplingStopParamsSize;
    pcSamplingStopParams.ctx                        = cuCtx;
    // cuptiPCSamplingStop(&pcSamplingStopParams);

    return std::make_tuple(cuCtx, pcSamplingEnableParams, numStallReasonsParams,
                           stallReasonsParams, pcSamplingData,
                           std::move(pcSamplingConfigurationInfo),
                           pcSamplingConfigurationInfoParams, pcSamplingStartParams,
                           pcSamplingStopParams, stallReasonCount, _num_collect);
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::initialize()
{
    auto& _cfg = get_configuration_data();
    if(!_cfg.enabled)
        std::tie(_cfg.enabled, _cfg.data) = std::make_tuple(true, configure());
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::finalize()
{
    auto& _cfg = get_configuration_data();
    if(_cfg.enabled)
    {
        _cfg.enabled    = false;
        CUcontext cuCtx = _cfg.context();
        // Disable PC Sampling
        CUpti_PCSamplingDisableParams pcSamplingDisableParams = {};
        pcSamplingDisableParams.size = CUpti_PCSamplingDisableParamsSize;
        pcSamplingDisableParams.ctx  = cuCtx;
        TIMEMORY_CUPTI_API_CALL(cuptiPCSamplingDisable(&pcSamplingDisableParams));
        // Destroy CUDA context
        // TIMEMORY_CUDA_DRIVER_API_CALL(cuCtxDestroy(cuCtx));
    }
}

TIMEMORY_CUPTI_INLINE cupti_pcsampling::data_type
                      cupti_pcsampling::record()
{
    auto& _cfg = get_configuration_data();
    if(!_cfg.enabled)
        return data_type{};

    auto pcSamplingData =
        get_pcsampling_data(_cfg.num_stall_reasons(), _cfg.num_collect_pcs());

    CUpti_PCSamplingGetDataParams pcSamplingGetDataParams = {};
    pcSamplingGetDataParams.size           = CUpti_PCSamplingGetDataParamsSize;
    pcSamplingGetDataParams.ctx            = _cfg.context();
    pcSamplingGetDataParams.pcSamplingData = static_cast<void*>(&pcSamplingData);
    TIMEMORY_CUPTI_API_CALL(cuptiPCSamplingGetData(&pcSamplingGetDataParams));
    if(settings::debug())
        printPCSamplingData(&pcSamplingData, _cfg.num_stall_reasons(),
                            cupti::pcstall::get_index_array(),
                            cupti::pcstall::get_name_array());
    return data_type{ std::move(pcSamplingData) };
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::sample()
{
    cupti::pcdata _data = record();
    for(auto&& itr : _data.samples)
    {
        component_tuple<cupti_pcsampling> _bundle{ itr.name() };
        _bundle.push();
        _bundle.store(std::move(itr));
        _bundle.pop();
    }
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::store(const value_type& _data)
{
    value = _data;
    if(get_configuration_data().region_totals)
        for(auto& itr : get_stack())
            itr->value += value;
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::store(value_type&& _data)
{
    value = std::move(_data);
    if(get_configuration_data().region_totals)
        for(auto& itr : get_stack())
            itr->value += value;
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::start()
{
    auto _n = tracker_type::start();
    if(_n == 0)
    {
        initialize();
        TIMEMORY_CUPTI_API_CALL(
            cuptiPCSamplingStart(&get_configuration_data().start_params()));
    }
    if(get_configuration_data().region_totals)
        get_stack().insert(this);
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::stop()
{
    sample();
    if(get_configuration_data().region_totals)
        get_stack().erase(this);

    auto _n = tracker_type::stop();
    if(_n == 0)
    {
        TIMEMORY_CUPTI_API_CALL(
            cuptiPCSamplingStop(&get_configuration_data().stop_params()));
        // finalize();
    }
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::set_started()
{
    base_type::set_started();
}

TIMEMORY_CUPTI_INLINE void
cupti_pcsampling::set_stopped()
{
    // don't increment laps
    if(get_is_running())
        set_is_running(false);
}

TIMEMORY_CUPTI_INLINE std::string
                      cupti_pcsampling::get_display() const
{
    std::stringstream ss;
    ss.precision(base_type::get_precision());
    ss.width(base_type::get_width());
    ss.setf(base_type::get_format_flags());
    ss << load();
    return ss.str();
}
//
TIMEMORY_CUPTI_INLINE std::vector<int64_t>
                      cupti_pcsampling::get() const
{
    auto                 _n = cupti::pcstall::get_size();
    std::vector<int64_t> _data{};
    _data.reserve(_n);
    auto _val = load();
    for(size_t i = 0; i < _n; ++i)
    {
        if(cupti::pcstall::enabled(i) && strlen(cupti::pcstall::name(i)) > 0)
        {
            // this is a packed field so directly emplacing it can cause compiler errors
            uint32_t _v = _val.stalls[i].samples;
            _data.emplace_back(static_cast<int64_t>(_v));
        }
    }
    return _data;
}
//
TIMEMORY_CUPTI_INLINE std::vector<std::string>
                      cupti_pcsampling::label_array()
{
    auto                     _n = cupti::pcstall::get_size();
    std::vector<std::string> _data{};
    _data.reserve(_n);
    for(size_t i = 0; i < _n; ++i)
    {
        if(cupti::pcstall::enabled(i) && strlen(cupti::pcstall::name(i)) > 0)
            _data.push_back(std::string{ cupti::pcstall::name(i) });
    }
    return _data;
}
//
}  // namespace component
}  // namespace tim

#endif  // TIMEMORY_USE_CUPTI_PCSAMPLING
