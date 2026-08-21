# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# NCCLx C++ namespace bindings (direct linkage, not dlsym)

from libc.stdint cimport int8_t, int16_t, uint64_t
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef extern from "cuda_runtime_api.h" nogil:
    ctypedef struct CUstream_st
    ctypedef CUstream_st* cudaStream_t


cdef extern from "nccl.h" nogil:
    ctypedef unsigned int ncclResult_t
    ctypedef int ncclDataType_t
    ctypedef void* ncclComm_t
    ctypedef void* ncclWindow_t
    ctypedef int ncclRedOp_t

    ctypedef int ncclWinAccessType
    # Declared as a cppclass (rather than a plain struct) so the heap instance
    # that ncclWinGetAttributes hands back can be released with `del` (C++
    # delete), matching the `new` used in the C++ implementation.
    cdef cppclass ncclWinAttr:
        ncclWinAccessType accessType
    ctypedef ncclWinAttr* ncclWinAttr_t

    ncclResult_t ncclWinSharedQuery(
        int rank, ncclComm_t comm, ncclWindow_t win, void** addr)
    ncclResult_t ncclWinGetAttributes(
        int rank, ncclWindow_t win, ncclWinAttr_t* attr)
    ncclResult_t ncclReduceScatterQuantize(
        const void* sendbuff, void* recvbuff, size_t recvcount,
        ncclDataType_t inputType, ncclDataType_t transportType,
        ncclRedOp_t op, uint64_t* seedPtr,
        ncclComm_t comm, cudaStream_t stream)
    ncclResult_t ncclAllToAllv(
        const void* sendbuff, const size_t* sendcounts, const size_t* sdispls,
        void* recvbuff, const size_t* recvcounts, const size_t* rdispls,
        ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
    ncclResult_t ncclCommDump(
        ncclComm_t comm, unordered_map[string, string]& result)
    ncclResult_t ncclCommDumpAll(
        unordered_map[string, unordered_map[string, string]]& result)

    enum: NCCL_COLL_TUNING_VERSION
    enum: NCCL_TUNING_MAX_FUNCTIONS
    enum: NCCL_TUNING_MAX_ALGORITHMS
    enum: NCCL_TUNING_MAX_PROTOCOLS
    enum: NCCL_TUNING_SIZE_POINTS
    enum: NCCL_TUNING_NAME_LEN

    ctypedef struct ncclCollTuningEntry:
        int8_t algorithm
        int8_t protocol
        int16_t nChannels
        int16_t nThreads
        float timeUs

    ctypedef struct ncclCollTuning:
        int version
        int nRanks
        int nNodes
        int nChannels
        int minCompCap
        int maxCompCap
        int numFunctions
        int numAlgorithms
        int numProtocols
        char functionNames[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_NAME_LEN]
        char algorithmNames[NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_NAME_LEN]
        char protocolNames[NCCL_TUNING_MAX_PROTOCOLS][NCCL_TUNING_NAME_LEN]
        float bandwidths[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_MAX_PROTOCOLS]
        float latencies[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_MAX_ALGORITHMS][NCCL_TUNING_MAX_PROTOCOLS]
        uint64_t messageSizes[NCCL_TUNING_SIZE_POINTS]
        ncclCollTuningEntry bestBySize[NCCL_TUNING_MAX_FUNCTIONS][NCCL_TUNING_SIZE_POINTS]

    ncclResult_t ncclQueryCollTuning(ncclComm_t comm, ncclCollTuning* tuning)

    ctypedef struct ncclConfig_t:
        pass


cdef extern from "nccl.h" namespace "ncclx" nogil:
    # Hints class
    cdef cppclass Hints:
        Hints()
        ncclResult_t set(const char* key, const char* val)
        ncclResult_t get(const char* key, char* val)

    ncclResult_t setGlobalHint(string key, string val)

    # Window-based RMA put
    ncclResult_t ncclPut(
        const void* originBuff, size_t count, ncclDataType_t datatype,
        int peer, size_t targetDisp, ncclWindow_t win, cudaStream_t stream)

    # Live comm reconfiguration
    ncclResult_t commSetConfig(ncclComm_t comm, const ncclConfig_t* config)


cdef extern from "nccl.h" namespace "ncclx::colltrace" nogil:
    cdef enum class LifecycleEventType:
        Enqueue
        Start
        End

    cdef cppclass LifecycleEvent:
        uint64_t replayId
        uint64_t commId
        uint64_t collId
        uint64_t executionCollId
        LifecycleEventType eventType
        double timestamp

    ncclResult_t getCollTraceCommId(ncclComm_t comm, uint64_t& comm_id)
    ncclResult_t getLatestCollTraceCollectiveId(
        ncclComm_t comm, uint64_t& coll_id)
    ncclResult_t drainUnreadLifecycleEvents(vector[LifecycleEvent]& events)
