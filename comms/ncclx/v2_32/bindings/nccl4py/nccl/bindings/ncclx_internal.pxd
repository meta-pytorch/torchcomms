# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# NCCLX C++ namespace bindings (direct linkage, not dlsym).

from libc.stdint cimport uint64_t
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
    cdef cppclass ncclWinAttr:
        ncclWinAccessType accessType
    ctypedef ncclWinAttr* ncclWinAttr_t

    ncclResult_t ncclWinSharedQuery(
        int rank, ncclComm_t comm, ncclWindow_t win, void** addr)
    ncclResult_t ncclWinGetAttributes(
        int rank, ncclWindow_t win, ncclWinAttr_t* attr)
    ncclResult_t ncclCommDump(
        ncclComm_t comm, unordered_map[string, string]& result)
    ncclResult_t ncclCommDumpAll(
        unordered_map[string, unordered_map[string, string]]& result)

    ctypedef struct ncclConfig_t:
        pass


cdef extern from *:
    """
    #include "nccl.h"
    static inline void nccl4pyDeleteWinAttr(ncclWinAttr* attr) {
      delete attr;
    }
    """
    void nccl4pyDeleteWinAttr(ncclWinAttr* attr) noexcept nogil


cdef extern from "nccl.h" namespace "ncclx" nogil:
    cdef cppclass Hints:
        Hints()
        ncclResult_t set(const char* key, const char* val)
        ncclResult_t get(const char* key, char* val)

    ncclResult_t ncclPut(
        const void* originBuff, size_t count, ncclDataType_t datatype,
        int peer, size_t targetDisp, ncclWindow_t win, cudaStream_t stream)
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
