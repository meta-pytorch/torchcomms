# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# NCCLx C++ namespace bindings (direct linkage)

from libc.stdint cimport intptr_t, uint64_t
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from .ncclx_internal cimport (
    commSetConfig as _commSetConfig,
    cudaStream_t,
    drainUnreadLifecycleEvents as _drainUnreadLifecycleEvents,
    getCollTraceCommId as _getCollTraceCommId,
    getLatestCollTraceCollectiveId as _getLatestCollTraceCollectiveId,
    Hints as CppHints,
    LifecycleEvent as CppLifecycleEvent,
    LifecycleEventType as CppLifecycleEventType,
    NCCL_TUNING_SIZE_POINTS,
    ncclAllToAllv as _ncclAllToAllv,
    ncclCollTuning,
    ncclCommDump as _ncclCommDump,
    ncclCommDumpAll as _ncclCommDumpAll,
    ncclComm_t,
    ncclConfig_t,
    ncclDataType_t,
    ncclPut as _ncclPut,
    ncclQueryCollTuning as _ncclQueryCollTuning,
    ncclRedOp_t,
    ncclReduceScatterQuantize as _ncclReduceScatterQuantize,
    ncclWindow_t,
    ncclWinAttr,
    ncclWinGetAttributes as _ncclWinGetAttributes,
    ncclWinSharedQuery as _ncclWinSharedQuery,
    setGlobalHint as _setGlobalHint,
)

from .nccl import check_status


cpdef put(
    intptr_t origin_buff, size_t count, int datatype,
    int peer, size_t target_disp, intptr_t win, intptr_t stream,
):
    cdef int status
    with nogil:
        status = _ncclPut(
            <const void*>origin_buff, count, <ncclDataType_t>datatype,
            peer, target_disp, <ncclWindow_t>win, <cudaStream_t>stream,
        )
    check_status(status)


cpdef intptr_t win_shared_query(
    int rank, intptr_t comm, intptr_t win,
) except? 0:
    cdef void* addr = NULL
    cdef int status
    with nogil:
        status = _ncclWinSharedQuery(
            rank, <ncclComm_t>comm, <ncclWindow_t>win, &addr,
        )
    check_status(status)
    return <intptr_t>addr


cpdef int win_get_attributes(int rank, intptr_t win) except? -1:
    # ncclWinGetAttributes heap-allocates the attr struct and transfers
    # ownership to the caller through the out-pointer, so read the value out
    # of the returned pointer (not a local) and free it here.
    cdef ncclWinAttr* attr_ptr = NULL
    cdef int status
    cdef int access_type
    with nogil:
        status = _ncclWinGetAttributes(rank, <ncclWindow_t>win, &attr_ptr)
    check_status(status)
    access_type = <int>attr_ptr.accessType
    del attr_ptr
    return access_type


cdef class NcclxHints:
    cdef CppHints _hints

    def __init__(self, dict hints=None):
        if hints:
            for k, v in hints.items():
                _check_hints_status(self._hints.set(
                    k.encode("utf-8"), _to_hint_str(v).encode("utf-8"),
                ))

    cdef CppHints* ptr(self):
        return &self._hints

    def as_ptr(self) -> int:
        return <intptr_t>&self._hints


def _to_hint_str(v) -> str:
    """Stringify a hint value for the C++ Hints map.

    NCCLX hints are string-typed at the C++ level. This helper accepts
    natural Python types and renders them in the canonical hint format:
      - bool -> "true"/"false" (lowercase)
      - everything else -> str(v)
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


cdef _check_hints_status(int status):
    check_status(status)


cpdef set_global_hint(str key, str value):
    cdef string k = key.encode("utf-8")
    cdef string v = value.encode("utf-8")
    cdef int status
    with nogil:
        status = _setGlobalHint(k, v)
    check_status(status)


cpdef reduce_scatter_quantize(
    intptr_t sendbuff, intptr_t recvbuff, size_t recvcount,
    int input_type, int transport_type, int op,
    intptr_t seed_ptr, intptr_t comm, intptr_t stream,
):
    cdef int status
    with nogil:
        status = _ncclReduceScatterQuantize(
            <const void*>sendbuff, <void*>recvbuff, recvcount,
            <ncclDataType_t>input_type, <ncclDataType_t>transport_type,
            <ncclRedOp_t>op, <uint64_t*>seed_ptr,
            <ncclComm_t>comm, <cudaStream_t>stream,
        )
    check_status(status)


cpdef allto_allv(
    intptr_t sendbuff, intptr_t sendcounts, intptr_t sdispls,
    intptr_t recvbuff, intptr_t recvcounts, intptr_t rdispls,
    int datatype, intptr_t comm, intptr_t stream,
):
    cdef int status
    with nogil:
        status = _ncclAllToAllv(
            <const void*>sendbuff, <const size_t*>sendcounts, <const size_t*>sdispls,
            <void*>recvbuff, <const size_t*>recvcounts, <const size_t*>rdispls,
            <ncclDataType_t>datatype, <ncclComm_t>comm, <cudaStream_t>stream,
        )
    check_status(status)


cpdef dict comm_dump(intptr_t comm):
    cdef unordered_map[string, string] result
    cdef int status
    with nogil:
        status = _ncclCommDump(<ncclComm_t>comm, result)
    check_status(status)
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in result}


cpdef dict comm_dump_all():
    cdef unordered_map[string, unordered_map[string, string]] result
    cdef int status
    with nogil:
        status = _ncclCommDumpAll(result)
    check_status(status)
    return {
        k.decode("utf-8"): {
            ik.decode("utf-8"): iv.decode("utf-8") for ik, iv in v.items()
        }
        for k, v in result
    }


cdef list _colltrace_events_to_python(vector[CppLifecycleEvent]& result):
    cdef uint64_t invalid_replay_id = <uint64_t>-1
    cdef list events = []
    cdef object event_type
    for event in result:
        if event.eventType == CppLifecycleEventType.Enqueue:
            event_type = "enqueue"
        elif event.eventType == CppLifecycleEventType.Start:
            event_type = "start"
        else:
            event_type = "end"
        events.append((
            None if event.replayId == invalid_replay_id else event.replayId,
            event.commId,
            event.collId,
            event.executionCollId,
            event_type,
            event.timestamp,
        ))
    return events


cpdef dict query_coll_tuning(intptr_t comm):
    """Snapshot of the communicator's collective tuning model.

    Returns a dict:
      - "version", "n_ranks", "n_nodes", "n_channels", "min_comp_cap",
        "max_comp_cap": ints
      - "functions", "algorithms", "protocols": name lists; indexes below
        refer to these
      - "bandwidths", "latencies": {function: [algorithm][protocol]} nested
        lists -- the raw init-time model (GB/s and us; bandwidth 0 means the
        combination is disabled). Per-size correction factors NOT included.
      - "best_by_size": {function: [(size_bytes, algorithm, protocol,
        n_channels, n_threads, time_us), ...]} -- the selection and predicted
        time (all correction factors included) a collective of that size
        would run with; algorithm/protocol are None and time_us is -1.0 when
        no combination is available. Linear interpolation between adjacent
        sizes reproduces the model within a selection regime.
    """
    cdef ncclCollTuning tuning
    cdef int status
    cdef int f, a, p, s
    with nogil:
        status = _ncclQueryCollTuning(<ncclComm_t>comm, &tuning)
    check_status(status)

    functions = [
        (<bytes>tuning.functionNames[f]).decode("utf-8")
        for f in range(tuning.numFunctions)
    ]
    algorithms = [
        (<bytes>tuning.algorithmNames[a]).decode("utf-8")
        for a in range(tuning.numAlgorithms)
    ]
    protocols = [
        (<bytes>tuning.protocolNames[p]).decode("utf-8")
        for p in range(tuning.numProtocols)
    ]

    bandwidths = {}
    latencies = {}
    best_by_size = {}
    for f in range(tuning.numFunctions):
        bandwidths[functions[f]] = [
            [tuning.bandwidths[f][a][p] for p in range(tuning.numProtocols)]
            for a in range(tuning.numAlgorithms)
        ]
        latencies[functions[f]] = [
            [tuning.latencies[f][a][p] for p in range(tuning.numProtocols)]
            for a in range(tuning.numAlgorithms)
        ]
        entries = []
        for s in range(NCCL_TUNING_SIZE_POINTS):
            if tuning.bestBySize[f][s].algorithm < 0:
                entries.append(
                    (tuning.messageSizes[s], None, None, 0, 0, -1.0)
                )
            else:
                entries.append((
                    tuning.messageSizes[s],
                    algorithms[tuning.bestBySize[f][s].algorithm],
                    protocols[tuning.bestBySize[f][s].protocol],
                    tuning.bestBySize[f][s].nChannels,
                    tuning.bestBySize[f][s].nThreads,
                    tuning.bestBySize[f][s].timeUs,
                ))
        best_by_size[functions[f]] = entries

    return {
        "version": tuning.version,
        "n_ranks": tuning.nRanks,
        "n_nodes": tuning.nNodes,
        "n_channels": tuning.nChannels,
        "min_comp_cap": tuning.minCompCap,
        "max_comp_cap": tuning.maxCompCap,
        "functions": functions,
        "algorithms": algorithms,
        "protocols": protocols,
        "bandwidths": bandwidths,
        "latencies": latencies,
        "best_by_size": best_by_size,
    }


cpdef uint64_t colltrace_get_comm_id(intptr_t comm) except? 0:
    cdef uint64_t comm_id
    cdef int status
    with nogil:
        status = _getCollTraceCommId(<ncclComm_t>comm, comm_id)
    check_status(status)
    return comm_id


cpdef uint64_t colltrace_get_latest_coll_id(intptr_t comm) except? 0:
    cdef uint64_t coll_id
    cdef int status
    with nogil:
        status = _getLatestCollTraceCollectiveId(<ncclComm_t>comm, coll_id)
    check_status(status)
    return coll_id


cpdef list colltrace_get_unread_events():
    cdef vector[CppLifecycleEvent] result
    cdef int status
    with nogil:
        status = _drainUnreadLifecycleEvents(result)
    check_status(status)
    return _colltrace_events_to_python(result)


cpdef comm_set_config(intptr_t comm, intptr_t config):
    cdef int status
    with nogil:
        status = _commSetConfig(
            <ncclComm_t>comm, <const ncclConfig_t*>config,
        )
    check_status(status)
