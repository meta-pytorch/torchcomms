# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# NCCLX C++ namespace bindings (direct linkage).

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
    nccl4pyDeleteWinAttr as _nccl4pyDeleteWinAttr,
    ncclCommDump as _ncclCommDump,
    ncclCommDumpAll as _ncclCommDumpAll,
    ncclComm_t,
    ncclConfig_t,
    ncclDataType_t,
    ncclPut as _ncclPut,
    ncclWindow_t,
    ncclWinAttr,
    ncclWinGetAttributes as _ncclWinGetAttributes,
    ncclWinSharedQuery as _ncclWinSharedQuery,
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
    cdef ncclWinAttr* attr_ptr = NULL
    cdef int status
    cdef int access_type
    with nogil:
        status = _ncclWinGetAttributes(rank, <ncclWindow_t>win, &attr_ptr)
    check_status(status)
    if attr_ptr == NULL:
        raise RuntimeError("ncclWinGetAttributes returned a null attribute")
    access_type = <int>attr_ptr.accessType
    with nogil:
        _nccl4pyDeleteWinAttr(attr_ptr)
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
        """Return the native pointer; the ``NcclxHints`` owner must stay alive."""
        return <intptr_t>&self._hints


def _to_hint_str(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


cdef _check_hints_status(int status):
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


cpdef uint64_t colltrace_get_comm_id(intptr_t comm) except? 0:
    cdef uint64_t comm_id = 0
    cdef int status
    with nogil:
        status = _getCollTraceCommId(<ncclComm_t>comm, comm_id)
    check_status(status)
    return comm_id


cpdef uint64_t colltrace_get_latest_coll_id(intptr_t comm) except? 0:
    cdef uint64_t coll_id = 0
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
