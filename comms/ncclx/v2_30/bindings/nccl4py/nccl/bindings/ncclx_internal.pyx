# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# NCCLx C++ namespace bindings (direct linkage)

from libc.stdint cimport intptr_t, uint64_t
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from .ncclx_internal cimport (
    commSetConfig as _commSetConfig,
    cudaStream_t,
    Hints as CppHints,
    ncclAllToAllv as _ncclAllToAllv,
    ncclCommDump as _ncclCommDump,
    ncclCommDumpAll as _ncclCommDumpAll,
    ncclComm_t,
    ncclConfig_t,
    ncclDataType_t,
    ncclPut as _ncclPut,
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


cpdef comm_set_config(intptr_t comm, intptr_t config):
    cdef int status
    with nogil:
        status = _commSetConfig(
            <ncclComm_t>comm, <const ncclConfig_t*>config,
        )
    check_status(status)
