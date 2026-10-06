# Copyright (c) Meta Platforms, Inc. and affiliates.

set(_UNIFLOW_PUBLIC_HEADER_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

set(UNIFLOW_PUBLIC_HEADERS
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/Connection.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/MultiTransport.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/Result.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/Segment.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/Uniflow.h

    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/controller/Controller.h

    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/core/Func.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/core/MpscQueue.h

    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/executor/EpollEventBase.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/executor/EventBase.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/executor/LockFreeEventBase.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/executor/LockFreeQueuePolicy.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/executor/MutexEventBase.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/executor/MutexQueuePolicy.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/executor/ScopedEventBaseThread.h

    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/logging/Logger.h

    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/transport/Topology.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/transport/Transport.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/transport/TransportType.h
)

if(UNIFLOW_ENABLE_TCP)
  list(APPEND UNIFLOW_PUBLIC_HEADERS
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/controller/TcpController.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/transport/tcp/TcpPinnedSlabPool.h
    ${_UNIFLOW_PUBLIC_HEADER_ROOT}/transport/tcp/TcpTransport.h
  )
endif()

unset(_UNIFLOW_PUBLIC_HEADER_ROOT)
