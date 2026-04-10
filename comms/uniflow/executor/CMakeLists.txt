# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_EXECUTOR_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/EventBase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/EpollEventBase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MutexQueuePolicy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LockFreeQueuePolicy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MutexEventBase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LockFreeEventBase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ScopedEventBaseThread.h
)

add_library(uniflow_executor INTERFACE)

target_sources(uniflow_executor INTERFACE
    ${UNIFLOW_EXECUTOR_HEADERS}
)

target_compile_features(uniflow_executor INTERFACE cxx_std_20)

target_include_directories(uniflow_executor INTERFACE
    ${ROOT}
)

target_link_libraries(uniflow_executor INTERFACE
    Threads::Threads
    uniflow_logging
)
