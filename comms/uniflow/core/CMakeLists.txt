# Copyright (c) Meta Platforms, Inc. and affiliates.

set(UNIFLOW_CORE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Func.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MpscQueue.h
    ${CMAKE_CURRENT_SOURCE_DIR}/NumaUtils.h
    PARENT_SCOPE
)

set(UNIFLOW_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/NumaUtils.cpp
    PARENT_SCOPE
)
