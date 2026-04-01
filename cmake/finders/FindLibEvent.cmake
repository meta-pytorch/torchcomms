# Copyright (c) Meta Platforms, Inc. and affiliates.
# Stub Find module: short-circuits when 'event' target exists (FetchContent libevent).
# Replaces folly/fizz/wangle/fbthrift's bundled FindLibEvent.cmake which uses
# CMP0026 (get_target_property LOCATION) that doesn't work with build targets.

if(TARGET event)
    set(LibEvent_FOUND TRUE)
    set(LIBEVENT_FOUND TRUE)
    set(LIBEVENT_LIB event)
    if(NOT LIBEVENT_INCLUDE_DIR)
        get_target_property(LIBEVENT_INCLUDE_DIR event INTERFACE_INCLUDE_DIRECTORIES)
    endif()
    return()
endif()

# Fallback: system search
find_path(LIBEVENT_INCLUDE_DIR event2/event.h)
find_library(LIBEVENT_LIB NAMES event)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibEvent DEFAULT_MSG LIBEVENT_LIB LIBEVENT_INCLUDE_DIR)
