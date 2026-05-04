# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

# torchcomms uses fmt via its headers. Keep the dependency header-only so we
# do not require libfmt binaries in local builds or shipped wheels.
function(_torchcomms_configure_header_only_fmt include_dir)
    add_library(fmt::fmt INTERFACE IMPORTED GLOBAL)
    set_target_properties(fmt::fmt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${include_dir}"
        INTERFACE_COMPILE_DEFINITIONS "FMT_HEADER_ONLY=1"
    )
endfunction()

find_path(_FMT_INCLUDE_DIR
    NAMES fmt/format.h
    HINTS "${CONDA_INCLUDE}"
)

if(_FMT_INCLUDE_DIR)
    _torchcomms_configure_header_only_fmt("${_FMT_INCLUDE_DIR}")
    message(STATUS "Using header-only fmt from: ${_FMT_INCLUDE_DIR}")
else()
    message(STATUS "fmt headers not found, fetching 11.2.0 header-only via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 11.2.0
    )
    FetchContent_Populate(fmt)
    _torchcomms_configure_header_only_fmt("${fmt_SOURCE_DIR}/include")
endif()
