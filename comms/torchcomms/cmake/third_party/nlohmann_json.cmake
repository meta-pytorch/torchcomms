# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

find_package(nlohmann_json 3.11.3 QUIET CONFIG NO_CMAKE_PACKAGE_REGISTRY)
if(nlohmann_json_FOUND)
    message(STATUS "Found system nlohmann_json: ${nlohmann_json_VERSION}")
else()
    message(STATUS "System nlohmann_json not found, fetching v3.11.3 via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_Populate(nlohmann_json)
    add_subdirectory(${nlohmann_json_SOURCE_DIR} ${nlohmann_json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
