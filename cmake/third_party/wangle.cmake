# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(wangle REQUIRED)
elseif(CONDA_PREFIX)
    find_package(wangle QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET wangle::wangle)
    include(FetchContent)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        wangle
        GIT_REPOSITORY https://github.com/facebook/wangle.git
        GIT_TAG ${TORCHCOMMS_WANGLE_VERSION}
        GIT_SHALLOW TRUE
    )
    # Use Populate + add_subdirectory so we can patch wangle's Find modules:
    # - FindSodium unsets our pre-set cache vars via change-tracking guard
    # - FindLibEvent uses get_target_property(LOCATION) which fails on build targets
    FetchContent_Populate(wangle)
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindSodium.cmake"
         DESTINATION "${wangle_SOURCE_DIR}/build/fbcode_builder/CMake/")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindLibEvent.cmake"
         DESTINATION "${wangle_SOURCE_DIR}/build/fbcode_builder/CMake/")
    add_subdirectory(${wangle_SOURCE_DIR}/wangle ${wangle_BINARY_DIR} EXCLUDE_FROM_ALL)

    # wangle uses $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..> for its include
    # path, but CMAKE_SOURCE_DIR is our root, not wangle's.  Fix the include
    # path so #include <wangle/...> works.
    target_include_directories(wangle PUBLIC "${wangle_SOURCE_DIR}")

    # wangle's CMakeLists.txt creates a non-namespaced "wangle" target.
    # Downstream libs (fbthrift) expect wangle::wangle.
    if(NOT TARGET wangle::wangle)
        add_library(wangle::wangle ALIAS wangle)
    endif()

    # Make find_package(wangle CONFIG) succeed for downstream libs.
    set(wangle_FOUND TRUE CACHE BOOL "" FORCE)
    set(wangle_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config-stubs" CACHE PATH "" FORCE)
endif()
