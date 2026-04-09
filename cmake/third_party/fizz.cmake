# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(fizz REQUIRED)
elseif(CONDA_PREFIX)
    find_package(fizz QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET fizz::fizz)
    include(FetchContent)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        fizz
        GIT_REPOSITORY https://github.com/facebookincubator/fizz.git
        GIT_TAG ${TORCHCOMMS_FIZZ_VERSION}
        GIT_SHALLOW TRUE
    )
    # Use Populate + add_subdirectory so we can patch fizz's FindSodium
    # before fizz's CMakeLists.txt runs. fizz's bundled FindSodium.cmake
    # unsets our pre-set sodium cache variables via a change-tracking guard.
    FetchContent_Populate(fizz)
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindSodium.cmake"
         DESTINATION "${fizz_SOURCE_DIR}/build/fbcode_builder/CMake/")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindZstd.cmake"
         DESTINATION "${fizz_SOURCE_DIR}/build/fbcode_builder/CMake/")
    add_subdirectory(${fizz_SOURCE_DIR}/fizz ${fizz_BINARY_DIR} EXCLUDE_FROM_ALL)

    # fizz uses $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..> for its include path,
    # but CMAKE_SOURCE_DIR is our root, not fizz's.  Fix the include path.
    target_include_directories(fizz PUBLIC "${fizz_SOURCE_DIR}")

    # fizz's CMakeLists.txt creates a non-namespaced "fizz" target.
    # Downstream libs (fbthrift) expect fizz::fizz.
    if(NOT TARGET fizz::fizz)
        add_library(fizz::fizz ALIAS fizz)
    endif()

    # Make find_package(fizz CONFIG) succeed for downstream libs.
    set(fizz_FOUND TRUE CACHE BOOL "" FORCE)
    set(fizz_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config-stubs" CACHE PATH "" FORCE)
    # wangle uses ${FIZZ_INCLUDE_DIR} in target_include_directories.
    # Include both source (for headers) and binary/generated (for fizz-config.h).
    set(FIZZ_INCLUDE_DIR "${fizz_SOURCE_DIR};${fizz_BINARY_DIR}/generated" CACHE PATH "" FORCE)
endif()
