# Copyright (c) Meta Platforms, Inc. and affiliates.

include_guard(GLOBAL)

if(USE_SYSTEM_LIBS)
    find_package(folly REQUIRED)
elseif(CONDA_PREFIX)
    find_package(folly QUIET HINTS ${CONDA_PREFIX} NO_DEFAULT_PATH)
endif()

if(NOT TARGET Folly::folly)
    include(FetchContent)
    # Don't force static OpenSSL — conda/system may only have shared libs.
    set(OPENSSL_USE_STATIC_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        folly
        GIT_REPOSITORY https://github.com/facebook/folly.git
        GIT_TAG ${TORCHCOMMS_FOLLY_VERSION}
        GIT_SHALLOW TRUE
    )
    # Use Populate + add_subdirectory so we can patch folly's FindLibEvent.
    # The bundled version uses get_target_property(LOCATION) which returns
    # empty for non-imported build targets from FetchContent.
    FetchContent_Populate(folly)
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindLibEvent.cmake"
         DESTINATION "${folly_SOURCE_DIR}/build/fbcode_builder/CMake/")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindZstd.cmake"
         DESTINATION "${folly_SOURCE_DIR}/build/fbcode_builder/CMake/")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/finders/FindZstd.cmake"
         DESTINATION "${folly_SOURCE_DIR}/CMake/")
    # Patch Demangle.cpp: glog leaks src/demangle.h as a PUBLIC include,
    # causing __has_include(<demangle.h>) to find glog's version instead of
    # libiberty's. Disable the libiberty code path unconditionally.
    file(READ "${folly_SOURCE_DIR}/folly/Demangle.cpp" _demangle_content)
    string(REPLACE "__has_include(<demangle.h>)" "0" _demangle_content "${_demangle_content}")
    file(WRITE "${folly_SOURCE_DIR}/folly/Demangle.cpp" "${_demangle_content}")
    add_subdirectory(${folly_SOURCE_DIR} ${folly_BINARY_DIR} EXCLUDE_FROM_ALL)

    # folly's CMakeLists.txt creates a non-namespaced "folly" target.
    # Downstream libs (fizz, wangle, fbthrift) expect Folly::folly.
    if(NOT TARGET Folly::folly)
        add_library(Folly::folly ALIAS folly)
    endif()
    if(NOT TARGET Folly::folly_test_util AND TARGET folly_test_util)
        add_library(Folly::folly_test_util ALIAS folly_test_util)
    endif()

    # Make find_package(folly CONFIG) succeed for downstream FB OSS libs.
    set(folly_FOUND TRUE CACHE BOOL "" FORCE)
    set(folly_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config-stubs" CACHE PATH "" FORCE)
    # fizz/wangle use ${FOLLY_INCLUDE_DIR} in target_include_directories:
    # Include both source and binary dirs — folly-config.h is generated in the build dir.
    set(FOLLY_INCLUDE_DIR "${folly_SOURCE_DIR};${folly_BINARY_DIR}" CACHE PATH "" FORCE)
    set(FOLLY_LIBRARIES Folly::folly CACHE STRING "" FORCE)
    set(FOLLY_LIBRARY Folly::folly CACHE STRING "" FORCE)
endif()
