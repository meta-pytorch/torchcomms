if(NOT timemory_MAIN_PROJECT
   OR NOT TIMEMORY_BUILD_ERT
   OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

# project(timemory-ert-tool LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/*.hpp")
file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

if(TIMEMORY_USE_CUDA)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
endif()

add_executable(timemory-ert-exe ${_EXCLUDE} ${sources} ${headers})

target_link_libraries(
    timemory-ert-exe
    PRIVATE timemory::timemory-headers
            timemory::timemory-compile-options
            timemory::timemory-vector
            timemory::timemory-arch
            timemory::timemory-dmp
            timemory::timemory-cuda
            timemory::timemory-cuda-compiler
            timemory::timemory-hip
            timemory::timemory-core
            timemory::timemory-common
            timemory::timemory-extensions)

if(TARGET timemory::timemory-ert)
    target_link_libraries(timemory-ert-exe PRIVATE timemory::timemory-ert)
endif()

if(UNIX AND NOT APPLE)
    set(_RPATH INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

set_target_properties(timemory-ert-exe PROPERTIES OUTPUT_NAME timemory-ert ${_RPATH})

install(
    TARGETS timemory-ert-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tools
    OPTIONAL)
