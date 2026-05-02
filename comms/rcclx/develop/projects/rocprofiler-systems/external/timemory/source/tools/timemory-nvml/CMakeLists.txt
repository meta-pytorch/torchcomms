if(NOT TIMEMORY_USE_NVML)
    return()
endif()

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

if(NOT DEFINED TIMEMORY_BUILD_NVML)
    set(TIMEMORY_BUILD_NVML
        ON
        CACHE BOOL "Build NVML exe")
endif()

# ----------------------------------------------------------------------------------------#
# Build and install nvml tool
#
if(NOT TIMEMORY_BUILD_NVML OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    # if not enabled as option, make it available to be built and installed
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

add_executable(
    timemory-nvml-exe
    ${_EXCLUDE}
    nvml.cpp
    nvml.hpp
    nvml_types.hpp
    nvml_utilization_rate.cpp
    nvml_utilization_rate.hpp
    nvml_temperature.cpp
    nvml_temperature.hpp
    nvml_memory_info.cpp
    nvml_memory_info.hpp
    nvml_processes.cpp
    nvml_processes.hpp
    nvml_backends.hpp
    parse_args.cpp
    monitor.cpp)

target_link_libraries(
    timemory-nvml-exe
    PRIVATE timemory::timemory-compile-options timemory::timemory-develop-options
            timemory::timemory-threading timemory::timemory-headers
            timemory::timemory-nvml)

add_target_flag_if_avail(timemory-nvml-exe "-g")

set_target_properties(
    timemory-nvml-exe PROPERTIES OUTPUT_NAME timemory-nvml INSTALL_RPATH_USE_LINK_PATH
                                                           ${NVML_LIBRARY_HAS_SYMBOLS})

install(
    TARGETS timemory-nvml-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tools
    OPTIONAL)
