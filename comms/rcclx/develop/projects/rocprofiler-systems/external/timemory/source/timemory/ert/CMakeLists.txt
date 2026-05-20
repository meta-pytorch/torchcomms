if(NOT TIMEMORY_BUILD_ERT)
    return()
endif()

set(CMAKE_UNITY_BUILD OFF)
set(_DEPENDS)

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE cu_source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE hip_source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hip)

if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    list(APPEND source_files ${cu_source_files})
elseif(TIMEMORY_USE_HIP)
    set_source_files_properties(${hip_source_files} PROPERTIES LANGUAGE CXX)
    add_library(timemory-ert-hip-object EXCLUDE_FROM_ALL OBJECT ${hip_source_files})
    target_link_libraries(
        timemory-ert-hip-object
        PRIVATE timemory::timemory-vector timemory::timemory-hip timemory::timemory-core
                timemory::timemory-roofline-options hip::device)
    set_target_properties(timemory-ert-hip-object PROPERTIES POSITION_INDEPENDENT_CODE ON)
    list(APPEND source_files $<TARGET_OBJECTS:timemory-ert-hip-object>)
endif()

timemory_build_intermediate_library(
    USE_INTERFACE
    NAME ert
    TARGET ert
    CATEGORY GLOBAL
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    HEADERS ${header_files}
    SOURCES ${source_files}
    DEPENDS timemory::timemory-vector timemory::timemory-cuda timemory::timemory-hip
            timemory::timemory-core
    PRIVATE_LINK timemory::timemory-roofline-options)
