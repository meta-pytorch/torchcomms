if(DEFINED TIMEMORY_BUILD_CONTAINERS AND NOT TIMEMORY_BUILD_CONTAINERS)
    return()
endif()

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(${source_files} PROPERTIES # LANGUAGE CUDA
                                                           LINKER_LANGUAGE CUDA)
endif()

timemory_build_intermediate_library(
    USE_INTERFACE
    NAME containers
    TARGET containers
    CATEGORY EXTERN
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    HEADERS ${header_files}
    SOURCES ${source_files}
    PROPERTY_DEPENDS GLOBAL COMPONENT USER_BUNDLE)
