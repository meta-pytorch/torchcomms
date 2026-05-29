if(WIN32)
    return()
endif()

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

timemory_build_intermediate_library(
    USE_INTERFACE INSTALL_SOURCE
    NAME unwind
    TARGET core
    CATEGORY GLOBAL
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    HEADERS ${header_files}
    SOURCES ${source_files}
    DEPENDS timemory::timemory-libunwind timemory::timemory-bfd
    PROPERTY_DEPENDS GLOBAL)
