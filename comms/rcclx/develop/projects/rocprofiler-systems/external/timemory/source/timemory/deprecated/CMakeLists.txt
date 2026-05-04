file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(TIMEMORY_USE_DEPRECATED)
    timemory_build_intermediate_library(
        USE_INTERFACE INSTALL_SOURCE
        NAME deprecated
        TARGET deprecated
        CATEGORY GLOBAL
        FOLDER ${CMAKE_INSTALL_LIBDIR}
        HEADERS ${header_files}
        SOURCES ${source_files}
        PROPERTY_DEPENDS GLOBAL)
else()
    timemory_install_header_files(${header_files} ${source_files})
endif()
