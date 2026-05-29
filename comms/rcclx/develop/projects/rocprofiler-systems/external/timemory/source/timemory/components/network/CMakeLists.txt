if(NOT UNIX OR APPLE)
    return()
endif()

set(NAME network)

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

timemory_build_intermediate_library(
    NAME ${NAME}
    TARGET ${NAME}-component
    CATEGORY COMPONENT
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    HEADERS ${header_files}
    SOURCES ${source_files}
    PROPERTY_DEPENDS GLOBAL)
