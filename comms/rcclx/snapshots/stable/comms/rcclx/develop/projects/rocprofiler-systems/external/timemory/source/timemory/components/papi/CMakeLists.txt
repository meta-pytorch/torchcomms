set(NAME papi)
set(DEPS timemory::timemory-papi)

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(TIMEMORY_USE_PAPI)
    timemory_build_intermediate_library(
        USE_INTERFACE
        NAME ${NAME}
        TARGET ${NAME}-component
        CATEGORY COMPONENT
        FOLDER ${CMAKE_INSTALL_LIBDIR}
        HEADERS ${header_files}
        SOURCES ${source_files}
        DEPENDS ${DEPS}
        PROPERTY_DEPENDS GLOBAL)
endif()

list(REMOVE_ITEM source_files ${CMAKE_CURRENT_SOURCE_DIR}/extern.cpp)
timemory_install_header_files(${source_files})
