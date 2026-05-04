file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

get_filename_component(folder_name "${CMAKE_CURRENT_LIST_DIR}" NAME)

timemory_build_intermediate_library(
    USE_INTERFACE
    NAME tpls
    TARGET core
    CATEGORY GLOBAL
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    HEADERS ${header_files}
    SOURCES ${source_files}
    DEPENDS timemory::timemory-yaml $<TARGET_NAME_IF_EXISTS:timemory::timemory-yaml-cpp>
    PROPERTY_DEPENDS GLOBAL)

list(REMOVE_ITEM source_files ${CMAKE_CURRENT_SOURCE_DIR}/extern.cpp)
timemory_install_header_files(${source_files})
