file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

get_filename_component(folder_name "${CMAKE_CURRENT_LIST_DIR}" NAME)

timemory_build_intermediate_library(
    USE_INTERFACE INSTALL_SOURCE
    NAME ${folder_name}
    TARGET ${folder_name}
    CATEGORY GLOBAL
    FOLDER ${CMAKE_INSTALL_LIBDIR}
    HEADERS ${header_files}
    SOURCES ${source_files}
    PROPERTY_DEPENDS GLOBAL)
