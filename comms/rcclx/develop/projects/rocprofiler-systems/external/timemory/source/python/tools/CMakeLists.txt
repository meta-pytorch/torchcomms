# ########################################################################################
#
# pytimem tool
#
# ########################################################################################

if(NOT TIMEMORY_BUILD_TOOLS)
    return()
endif()

# copy over the python file
configure_file(${CMAKE_CURRENT_LIST_DIR}/timem.py ${PROJECT_BINARY_DIR}/pytimem @ONLY)

# timem installation
install(
    PROGRAMS ${PROJECT_BINARY_DIR}/pytimem
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
