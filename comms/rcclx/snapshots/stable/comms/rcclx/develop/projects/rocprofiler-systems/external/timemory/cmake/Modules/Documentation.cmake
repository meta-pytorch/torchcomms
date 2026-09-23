#
# Create a "make doc" target using Doxygen Prototype:
# GENERATE_DOCUMENTATION(doxygen_config_file) Parameters: doxygen_config_file    Doxygen
# configuration file (must in the root of the source directory)

include(MacroUtilities)
include(CMakeDependentOption)

if(NOT "${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    return()
endif()

if(TIMEMORY_BUILD_DOCS)
    find_package(Doxygen)
    if(NOT Doxygen_FOUND)
        message(STATUS "Doxygen executable cannot be found. Disable TIMEMORY_BUILD_DOCS")
        set(TIMEMORY_BUILD_DOCS OFF)
    endif()
endif()

# ----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_DOCS AND Doxygen_FOUND)

    get_property(TIMEMORY_DOXYGEN_DEFINE GLOBAL PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES)
    get_property(TIMEMORY_CMAKE_OPTIONS GLOBAL PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC)
    get_property(TIMEMORY_CMAKE_INTERFACES GLOBAL
                 PROPERTY ${PROJECT_NAME}_CMAKE_INTERFACE_DOC)

    list(SORT TIMEMORY_CMAKE_OPTIONS)
    list(SORT TIMEMORY_CMAKE_INTERFACES)
    string(REPLACE ";" "\n| `" TIMEMORY_CMAKE_OPTIONS "${TIMEMORY_CMAKE_OPTIONS}")
    string(REPLACE ";" "\n| `" TIMEMORY_CMAKE_INTERFACES "${TIMEMORY_CMAKE_INTERFACES}")
    set(TIMEMORY_CMAKE_OPTIONS "| `${TIMEMORY_CMAKE_OPTIONS}")
    set(TIMEMORY_CMAKE_INTERFACES "| `${TIMEMORY_CMAKE_INTERFACES}")

    configure_file(${PROJECT_SOURCE_DIR}/docs/installation.md.in
                   ${PROJECT_SOURCE_DIR}/docs/installation.md @ONLY)
    configure_file(${PROJECT_SOURCE_DIR}/docs/getting_started/integrating.md.in
                   ${PROJECT_SOURCE_DIR}/docs/getting_started/integrating.md @ONLY)

    list(SORT TIMEMORY_DOXYGEN_DEFINE)
    string(REPLACE ";" " " TIMEMORY_DOXYGEN_DEFINE "${TIMEMORY_DOXYGEN_DEFINE}")

    # if BUILD_DOXYGEN_DOCS = ON, we want to build docs quietly else, don't build quietly
    cmake_dependent_option(
        TIMEMORY_BUILD_DOCS_QUIET "Suppress standard output when making the docs" ON
        "BUILD_DOXYGEN_DOCS" OFF)
    mark_as_advanced(TIMEMORY_BUILD_DOCS_QUIET)

    if(TIMEMORY_BUILD_DOCS_QUIET)
        set(DOXYGEN_QUIET YES)
    else()
        set(DOXYGEN_QUIET NO)
    endif()

    # GraphViz dot program is used to build call graphs, caller graphs, class graphs
    find_program(GRAPHVIZ_DOT_PATH dot)
    mark_as_advanced(GRAPHVIZ_DOT_PATH)

    if("${GRAPHVIZ_DOT_PATH}" STREQUAL "GRAPHVIZ_DOT_PATH-NOTFOUND")
        set(DOXYGEN_DOT_FOUND NO)
        set(GRAPHVIZ_DOT_PATH "")
    else()
        set(DOXYGEN_DOT_FOUND YES)
        set(GRAPHVIZ_DOT_PATH ${GRAPHVIZ_DOT_PATH})
    endif()

    # available Doxygen doc formats
    set(AVAILABLE_DOXYGEN_DOC_FORMATS HTML LATEX MAN XML RTF)
    # we want HTML, LATEX, and MAN to be default
    set(_default_on "MAN")
    foreach(_doc_format ${AVAILABLE_DOXYGEN_DOC_FORMATS})
        # find if we want it on
        string(REGEX MATCH "${_doc_format}" SET_TO_ON "${_default_on}")
        # if doc format is MAN and it is not a UNIX machine --> turn off
        if("${_doc_format}" STREQUAL "MAN" AND NOT UNIX)
            set(SET_TO_ON "")
        endif()
        # set ON/OFF
        if("${SET_TO_ON}" STREQUAL "")
            set(_default "OFF")
        else()
            set(_default "ON")
        endif()
        # add option
        timemory_add_option(ENABLE_DOXYGEN_${_doc_format}_DOCS
                            "Build documentation with ${_doc_format} format" ${_default})
        mark_as_advanced(ENABLE_DOXYGEN_${_doc_format}_DOCS)
    endforeach()

    # loop over doc formats and set GENERATE_DOXYGEN_${_doc_format}_DOC to YES/NO
    # GENERATE_DOXYGEN_${_doc_format}_DOC is used in configure_file @ONLY
    foreach(_doc_format ${AVAILABLE_DOXYGEN_DOC_FORMATS})
        if(ENABLE_DOXYGEN_${_doc_format}_DOCS)
            set(GENERATE_DOXYGEN_${_doc_format}_DOC YES)
        else()
            set(GENERATE_DOXYGEN_${_doc_format}_DOC NO)
        endif()
    endforeach()

    if(DOXYGEN_DOT_FOUND)
        set(CLASS_GRAPH_DEFAULT OFF)
        set(CALL_GRAPH_DEFAULT OFF)
        set(CALLER_GRAPH_DEFAULT OFF)
        set(DOXYGEN_DOT_GRAPH_TYPES CLASS CALL CALLER)
        # options to turn generation of class, call, and caller graphs
        foreach(_graph_type ${DOXYGEN_DOT_GRAPH_TYPES})
            # create CMake doc string
            string(TOLOWER _graph_type_desc ${_graph_type})
            # add option
            timemory_add_option(ENABLE_DOXYGEN_${_graph_type}_GRAPH "${_message}"
                                ${${_graph_type}_GRAPH_DEFAULT})
            mark_as_advanced(ENABLE_DOXYGEN_${_graph_type}_GRAPH)
            # set GENERATE_DOXYGEN_${_graph_type}_GRAPH to YES/NO
            # GENERATE_DOXYGEN_${_graph_type}_GRAPH is used in configure_file @ONLY
            if(ENABLE_DOXYGEN_${_graph_type}_GRAPH)
                set(GENERATE_DOXYGEN_${_graph_type}_GRAPH YES)
            else()
                set(GENERATE_DOXYGEN_${_graph_type}_GRAPH NO)
            endif()
        endforeach()
    endif()

    # get the documentation include directories
    get_property(BUILDTREE_INCLUDE_DIRS GLOBAL
                 PROPERTY TIMEMORY_DOCUMENTATION_INCLUDE_DIRS)

    if("${BUILDTREE_INCLUDE_DIRS}" STREQUAL "")
        message(FATAL_ERROR "Property TIMEMORY_DOCUMENTATION_INCLUDE_DIRS is empty")
    endif()

    list(REMOVE_DUPLICATES BUILDTREE_INCLUDE_DIRS)
    # if the headers are in include/ and sources are in src/
    set(BUILDTREE_SOURCE_DIRS)
    foreach(_dir ${BUILDTREE_INCLUDE_DIRS})
        get_filename_component(_path "${_dir}" PATH)
        get_filename_component(_name "${_dir}" NAME)
        get_filename_component(_path "${_path}" ABSOLUTE)
        if("${_name}" STREQUAL "include")
            set(_src_dir "${_path}/src")
        endif()
        if(EXISTS "${_src_dir}")
            list(APPEND BUILDTREE_SOURCE_DIRS ${_src_dir})
        endif()
    endforeach()

    # define BUILDTREE_DIRS which is used in configure_file @ONLY
    set(BUILDTREE_DIRS ${BUILDTREE_INCLUDE_DIRS})
    # if there were src/ folders, add them
    if(NOT "${BUILDTREE_SOURCE_DIRS}" STREQUAL "")
        list(REMOVE_DUPLICATES BUILDTREE_SOURCE_DIRS)
        list(APPEND BUILDTREE_DIRS ${BUILDTREE_SOURCE_DIRS})
    endif()

    list(REMOVE_DUPLICATES BUILDTREE_DIRS)
    # Doxyfiles was spaces not semi-colon separated lists
    string(REPLACE ";" " " BUILDTREE_DIRS "${BUILDTREE_DIRS}")

    # -----------------------------------------------------------------------

    if(XCODE)
        set(GENERATE_DOCSET_IF_XCODE YES)
    else()
        set(GENERATE_DOCSET_IF_XCODE NO)
    endif()

    # -----------------------------------------------------------------------

    configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/Doxyfile.in
                   ${PROJECT_BINARY_DIR}/doc/Doxyfile.${PROJECT_NAME} @ONLY)

    if(ENABLE_DOXYGEN_HTML_DOCS)
        file(WRITE ${PROJECT_BINARY_DIR}/doc/${PROJECT_NAME}_Documentation.html
             "<meta http-equiv=\"refresh\" content=\"1;url=html/index.html\">")
    endif()

endif() # TIMEMORY_BUILD_DOCS

# ----------------------------------------------------------------------------------------#
# Macro to generate documentation from:
# http://www.cmake.org/pipermail/cmake/2007-May/014174.html
function(GENERATE_DOCUMENTATION DOXYGEN_CONFIG_FILE)

    set(DOXYFILE_FOUND false)

    if(EXISTS ${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE})
        set(DOXYFILE_FOUND true)
    else(EXISTS ${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE})
        message(
            STATUS
                "Doxygen config file was not found at ${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE}"
            )
    endif(EXISTS ${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE})

    if(DOXYGEN_FOUND)
        if(DOXYFILE_FOUND)
            # Add target
            if(TIMEMORY_BUILD_DOXYGEN)
                add_custom_target(
                    doc ALL ${DOXYGEN_EXECUTABLE}
                            "${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE}")
            else()
                add_custom_target(doc ${DOXYGEN_EXECUTABLE}
                                      "${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE}")
            endif()

            install(
                DIRECTORY ${PROJECT_BINARY_DIR}/doc/man/
                DESTINATION ${CMAKE_INSTALL_MANDIR}
                OPTIONAL
                COMPONENT documentation)

            install(
                DIRECTORY ${PROJECT_BINARY_DIR}/doc/html
                DESTINATION ${CMAKE_INSTALL_DOCDIR}
                OPTIONAL
                COMPONENT documentation)

            install(
                DIRECTORY ${PROJECT_BINARY_DIR}/doc/latex
                DESTINATION ${CMAKE_INSTALL_DOCDIR}
                OPTIONAL
                COMPONENT documentation)

            install(
                FILES ${PROJECT_BINARY_DIR}/doc/${PROJECT_NAME}_Documentation.html
                DESTINATION ${CMAKE_INSTALL_DOCDIR}
                OPTIONAL
                COMPONENT documentation)

        else(DOXYFILE_FOUND)
            message(
                STATUS
                    "Doxygen configuration file not found - Documentation will not be generated"
                )
        endif(DOXYFILE_FOUND)
    else(DOXYGEN_FOUND)
        message(STATUS "Doxygen not found - Documentation will not be generated")
    endif(DOXYGEN_FOUND)

endfunction()

# ----------------------------------------------------------------------------------------#
# Macro to generate PDF manual from LaTeX using pdflatex assumes manual is in
# ${PROJECT_SOURCE_DIR}/doc
function(GENERATE_MANUAL MANUAL_TEX MANUAL_BUILD_PATH EXTRA_FILES_TO_COPY)

    find_program(PDFLATEX pdflatex)

    if(PDFLATEX AND NOT "${PDFLATEX}" STREQUAL "PDFLATEX-NOTFOUND")
        # name with no path is given
        set(MANUAL_NAME ${MANUAL_TEX})
        # set to full path
        set(MANUAL_BUILD_PATH ${PROJECT_BINARY_DIR}/${MANUAL_BUILD_PATH})

        if(NOT EXISTS ${PROJECT_SOURCE_DIR}/doc/${MANUAL_TEX})
            message(
                FATAL_ERROR
                    "LaTeX of manual for ${PROJECT_NAME} is not in ${PROJECT_SOURCE_DIR}/doc"
                )
        endif()

        configure_file(${PROJECT_SOURCE_DIR}/doc/${MANUAL_TEX}
                       ${MANUAL_BUILD_PATH}/${MANUAL_NAME} COPYONLY)

        foreach(_file ${EXTRA_FILES_TO_COPY})
            configure_file(${PROJECT_SOURCE_DIR}/doc/${_file}
                           ${MANUAL_BUILD_PATH}/${_file} COPYONLY)
        endforeach()

        add_custom_target(
            man
            ${PDFLATEX} "${MANUAL_NAME}"
            WORKING_DIRECTORY ${MANUAL_BUILD_PATH})
    endif()
endfunction()
