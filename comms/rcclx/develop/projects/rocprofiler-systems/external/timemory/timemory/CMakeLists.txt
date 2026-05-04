# ########################################################################################
#
# timemory (Python)
#
# ########################################################################################

if(SKBUILD)
    set(CMAKE_INSTALL_PYTHONDIR ${CMAKE_INSTALL_PREFIX})
endif()

set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})
if(NOT IS_ABSOLUTE "${_PYLIB}")
    set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
endif()
file(RELATIVE_PATH LIB_RELPATH "${_PYLIB}"
     "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# ----------------------------------------------------------------------------
#
# Always install these files since timemory has python scripts which do not depend on the
# library being built
#
# ----------------------------------------------------------------------------

set(PYTHON_SUBMODULE_FOLDERS
    analyze
    api
    ert
    mpi
    mpi_support
    plotting
    profiler
    roofline
    util
    bundle
    component
    hardware_counters
    test
    trace
    region
    tools
    libs)

file(GLOB PYTHON_SUBMODULE_FILES ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/*.py)
string(REPLACE "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/" "" PYTHON_SUBMODULE_FILES
               "${PYTHON_SUBMODULE_FILES}")

foreach(_SUBMODULE ${PYTHON_SUBMODULE_FOLDERS})
    set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${_SUBMODULE})
    file(GLOB_RECURSE _FILES ${SOURCE_DIR}/*.py ${SOURCE_DIR}/*.py.in)

    # copy over the python file
    foreach(_FILE ${_FILES})
        get_filename_component(_DIR "${_FILE}" DIRECTORY)
        get_filename_component(_FILE "${_FILE}" NAME_WE)
        string(REPLACE "${PROJECT_SOURCE_DIR}/" "" _DIR "${_DIR}")
        if(EXISTS ${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py)
            configure_file(${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py
                           ${PROJECT_BINARY_DIR}/${_DIR}/${_FILE}.py @ONLY)
        elseif(EXISTS ${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py.in)
            configure_file(${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py.in
                           ${PROJECT_BINARY_DIR}/${_DIR}/${_FILE}.py @ONLY)
        endif()
        if(CMAKE_INSTALL_PYTHONDIR AND timemory_MAIN_PROJECT)
            install(
                FILES ${PROJECT_BINARY_DIR}/${_DIR}/${_FILE}.py
                DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${_DIR}
                OPTIONAL)
        endif()
    endforeach()
endforeach()

# ----------------------------------------------------------------------------
# MPI info
#
set(MPI_EXE_INFO "MPI not supported")
set(MPI_C_INFO "MPI not supported")
set(MPI_CXX_INFO "MPI not supported")
if(TIMEMORY_USE_MPI AND MPI_FOUND)

    execute_process(
        COMMAND ${MPIEXEC_EXECUTABLE} --version
        OUTPUT_VARIABLE MPI_EXE_INFO
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

    execute_process(
        COMMAND ${MPI_C_COMPILER} --version
        OUTPUT_VARIABLE MPI_C_INFO
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

    execute_process(
        COMMAND ${MPI_CXX_COMPILER} --version
        OUTPUT_VARIABLE MPI_CXX_INFO
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

endif()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mpi_support/mpi_exe_info.txt "${MPI_EXE_INFO}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mpi_support/mpi_c_info.txt "${MPI_C_INFO}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mpi_support/mpi_cxx_info.txt "${MPI_CXX_INFO}")

if(CMAKE_INSTALL_PYTHONDIR AND timemory_MAIN_PROJECT)
    foreach(_TYPE exe c cxx)
        install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/mpi_support/mpi_${_TYPE}_info.txt
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/mpi_support
            OPTIONAL)
    endforeach()
endif()

file(GLOB BIN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/bin/*)
foreach(_FILE ${BIN_FILES})
    get_filename_component(_NAME ${_FILE} NAME)
    configure_file(${_FILE} ${CMAKE_CURRENT_BINARY_DIR}/bin/${_NAME} COPYONLY)
    if(CMAKE_INSTALL_PYTHONDIR AND timemory_MAIN_PROJECT)
        install(
            PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/bin/${_NAME}
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/bin
            OPTIONAL)
    endif()
endforeach()

# ----------------------------------------------------------------------------
# Install the python stuff
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/__init__.py @ONLY)

configure_file(${PROJECT_SOURCE_DIR}/pytest.ini ${PROJECT_BINARY_DIR}/pytest.ini COPYONLY)

if(CMAKE_INSTALL_PYTHONDIR AND timemory_MAIN_PROJECT)
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}
        OPTIONAL)
    if(SKBUILD)
        install(
            FILES ${PROJECT_BINARY_DIR}/pytest.ini
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
            OPTIONAL)
    endif()
endif()

foreach(PYLIB_SUBMODULE ${PYTHON_SUBMODULE_FILES})
    get_filename_component(PYLIB_SUBMODULE "${PYLIB_SUBMODULE}" NAME_WE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PYLIB_SUBMODULE}.py
                   ${CMAKE_CURRENT_BINARY_DIR}/${PYLIB_SUBMODULE}.py @ONLY)

    if(CMAKE_INSTALL_PYTHONDIR AND timemory_MAIN_PROJECT)
        install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/${PYLIB_SUBMODULE}.py
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}
            OPTIONAL)
    endif()
endforeach()

# ----------------------------------------------------------------------------
# console scripts
#
if(SKBUILD)
    return()
elseif(TIMEMORY_USE_PYTHON AND CMAKE_INSTALL_PYTHONDIR)
    foreach(_FILE setup.py setup.cfg VERSION requirements.txt .requirements/build.txt
                  .requirements/runtime.txt .requirements/mpi_runtime.txt)
        configure_file(${PROJECT_SOURCE_DIR}/${_FILE} ${PROJECT_BINARY_DIR}/${_FILE}
                       COPYONLY)
    endforeach()
    if(PYTHON_EXECUTABLE)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} setup.py dist_info --dist-info
            OUTPUT_VARIABLE _OUT
            RESULT_VARIABLE _RET
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        set(_DIST_DIR ${PROJECT_BINARY_DIR}/timemory-${TIMEMORY_VERSION}.dist-info)
        if(NOT EXISTS ${_DIST_DIR})
            set(_DIST_DIR ${PROJECT_BINARY_DIR}/timemory.dist-info)
        endif()
        if(EXISTS ${_DIST_DIR} AND IS_DIRECTORY ${_DIST_DIR})
            configure_file(${PROJECT_SOURCE_DIR}/LICENSE ${_DIST_DIR}/LICENSE.txt
                           COPYONLY)
            install(
                DIRECTORY ${_DIST_DIR}
                DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
                OPTIONAL)
        endif()
    endif()
endif()

function(TIMEMORY_PYTHON_CONSOLE_SCRIPT SCRIPT_NAME SCRIPT_SUBMODULE SCRIPT_FUNCTION)
    set(CONSOLE_EXTRA "${ARGN}")

    if(NOT SKBUILD)
        # set(TIMEMORY_REQUIRES_PREFIX "# ")
    endif()

    configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/console-script.py.in
                   ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME} @ONLY)

    if(CMAKE_INSTALL_PYTHONDIR AND timemory_MAIN_PROJECT)
        install(
            PROGRAMS ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            OPTIONAL)
    endif()

    if(NOT WIN32 AND (TIMEMORY_BUILD_TESTING OR TIMEMORY_BUILD_PYTHON))
        add_test(
            NAME ${SCRIPT_NAME}-console-script-test
            COMMAND ${PROJECT_BINARY_DIR}/bin/${SCRIPT_NAME} --help
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        set_tests_properties(
            ${SCRIPT_NAME}-console-script-test
            PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}:$ENV{PYTHONPATH}")
    endif()
endfunction()

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python3")
endif()

timemory_python_console_script("timemory-plotter" "timemory.plotting" "try_plot")
timemory_python_console_script(
    "timemory-roofline" "timemory.roofline" "try_plot"
    "import os\n    os.environ[\"TIMEMORY_CXX_PLOT_MODE\"] = \"ON\"")
timemory_python_console_script("timemory-analyze" "timemory.analyze" "try_analyze")

if(TIMEMORY_USE_PYTHON)
    timemory_python_console_script("timemory-python-profiler" "timemory.profiler" "main")
    timemory_python_console_script("timemory-python-trace" "timemory.trace" "main")
endif()
