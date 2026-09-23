# Utilities for find_package
#
include(CMakeParseArguments)

function(TIMEMORY_FIND_STATIC_LIBRARY)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    find_library(${ARGN})
endfunction()

#
# find_root_hints(...) get the realpath to certain files and then walks up the directory
# tree
#
function(TIMEMORY_FIND_ROOT_PATH _OUTPUT_VAR _FNAME)

    if(DEFINED ${_OUTPUT_VAR} AND ${_OUTPUT_VAR})
        return()
    endif()

    # default behavior is local variable in parent scope
    set(_SCOPED ON)
    if("CACHE" IN_LIST ARGN)
        set(_SCOPED OFF)
        list(REMOVE_ITEM ARGN CACHE)
    endif()

    # find the file
    find_file(_FILE_REALPATH ${_FNAME} ${ARGN})

    set(_HINT)
    if(_FILE_REALPATH)
        # get the realpath
        get_filename_component(_FILE_REALPATH_0 "${_FILE_REALPATH}" REALPATH)
        # for the loop
        set(_LAST_VAR _FILE_REALPATH_0)
        # loop over two parent directories
        foreach(_IDX 1 2)
            # this is the current variable
            set(_CURR_VAR _FILE_REALPATH_${_IDX})
            # get the directory of the current variable
            get_filename_component(${_CURR_VAR} "${${_LAST_VAR}}" DIRECTORY)
            # append the hint
            set(_HINT ${${_CURR_VAR}})
            # set the last variable to the parent directory
            set(_LAST_VAR ${_CURR_VAR})
        endforeach()
        # set the variable
        if(_SCOPED)
            set(${_OUTPUT_VAR}
                "${_HINT}"
                PARENT_SCOPE)
        else()
            set(${_OUTPUT_VAR}
                "${_HINT}"
                CACHE STRING "Root path hints to ${_FNAME}")
        endif()
    else()
        set(${_OUTPUT_VAR}
            ${_OUTPUT_VAR}-NOTFOUND
            PARENT_SCOPE)
    endif()

    # remove the cache entry for FIND_FILE
    unset(_FILE_REALPATH CACHE)
endfunction()
