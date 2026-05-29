foreach(_VAR INPUT_LIST_DIR EXE_BINARY_DIR FILE_BINARY_DIR)
    if(NOT DEFINED ${_VAR})
        message(WARNING "Warning! Aborting script because ${_VAR} not provided")
        return()
    endif()
endforeach()

configure_file(${INPUT_LIST_DIR}/generated/timemory-instr.sh
               ${EXE_BINARY_DIR}/timemory-instr COPYONLY)

file(GLOB COLLECTION_FILES "${INPUT_LIST_DIR}/collections/*")
foreach(_FILE ${COLLECTION_FILES})
    get_filename_component(_FNAME "${_FILE}" NAME)
    configure_file(${_FILE} ${FILE_BINARY_DIR}/${_FNAME} COPYONLY)
endforeach()
