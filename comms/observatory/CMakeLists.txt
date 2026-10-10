# libobservatory.so.1 is built once, by comms/utils/Makefile, and shipped by
# the commsutils package beside libcommsutils.so.1 (see README.md: one copy per
# process). This tree never builds a second copy: libctran/libmccl link the
# prebuilt one from COMMSUTILS_LIBRARY_DIR (the commsutils wheel's lib dir in a
# wheel build), else the environment's lib dir, where the conda package
# installs it.
find_library(OBSERVATORY_LIBRARY
    NAMES observatory
    HINTS ENV COMMSUTILS_LIBRARY_DIR ${CONDA_LIB}
)
if(NOT OBSERVATORY_LIBRARY)
    message(FATAL_ERROR
        "libobservatory.so not found (COMMSUTILS_LIBRARY_DIR=$ENV{COMMSUTILS_LIBRARY_DIR}, "
        "CONDA_LIB=${CONDA_LIB}); it is installed by the commsutils package.")
endif()
add_library(observatory SHARED IMPORTED GLOBAL)
set_target_properties(observatory PROPERTIES
    IMPORTED_LOCATION "${OBSERVATORY_LIBRARY}"
)
# Consumers get ${ROOT} from their own build; the headers are the ones in this
# source tree, the same revision the library was built from.
