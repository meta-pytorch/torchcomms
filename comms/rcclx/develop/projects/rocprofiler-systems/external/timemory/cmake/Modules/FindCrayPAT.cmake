# ------------------------------------------------------------------------------#
#
# Finds headers and libraries for CrayPAT instrumentation
#
# ------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)

# ------------------------------------------------------------------------------#

set(CrayPAT_HINTS "$ENV{CRAYPAT_ROOT}")

# ------------------------------------------------------------------------------#

find_path(
    CrayPAT_INCLUDE_DIR
    NAMES pat_api.h
    PATH_SUFFIXES include
    HINTS ${CrayPAT_ROOT_DIR} ${CrayPAT_HINTS}
    PATHS ${CrayPAT_ROOT_DIR} ${CrayPAT_HINTS})

# ------------------------------------------------------------------------------#

find_library(
    CrayPAT_LIBRARY
    NAMES _pat_rt
    PATH_SUFFIXES lib64 lib lib32
    HINTS ${CrayPAT_ROOT_DIR} ${CrayPAT_HINTS}
    PATHS ${CrayPAT_ROOT_DIR} ${CrayPAT_HINTS})

# ------------------------------------------------------------------------------#

if(CrayPAT_INCLUDE_DIR)
    set(CrayPAT_INCLUDE_DIRS ${CrayPAT_INCLUDE_DIR})
endif()

# ------------------------------------------------------------------------------#

if(CrayPAT_LIBRARY)
    set(CrayPAT_LIBRARIES ${CrayPAT_LIBRARY})
    get_filename_component(CrayPAT_LIBRARY_DIR "${CrayPAT_LIBRARY}" PATH CACHE)
endif()

# ------------------------------------------------------------------------------#

if(CrayPAT_LIBRARY_DIR)
    set(CrayPAT_LIBRARY_DIRS ${CrayPAT_LIBRARY_DIR})
endif()

# ------------------------------------------------------------------------------#

set(CrayPAT_AVAILABLE_COMPONENTS
    adios2
    aio
    ampi
    armci
    base
    blacs
    blas
    caf
    chapel
    charm
    comex
    converse
    craymem
    cuda
    curl
    dl
    dmapp
    fabric
    ffio
    fftw
    fftw_cl
    ga
    gmp
    gni
    hbw
    hdf5
    heap
    hip
    hsa
    huge
    jemalloc
    lapack
    lustre
    math
    memkind
    memory
    mpfr
    mpi
    netcdf
    numa
    oacc
    omp
    opencl
    ovhd_null
    ovhd_null_dl
    ovhd_null_inner
    ovhd_null_inner_dl
    ovhd_null_outer
    ovhd_null_outer_dl
    pblas
    petsc
    pgas
    pnetcdf
    pthread
    rt
    scalapack
    shmem
    shmemx
    signal
    spawn
    stdio
    string
    stubs_acc
    stubs_api
    stubs_omp
    stubs_tps
    syscall
    sysfs
    sysio
    upc
    xpmem
    zmq)

# ------------------------------------------------------------------------------#

foreach(_COMP ${CrayPAT_FIND_COMPONENTS})

    if(NOT "${_COMP}" IN_LIST CrayPAT_AVAILABLE_COMPONENTS)
        message(AUTHOR_WARNING "Non-standard CrayPAT component: '${_COMP}'")
    endif()

    find_library(
        CrayPAT_${_COMP}_LIBRARY
        NAMES _pat_${_COMP}
        PATH_SUFFIXES lib64 lib lib32
        HINTS ${CrayPAT_ROOT_DIR} ${CrayPAT_LIBRARY_DIR} ${CrayPAT_HINTS}
        PATHS ${CrayPAT_ROOT_DIR} ${CrayPAT_LIBRARY_DIR} ${CrayPAT_HINTS})

    if(CrayPAT_${_COMP}_LIBRARY)
        list(APPEND CrayPAT_LIBRARIES ${CrayPAT_${_COMP}_LIBRARY})
        get_filename_component(_COMP_DIR "${CrayPAT_${_COMP}_LIBRARY}" PATH)
        list(APPEND CrayPAT_LIBRARY_DIRS ${_COMP_DIR})
    else()
        if(CrayPAT_FIND_REQUIRED_${_COMP})
            list(APPEND _CrayPAT_MISSING_COMPONENTS CrayPAT_${_COMP}_LIBRARY)
        endif()
    endif()
    mark_as_advanced(CrayPAT_${_COMP}_LIBRARY)

endforeach()

# ------------------------------------------------------------------------------#

mark_as_advanced(CrayPAT_INCLUDE_DIR CrayPAT_LIBRARY CrayPAT_LIBRARY_DIR)

find_package_handle_standard_args(
    CrayPAT REQUIRED_VARS CrayPAT_INCLUDE_DIR CrayPAT_LIBRARY CrayPAT_LIBRARY_DIR
                          ${_CrayPAT_MISSING_COMPONENTS})

# ------------------------------------------------------------------------------#
