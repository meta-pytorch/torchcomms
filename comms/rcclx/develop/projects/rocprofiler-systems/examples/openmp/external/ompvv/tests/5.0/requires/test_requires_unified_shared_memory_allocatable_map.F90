!===--- test_requires_unified_shared_memory_allocatable_map.F90 --------------------------===//
! 
! OpenMP API Version 5.0 Nov 2018
!
! This test checks for unified shared memory of an array that is allocated  
! using allocatable and that is accessed from host and device with the same pointer.
!
! The mapping of a pointer under shared memory should allow for the pointer to
! have the same value as in the host.
!
!===------------------------------------------------------------------------------===//

#define OMPVV_MODULE_REQUIRES_LINE !$omp requires unified_shared_memory
#include "ompvv.F90"

#define N 1024

PROGRAM test_requires_unified_shared_memory_allocatable_map
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none
   LOGICAL:: isOffloading

!$omp requires unified_shared_memory

  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)

  OMPVV_WARNING_IF(.NOT. isOffloading, "With no offloading, unified shared memory is guaranteed due to host execution")

   OMPVV_TEST_VERBOSE(unified_shared_memory_allocatable_map() .NE. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION unified_shared_memory_allocatable_map()
    INTEGER:: errors, i
    INTEGER, ALLOCATABLE:: anArray(:)
    INTEGER:: ERR

    OMPVV_INFOMSG("Unified shared memory testing - Array on allocatable")

    errors = 0

    ALLOCATE(anArray(N),  STAT=ERR)

    IF( ERR /= 0 ) THEN
      OMPVV_ERROR("Memory was not properly allocated")
      OMPVV_RETURN(ERR)
    END IF

    DO i = 1, N
      anArray(i) = i
    END DO

    ! Modify in the device
    !$omp target map(anArray)
    DO i = 1, N
      anArray(i) = anArray(i) + 10
    END DO
    !$omp end target

    DO i = 1, N
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArray(i) .NE. (i + 10))
      IF (errors .NE. 0) THEN
        exit
      END IF
    END DO

    DEALLOCATE(anArray)
    unified_shared_memory_allocatable_map = errors
  END FUNCTION unified_shared_memory_allocatable_map
END PROGRAM test_requires_unified_shared_memory_allocatable_map
