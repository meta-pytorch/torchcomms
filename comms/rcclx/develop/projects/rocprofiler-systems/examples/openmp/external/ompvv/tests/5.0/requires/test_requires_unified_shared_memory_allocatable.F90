!===--- test_requires_unified_shared_memory_allocatable.F90 ------------------------------===//
! 
! OpenMP API Version 5.0 Nov 2018
!
! This test checks for unified shared memory support of an allocatable array.
!
!===------------------------------------------------------------------------------===//

#define OMPVV_MODULE_REQUIRES_LINE !$omp requires unified_shared_memory
#include "ompvv.F90"

#define N 1024

PROGRAM test_requires_unified_shared_memory_allocatable
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

!$omp requires unified_shared_memory

   OMPVV_TEST_OFFLOADING
   OMPVV_TEST_VERBOSE(unified_shared_memory_allocatable() .NE. 0)
   OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION unified_shared_memory_allocatable()
    INTEGER:: errors, i
    INTEGER, ALLOCATABLE:: anArray(:)
    INTEGER, ALLOCATABLE:: anArrayCopy(:)

    OMPVV_INFOMSG("Unified shared memory testing - Array on allocatable")

    errors = 0

    ALLOCATE(anArray(N), anArrayCopy(N))

    OMPVV_ERROR_IF(.NOT. ALLOCATED(anArray), "Memory was not properly allocated")
    OMPVV_ERROR_IF(.NOT. ALLOCATED(anArrayCopy), "Memory was not properly allocated")

    DO i = 1, N
      anArray(i) = i
      anArrayCopy(i) = 0
    END DO

    ! Modify in the device
    !$omp target
    DO i = 1, N
      anArray(i) = anArray(i) + 10
    END DO
    !$omp end target

    ! Modify again on the host
    DO i = 1, N
      anArray(i) = anArray(i) + 10
    END DO

    ! Get the value the device is seeing
    !$omp target
    DO i = 1, N
      anArrayCopy(i) = anArray(i)
    END DO
    !$omp end target

    DO i = 1, N
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArray(i) .NE. (i + 20))
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArrayCopy(i) .NE. (i + 20))
      IF (errors .NE. 0) THEN
        exit
      END IF
    END DO

    DEALLOCATE(anArray, anArrayCopy)
    unified_shared_memory_allocatable = errors
  END FUNCTION unified_shared_memory_allocatable
END PROGRAM test_requires_unified_shared_memory_allocatable
