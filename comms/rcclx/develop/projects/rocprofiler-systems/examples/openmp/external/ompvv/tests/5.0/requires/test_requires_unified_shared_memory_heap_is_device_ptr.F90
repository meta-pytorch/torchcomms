!===--- test_requires_unified_shared_memory_heap_is_device_ptr.F90 --------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks for unified shared memory of an array that is allocated on 
! the heap and that is accessed from host and device with the same pointer.
!
! To guarantee the use of the device pointer, we use the is_device_ptr clause.
!
!===------------------------------------------------------------------------------===//

#define OMPVV_MODULE_REQUIRES_LINE !$omp requires unified_shared_memory
#include "ompvv.F90"

#define N 1024

PROGRAM test_requires_unified_shared_memory_heap_is_device_ptr
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none
   LOGICAL:: isOffloading

!$omp requires unified_shared_memory

  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)

  OMPVV_WARNING_IF(.NOT. isOffloading, "With no offloading, unified shared memory is guaranteed due to host execution")

  OMPVV_TEST_VERBOSE(unified_shared_memory_heap_is_device_ptr() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  SUBROUTINE test_is_device_ptr(anArrayDummy, anArrayCopyDummy)
    INTEGER, DIMENSION(N):: anArrayDummy
    INTEGER, DIMENSION(N):: anArrayCopyDummy
    INTEGER:: i

    ! Modify in the device
    !$omp target is_device_ptr(anArrayDummy)
    DO i = 1, N
      anArrayDummy(i) = anArrayDummy(i) + 10
    END DO
    !$omp end target

    ! Modify again on the host
    DO i = 1, N
      anArrayDummy(i) = anArrayDummy(i) + 10
    END DO

    ! Get the value the device is seeing
    !$omp target is_device_ptr(anArrayDummy)
    DO i = 1, N
      anArrayCopyDummy(i) = anArrayDummy(i)
    END DO
    !$omp end target
  END SUBROUTINE test_is_device_ptr

  INTEGER FUNCTION unified_shared_memory_heap_is_device_ptr()
    INTEGER:: errors, i, ERR
    INTEGER, TARGET, ALLOCATABLE:: anArray(:)
    INTEGER, DIMENSION(N):: anArrayCopy

    OMPVV_INFOMSG("Unified shared memory testing - Array on heap")

    errors = 0

    ALLOCATE(anArray(N), STAT=ERR)

    IF( .NOT. ALLOCATED(anArray) ) THEN
       OMPVV_ERROR("Memory was not properly allocated")
       OMPVV_RETURN(ERR)
    END IF

    DO i = 1, N
      anArray(i) = i
      anArrayCopy(i) = 0
    END DO

    CALL test_is_device_ptr(anArray, anArrayCopy)

    DO i = 1, N
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArray(i) .NE. (i + 20))
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArrayCopy(i) .NE. (i + 20))
      IF (errors .NE. 0) THEN
        exit
      END IF
    END DO
    
    DEALLOCATE(anArray)
    unified_shared_memory_heap_is_device_ptr = errors
  END FUNCTION unified_shared_memory_heap_is_device_ptr
END PROGRAM test_requires_unified_shared_memory_heap_is_device_ptr
      
   
