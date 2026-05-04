!===--- test_requires_unified_shared_memory_static_map.F90 ------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! We request the use of unified_shared_memory in this program.
! Checking for static arrays. The array is global and then accessed through 
! a pointer from the host and the device.
!
! We use the map clause on the pointer, with zero length array, to use the host
! pointer on the device.
!
!===------------------------------------------------------------------------------===//

#define OMPVV_MODULE_REQUIRES_LINE !$omp requires unified_shared_memory
#include "ompvv.F90"

#define N 1024

PROGRAM test_requires_unified_shared_memory_static_map
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none
   LOGICAL:: isOffloading
   ! STATIC ARRAY
   INTEGER, TARGET, DIMENSION(N):: anArray

!$omp requires unified_shared_memory

  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)

  OMPVV_WARNING_IF(.NOT. isOffloading, "With no offloading, unified shared memory is guaranteed due to host execution")

  OMPVV_TEST_VERBOSE(unified_shared_memory_static_map() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION unified_shared_memory_static_map()
    INTEGER:: errors, i
    INTEGER, DIMENSION(N):: anArrayCopy
    INTEGER, POINTER:: aPtr(:)

    OMPVV_INFOMSG("Unified shared memory testing - Static Array")

    errors = 0
    aPtr => anArray

    DO i = 1, N
      anArray(i) = i
      anArrayCopy(i) = 0
    END DO

    ! Test for writes to this varriable from device
    !$omp target map(aPtr) 
    DO i = 1, N
      aPtr(i) = aPtr(i) + 10
    END DO
    !$omp end target

    ! Modify again on the host
    DO i = 1, N
      aPtr(i) = aPtr(i) + 10
    END DO

    ! Test for reads to this variable from device
    !$omp target map(aPtr)
    DO i = 1, N
      anArrayCopy(i) = aPtr(i)
    END DO
    !$omp end target

    DO i = 1, N
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArray(i) .NE. (i + 20))
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArrayCopy(i) .NE. (i + 20))
      IF (errors .NE. 0) THEN
        exit
      END IF
    END DO
    
    unified_shared_memory_static_map = errors
  END FUNCTION unified_shared_memory_static_map
END PROGRAM test_requires_unified_shared_memory_static_map
      
   
