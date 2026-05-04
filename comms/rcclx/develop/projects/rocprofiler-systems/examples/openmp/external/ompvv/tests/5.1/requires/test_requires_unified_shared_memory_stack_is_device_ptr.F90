!===--- test_requires_unified_shared_memory_stack_is_device_ptr.F90 --------------------------===//
!
! OpenMP API Version 5.1 Aug 2021
!
! Testing memory access from host and device to a variable allocated in the stack
! The variable is accessed through a pointer to guarantee that there is no default
! mapping tofrom: of the stack array
!
! We use the is_device_ptr to guarantee using the host pointer is used in the device
! From Version 5.1, a list item that appears in an is_device_ptr clause must be of type C_PTR.
!
!===------------------------------------------------------------------------------===//

#define OMPVV_MODULE_REQUIRES_LINE !$omp requires unified_shared_memory
#include "ompvv.F90"

#define N 1024

PROGRAM test_requires_unified_shared_memory_stack_is_device_ptr
   USE iso_fortran_env
   USE iso_c_binding
   USE ompvv_lib
   USE omp_lib
   implicit none
   LOGICAL:: isOffloading

!$omp requires unified_shared_memory

  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)

  OMPVV_WARNING_IF(.NOT. isOffloading, "With no offloading, unified shared memory is guaranteed due to host execution")

  OMPVV_TEST_VERBOSE(unified_shared_memory_stack_is_device_ptr() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION unified_shared_memory_stack_is_device_ptr()
    INTEGER:: errors, i
    INTEGER, TARGET, DIMENSION(N):: anArray
    INTEGER, DIMENSION(N):: anArrayCopy
    TYPE(C_PTR):: aPtr

    OMPVV_INFOMSG("Unified shared memory testing - Array on stack")

    errors = 0

    aPtr = C_LOC(anArray(1))

    DO i = 1, N
      anArray(i) = i
      anArrayCopy(i) = 0
    END DO

    ! Modify in the device
    !$omp target is_device_ptr(aPtr)
    BLOCK
      INTEGER, POINTER :: anPtr(:)
      call c_f_pointer(aPtr, anPtr, [N])
      DO i = 1, N
        anPtr(i) = anPtr(i) + 10
      END DO
    END BLOCK

    ! Modify again on the host
    DO i = 1, N
      anArray(i) = anArray(i) + 10
    END DO

    ! Get the value the device is seeing
    !$omp target is_device_ptr(aPtr)
    BLOCK
      INTEGER, POINTER :: anPtr(:)
      call c_f_pointer(aPtr, anPtr, [N])
      DO i = 1, N
        anArrayCopy(i) = anPtr(i)
      END DO
    END BLOCK

    DO i = 1, N
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArray(i) .NE. (i + 20))
      OMPVV_TEST_AND_SET_VERBOSE(errors, anArrayCopy(i) .NE. (i + 20))
      IF (errors .NE. 0) THEN
        exit
      END IF
    END DO
    
    unified_shared_memory_stack_is_device_ptr = errors
  END FUNCTION unified_shared_memory_stack_is_device_ptr
END PROGRAM test_requires_unified_shared_memory_stack_is_device_ptr
      
   
