!//===------ test_atomic_num_hint_device.F90 ------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
!  This test checks if atomic hints passed by enum value
!  are processed correctly for device execution. If the sync hint is not
!  yet defined in the specification, it defaults to 
!  omp_sync_hint_none (0x0). 
!
!//===---------------------------------------------------------------------===//

#include "ompvv.F90"


PROGRAM test_atomic_num_hint_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_VERBOSE(test_atomic_with_used_enum_value() .ne. 0)
  OMPVV_TEST_VERBOSE(test_atomic_with_unused_enum_value() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_atomic_with_used_enum_value()
    INTEGER:: x, num_threads, errors

    errors = 0
    x = 0
    num_threads = -1

    OMPVV_INFOMSG("test_atomic_with_used_enum_value")

    !$omp target map(tofrom: num_threads,x) 
    !$omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE) default(shared)
       IF (omp_get_thread_num() .EQ. 0) THEN
          num_threads = omp_get_num_threads()
       END IF
       !$omp atomic hint(4) ! corrosponds to omp_sync_hint_nonspeculative
          x = x + 1
       !$omp end atomic 
    !$omp end parallel
    !$omp end target

    OMPVV_ERROR_IF(num_threads .LT. 0, "Test ran with invalid number of threads (less than zero)")
    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so the results are not conclusive")

    OMPVV_TEST_AND_SET_VERBOSE(errors, x .NE. num_threads)

    test_atomic_with_used_enum_value = errors
  END FUNCTION test_atomic_with_used_enum_value

  INTEGER FUNCTION test_atomic_with_unused_enum_value()
    INTEGER:: x, num_threads, errors

    errors = 0
    x = 0
    num_threads = -1

    OMPVV_INFOMSG("test_atomic_with_unused_enum_value")

    !$omp target map(tofrom: num_threads,x) 
    !$omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE) default(shared)
       IF (omp_get_thread_num() .EQ. 0) THEN
          num_threads = omp_get_num_threads()
       END IF
       !$omp atomic hint(4132) ! As of OMP Spec 5.0 only values till 0x8 have been taken
          x = x + 1
       !$omp end atomic 
    !$omp end parallel
    !$omp end  target

    OMPVV_ERROR_IF(num_threads .LT. 0, "Test ran with invalid number of threads (less than zero)")
    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so the results are not conclusive")

    OMPVV_TEST_AND_SET_VERBOSE(errors, x .NE. num_threads)

    test_atomic_with_unused_enum_value = errors
  END FUNCTION test_atomic_with_unused_enum_value

END PROGRAM test_atomic_num_hint_device
